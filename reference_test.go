package blake3_test

// A port of the BLAKE3 specification's reference implementation
// (BLAKE3-team/BLAKE3, reference_impl/reference_impl.rs, CC0): the plain
// chunk-by-chunk construction with no SIMD, no eigentrees, no buffering
// beyond one chunk. It is the oracle the optimized Hasher is held to at
// every length and every output offset, independent of the code under test:
// it shares nothing with package guts.

import (
	"encoding/binary"
	"math/bits"
	"slices"
)

const (
	refBlockLen = 64
	refChunkLen = 1024

	refChunkStart        = 1 << 0
	refChunkEnd          = 1 << 1
	refParent            = 1 << 2
	refRoot              = 1 << 3
	refKeyedHash         = 1 << 4
	refDeriveKeyContext  = 1 << 5
	refDeriveKeyMaterial = 1 << 6
)

//nolint:gochecknoglobals // the specification's constants, which an array cannot be
var refIV = [8]uint32{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
}

//nolint:gochecknoglobals // the specification's constants, which an array cannot be
var refPermutation = [16]int{2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}

func refG(state *[16]uint32, a, b, c, d int, mx, my uint32) {
	state[a] = state[a] + state[b] + mx
	state[d] = bits.RotateLeft32(state[d]^state[a], -16)
	state[c] += state[d]
	state[b] = bits.RotateLeft32(state[b]^state[c], -12)
	state[a] = state[a] + state[b] + my
	state[d] = bits.RotateLeft32(state[d]^state[a], -8)
	state[c] += state[d]
	state[b] = bits.RotateLeft32(state[b]^state[c], -7)
}

func refRound(state, m *[16]uint32) {
	// columns
	refG(state, 0, 4, 8, 12, m[0], m[1])
	refG(state, 1, 5, 9, 13, m[2], m[3])
	refG(state, 2, 6, 10, 14, m[4], m[5])
	refG(state, 3, 7, 11, 15, m[6], m[7])
	// diagonals
	refG(state, 0, 5, 10, 15, m[8], m[9])
	refG(state, 1, 6, 11, 12, m[10], m[11])
	refG(state, 2, 7, 8, 13, m[12], m[13])
	refG(state, 3, 4, 9, 14, m[14], m[15])
}

func refPermute(m *[16]uint32) {
	var permuted [16]uint32
	for i := range permuted {
		permuted[i] = m[refPermutation[i]]
	}

	*m = permuted
}

func refCompress(cv *[8]uint32, block [16]uint32, counter uint64, blockLen, flags uint32) [16]uint32 {
	state := [16]uint32{
		cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
		refIV[0], refIV[1], refIV[2], refIV[3],
		uint32(counter), uint32(counter >> 32), blockLen, flags,
	}

	for i := range 7 {
		refRound(&state, &block)

		if i < 6 {
			refPermute(&block)
		}
	}

	for i := range 8 {
		state[i] ^= state[i+8]
		state[i+8] ^= cv[i]
	}

	return state
}

func refWords(b []byte) (words [16]uint32) {
	var block [refBlockLen]byte
	copy(block[:], b)

	for i := range words {
		words[i] = binary.LittleEndian.Uint32(block[4*i:])
	}

	return words
}

func refFirst8(words [16]uint32) (cv [8]uint32) {
	copy(cv[:], words[:8])

	return cv
}

// refOutput is a node ready to be a chaining value or, with the root flag,
// the start of the output stream.
type refOutput struct {
	cv       [8]uint32
	block    [16]uint32
	counter  uint64
	blockLen uint32
	flags    uint32
}

func (o refOutput) chainingValue() [8]uint32 {
	return refFirst8(refCompress(&o.cv, o.block, o.counter, o.blockLen, o.flags))
}

// rootBytes is the output stream from byte offset off, n bytes: block
// counter off/64, the first off%64 bytes of it dropped.
func (o refOutput) rootBytes(off uint64, n int) []byte {
	out := make([]byte, 0, n+refBlockLen)
	counter := off / refBlockLen

	for len(out) < n+int(off%refBlockLen) {
		words := refCompress(&o.cv, o.block, counter, o.blockLen, o.flags|refRoot)
		for _, w := range words {
			out = binary.LittleEndian.AppendUint32(out, w)
		}

		counter++
	}

	return out[off%refBlockLen:][:n]
}

type refChunkState struct {
	cv               [8]uint32
	counter          uint64
	block            []byte
	blocksCompressed int
	flags            uint32
}

func (c *refChunkState) startFlag() uint32 {
	if c.blocksCompressed == 0 {
		return refChunkStart
	}

	return 0
}

func (c *refChunkState) length() int {
	return refBlockLen*c.blocksCompressed + len(c.block)
}

func (c *refChunkState) update(input []byte) {
	for len(input) > 0 {
		if len(c.block) == refBlockLen {
			c.cv = refFirst8(refCompress(&c.cv, refWords(c.block), c.counter, refBlockLen, c.flags|c.startFlag()))
			c.blocksCompressed++
			c.block = c.block[:0]
		}

		take := min(refBlockLen-len(c.block), len(input))
		c.block = append(c.block, input[:take]...)
		input = input[take:]
	}
}

func (c *refChunkState) output() refOutput {
	return refOutput{
		cv:       c.cv,
		block:    refWords(c.block),
		counter:  c.counter,
		blockLen: uint32(len(c.block)),
		flags:    c.flags | c.startFlag() | refChunkEnd,
	}
}

func refParentOutput(left, right, key [8]uint32, flags uint32) refOutput {
	var block [16]uint32
	copy(block[:8], left[:])
	copy(block[8:], right[:])

	return refOutput{cv: key, block: block, counter: 0, blockLen: refBlockLen, flags: refParent | flags}
}

// refHasher is the reference's incremental hasher.
type refHasher struct {
	chunk refChunkState
	key   [8]uint32
	stack [][8]uint32
	flags uint32
}

func newRefHasherInternal(key [8]uint32, flags uint32) *refHasher {
	return &refHasher{chunk: refChunkState{cv: key, flags: flags}, key: key, flags: flags}
}

func newRefHasher() *refHasher { return newRefHasherInternal(refIV, 0) }

func newRefKeyed(key [32]byte) *refHasher {
	return newRefHasherInternal(refKeyWords(key), refKeyedHash)
}

func newRefDeriveKey(context string) *refHasher {
	ctx := newRefHasherInternal(refIV, refDeriveKeyContext)
	ctx.update([]byte(context))

	var contextKey [32]byte
	copy(contextKey[:], ctx.finalize(0, 32))

	return newRefHasherInternal(refKeyWords(contextKey), refDeriveKeyMaterial)
}

func refKeyWords(key [32]byte) (words [8]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(key[4*i:])
	}

	return words
}

func (h *refHasher) addChunkChainingValue(cv [8]uint32, totalChunks uint64) {
	for totalChunks&1 == 0 {
		cv = refParentOutput(h.stack[len(h.stack)-1], cv, h.key, h.flags).chainingValue()
		h.stack = h.stack[:len(h.stack)-1]
		totalChunks >>= 1
	}

	h.stack = append(h.stack, cv)
}

func (h *refHasher) update(input []byte) {
	for len(input) > 0 {
		if h.chunk.length() == refChunkLen {
			cv := h.chunk.output().chainingValue()
			total := h.chunk.counter + 1
			h.addChunkChainingValue(cv, total)
			h.chunk = refChunkState{cv: h.key, counter: total, flags: h.flags}
		}

		take := min(refChunkLen-h.chunk.length(), len(input))
		h.chunk.update(input[:take])
		input = input[take:]
	}
}

// root is the root node; what the output stream is derived from.
func (h *refHasher) root() refOutput {
	output := h.chunk.output()
	for _, cv := range slices.Backward(h.stack) {
		output = refParentOutput(cv, output.chainingValue(), h.key, h.flags)
	}

	return output
}

// finalize is n bytes of the output stream from offset off.
func (h *refHasher) finalize(off uint64, n int) []byte {
	return h.root().rootBytes(off, n)
}

// refHash is the reference's unkeyed hash of data, n bytes of it.
func refHash(data []byte, n int) []byte {
	h := newRefHasher()
	h.update(data)

	return h.finalize(0, n)
}
