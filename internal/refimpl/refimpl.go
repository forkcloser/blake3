// Package refimpl is a port of the BLAKE3 specification's reference
// implementation (BLAKE3-team/BLAKE3, reference_impl/reference_impl.rs,
// CC0): the plain chunk-by-chunk construction with no SIMD, no eigentrees,
// no buffering beyond one chunk. It is the oracle the module's tests hold
// the optimized code to, at every length and every output offset, and it
// shares nothing with package guts.
package refimpl

import (
	"encoding/binary"
	"math/bits"
	"slices"
)

// The specification's sizes and flags.
const (
	BlockLen = 64
	ChunkLen = 1024

	FlagChunkStart        = 1 << 0
	FlagChunkEnd          = 1 << 1
	FlagParent            = 1 << 2
	FlagRoot              = 1 << 3
	FlagKeyedHash         = 1 << 4
	FlagDeriveKeyContext  = 1 << 5
	FlagDeriveKeyMaterial = 1 << 6
)

// IV is the specification's initialization vector.
//
//nolint:gochecknoglobals // the specification's constants, which an array cannot be
var IV = [8]uint32{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
}

//nolint:gochecknoglobals // the specification's constants, which an array cannot be
var permutation = [16]int{2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}

func g(state *[16]uint32, a, b, c, d int, mx, my uint32) {
	state[a] = state[a] + state[b] + mx
	state[d] = bits.RotateLeft32(state[d]^state[a], -16)
	state[c] += state[d]
	state[b] = bits.RotateLeft32(state[b]^state[c], -12)
	state[a] = state[a] + state[b] + my
	state[d] = bits.RotateLeft32(state[d]^state[a], -8)
	state[c] += state[d]
	state[b] = bits.RotateLeft32(state[b]^state[c], -7)
}

func round(state, m *[16]uint32) {
	// columns
	g(state, 0, 4, 8, 12, m[0], m[1])
	g(state, 1, 5, 9, 13, m[2], m[3])
	g(state, 2, 6, 10, 14, m[4], m[5])
	g(state, 3, 7, 11, 15, m[6], m[7])
	// diagonals
	g(state, 0, 5, 10, 15, m[8], m[9])
	g(state, 1, 6, 11, 12, m[10], m[11])
	g(state, 2, 7, 8, 13, m[12], m[13])
	g(state, 3, 4, 9, 14, m[14], m[15])
}

func permute(m *[16]uint32) {
	var permuted [16]uint32
	for i := range permuted {
		permuted[i] = m[permutation[i]]
	}

	*m = permuted
}

// Compress is the specification's compression function: seven rounds over
// the state built from cv, the counter, the block length and the flags, then
// the fold.
func Compress(cv *[8]uint32, block [16]uint32, counter uint64, blockLen, flags uint32) [16]uint32 {
	state := [16]uint32{
		cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
		IV[0], IV[1], IV[2], IV[3],
		// #nosec G115 -- the specification's split of the counter into its two words
		uint32(counter), uint32(counter >> 32), blockLen, flags,
	}

	for i := range 7 {
		round(&state, &block)

		if i < 6 {
			permute(&block)
		}
	}

	for i := range 8 {
		state[i] ^= state[i+8]
		state[i+8] ^= cv[i]
	}

	return state
}

// Words is a block's 64 bytes as 16 little-endian words, zero-padded.
func Words(b []byte) (words [16]uint32) {
	var block [BlockLen]byte
	copy(block[:], b)

	for i := range words {
		words[i] = binary.LittleEndian.Uint32(block[4*i:])
	}

	return words
}

func first8(words [16]uint32) (cv [8]uint32) {
	copy(cv[:], words[:8])

	return cv
}

// Output is a node ready to be a chaining value or, with the root flag,
// the start of the output stream.
type Output struct {
	cv       [8]uint32
	block    [16]uint32
	counter  uint64
	blockLen uint32
	flags    uint32
}

// ChainingValue is the node compressed, its first eight words.
func (o Output) ChainingValue() [8]uint32 {
	return first8(Compress(&o.cv, o.block, o.counter, o.blockLen, o.flags))
}

// RootBytes is the output stream from byte offset off, n bytes: block
// counter off/64, the first off%64 bytes of it dropped.
func (o Output) RootBytes(off uint64, n int) []byte {
	out := make([]byte, 0, n+BlockLen)
	counter := off / BlockLen

	for len(out) < n+int(off%BlockLen) {
		words := Compress(&o.cv, o.block, counter, o.blockLen, o.flags|FlagRoot)
		for _, w := range words {
			out = binary.LittleEndian.AppendUint32(out, w)
		}

		counter++
	}

	return out[off%BlockLen:][:n]
}

// ChunkState is one chunk being hashed: Cv the key it started from, Counter
// its chunk counter, Flags the mode's.
type ChunkState struct {
	Cv               [8]uint32
	Counter          uint64
	Flags            uint32
	block            []byte
	blocksCompressed int
}

func (c *ChunkState) startFlag() uint32 {
	if c.blocksCompressed == 0 {
		return FlagChunkStart
	}

	return 0
}

func (c *ChunkState) length() int {
	return BlockLen*c.blocksCompressed + len(c.block)
}

// Update feeds input to the chunk, compressing each full block as the next
// one arrives.
func (c *ChunkState) Update(input []byte) {
	for len(input) > 0 {
		if len(c.block) == BlockLen {
			c.Cv = first8(Compress(&c.Cv, Words(c.block), c.Counter, BlockLen, c.Flags|c.startFlag()))
			c.blocksCompressed++
			c.block = c.block[:0]
		}

		take := min(BlockLen-len(c.block), len(input))
		c.block = append(c.block, input[:take]...)
		input = input[take:]
	}
}

// Output is the chunk's final node: its last block with the chunk-end flag.
func (c *ChunkState) Output() Output {
	return Output{
		cv:       c.Cv,
		block:    Words(c.block),
		counter:  c.Counter,
		blockLen: uint32(len(c.block)), // #nosec G115 -- at most BlockLen, by Update's take
		flags:    c.Flags | c.startFlag() | FlagChunkEnd,
	}
}

// ParentOutput is a parent node over two chaining values.
func ParentOutput(left, right, key [8]uint32, flags uint32) Output {
	var block [16]uint32
	copy(block[:8], left[:])
	copy(block[8:], right[:])

	return Output{cv: key, block: block, counter: 0, blockLen: BlockLen, flags: FlagParent | flags}
}

// Hasher is the reference's incremental hasher.
type Hasher struct {
	chunk ChunkState
	key   [8]uint32
	stack [][8]uint32
	flags uint32
}

func newHasher(key [8]uint32, flags uint32) *Hasher {
	return &Hasher{chunk: ChunkState{Cv: key, Flags: flags}, key: key, flags: flags}
}

// New is the unkeyed hash.
func New() *Hasher { return newHasher(IV, 0) }

// NewKeyed is the keyed hash under key.
func NewKeyed(key [32]byte) *Hasher {
	return newHasher(KeyWords(key), FlagKeyedHash)
}

// NewDeriveKey is the key-derivation mode under context.
func NewDeriveKey(context string) *Hasher {
	ctx := newHasher(IV, FlagDeriveKeyContext)
	ctx.Update([]byte(context))

	var contextKey [32]byte
	copy(contextKey[:], ctx.Finalize(0, 32))

	return newHasher(KeyWords(contextKey), FlagDeriveKeyMaterial)
}

// KeyWords is a 32-byte key as eight little-endian words.
func KeyWords(key [32]byte) (words [8]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(key[4*i:])
	}

	return words
}

func (h *Hasher) addChunkChainingValue(cv [8]uint32, totalChunks uint64) {
	for totalChunks&1 == 0 {
		cv = ParentOutput(h.stack[len(h.stack)-1], cv, h.key, h.flags).ChainingValue()
		h.stack = h.stack[:len(h.stack)-1]
		totalChunks >>= 1
	}

	h.stack = append(h.stack, cv)
}

// Update feeds input to the hash, a chunk at a time.
func (h *Hasher) Update(input []byte) {
	for len(input) > 0 {
		if h.chunk.length() == ChunkLen {
			cv := h.chunk.Output().ChainingValue()
			total := h.chunk.Counter + 1
			h.addChunkChainingValue(cv, total)
			h.chunk = ChunkState{Cv: h.key, Counter: total, Flags: h.flags}
		}

		take := min(ChunkLen-h.chunk.length(), len(input))
		h.chunk.Update(input[:take])
		input = input[take:]
	}
}

// Root is the root node: what the output stream is derived from.
func (h *Hasher) Root() Output {
	output := h.chunk.Output()
	for _, cv := range slices.Backward(h.stack) {
		output = ParentOutput(cv, output.ChainingValue(), h.key, h.flags)
	}

	return output
}

// Finalize is n bytes of the output stream from offset off.
func (h *Hasher) Finalize(off uint64, n int) []byte {
	return h.Root().RootBytes(off, n)
}

// Hash is the reference's unkeyed hash of data, n bytes of it.
func Hash(data []byte, n int) []byte {
	h := New()
	h.Update(data)

	return h.Finalize(0, n)
}
