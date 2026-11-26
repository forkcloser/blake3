// Package blake3 implements the BLAKE3 cryptographic hash function.
//
// [New], [NewKeyed] and [NewDeriveKey] return a [Hasher] in each of BLAKE3's
// three modes: a [hash.Hash] with a 32-byte digest and an extendable output
// of any length ([Hasher.XOF]). [Sum256], [Sum512], [Sum256Keyed] and
// [DeriveKey] do the same in one call.
//
// A Hasher is not safe for concurrent use; its methods must not be called
// from more than one goroutine at a time. It holds no pointers, so copying a
// Hasher value forks its state: the copy and the original continue
// independently; [Hasher.MarshalBinary] and [Hasher.UnmarshalBinary] carry a
// state across processes and hosts. Write never fails (the error it returns
// to satisfy [hash.Hash] is always nil), and Sum leaves the state untouched,
// so Sum may be called repeatedly and interleaved with Write.
//
// The subpackages are [github.com/forkcloser/go-blake3/v2/bao], verified streaming
// over BLAKE3's tree, and [github.com/forkcloser/go-blake3/v2/guts], the tree-hashing
// primitives both packages are built from.
package blake3

import (
	"encoding/binary"
	"errors"
	"hash"
	"io"
	"math"
	"math/bits"
	"runtime"
	"sync"

	"github.com/forkcloser/go-blake3/v2/internal/kernel"
)

// The digest's size and the hash's block size, in bytes. Longer output is
// the XOF's.
const (
	Size      = 32
	BlockSize = kernel.BlockSize // 64
)

// Hasher implements hash.Hash. See the package documentation for its
// concurrency and copying rules. The zero value is the plain hash, as New
// returns it.
type Hasher struct {
	// key is the keyed mode's key or the derive mode's context key. The
	// plain mode has none: its key is the IV, read through keyWords, so that
	// the zero Hasher is the plain hash and not one under a key of zeros.
	key   [8]uint32
	flags uint32

	// log(n) set of Merkle subtree roots, at most one per height.
	stack [64][8]uint32
	// counter is the number of chunks compressed into the stack, and a bit
	// vector of the heights it occupies. Write keeps it a multiple of
	// kernel.Width(), which is 1 in pure Go: any count. Where the SIMD
	// kernels run it is a multiple of kernel.MaxSIMD except in a state
	// restored from a narrower host, until Write has realigned it.
	counter uint64

	// buf is the input not yet compressed: the tail, held back because the
	// last chunk has to wait for Sum, where it is the root or the tree's
	// right edge. How much is held is the path's: one chunk in pure Go, a
	// whole group of kernel.MaxSIMD chunks where the SIMD kernels run, so
	// that counter stays group-aligned, every tree Write compresses is
	// whole groups, and the kernels, which compute all their lanes
	// whatever they are handed, run full.
	buf    [kernel.MaxSIMD * kernel.ChunkSize]byte
	buflen int
}

// New returns an unkeyed Hasher.
func New() *Hasher {
	return new(Hasher)
}

// NewKeyed returns a Hasher in BLAKE3's keyed mode: a MAC under key.
func NewKeyed(key [32]byte) *Hasher {
	return &Hasher{key: keyWords(key), flags: kernel.FlagKeyedHash}
}

// NewDeriveKey returns a Hasher in BLAKE3's key-derivation mode: what is
// written is the key material, and the digest, or the XOF for longer, is the
// derived key. context should be hardcoded, globally unique, and
// application-specific, so that no two applications derive the same key
// from the same material; a good form is
//
//	[application] [commit timestamp] [purpose]
//
// such as "example.com 2019-12-25 16:18:03 session tokens v1".
func NewDeriveKey(context string) *Hasher {
	h := new(Hasher)
	h.setDeriveKey(context)

	return h
}

// Write implements hash.Hash. It always returns len(p), nil.
func (h *Hasher) Write(p []byte) (int, error) {
	lenp := len(p)
	width := kernel.Width()

	// A state restored from a host of another width (UnmarshalBinary) may
	// stand at a counter that is not a multiple of this one's: chunk by
	// chunk until it is, and the group logic below takes over.
	for width > 1 && h.counter%uint64(width) != 0 && len(p) > 0 { // #nosec G115 -- a small positive count
		n := copy(h.buf[h.buflen:kernel.ChunkSize], p)
		h.buflen += n
		p = p[n:]

		if h.buflen == kernel.ChunkSize && len(p) > 0 {
			h.pushSubtree(
				kernel.ChainingValue(kernel.CompressChunk(h.buf[:h.buflen], h.keyWords(), h.counter, h.flags)),
				0,
			)

			h.buflen = 0
		}
	}

	hold := width * kernel.ChunkSize

	// Fill the held buffer first; input that fits in it is held back.
	if h.buflen > 0 {
		n := copy(h.buf[h.buflen:hold], p)
		h.buflen += n
		p = p[n:]
	}

	if len(p) == 0 {
		return lenp, nil
	}

	// Past the buffer, which is now empty or full (a partial one absorbed
	// all of p above). A full buffer with input behind it is complete: a
	// chunk is compressed on the spot, a group joins the write's trees to
	// be dealt out with them. Then p's whole units follow, and p's last
	// unit, full or partial, is held back in its turn.
	flush := h.buflen == hold
	if flush && hold == kernel.ChunkSize {
		h.pushSubtree(kernel.ChainingValue(kernel.CompressChunk(h.buf[:hold], h.keyWords(), h.counter, h.flags)), 0)

		flush = false
	}

	rem := len(p) % hold
	if rem == 0 {
		rem = hold // never compress the last unit early
	}

	h.compressTrees(flush, p[:len(p)-rem])
	h.buflen = copy(h.buf[:hold], p[len(p)-rem:])

	return lenp, nil
}

// Sum implements hash.Hash: it appends the 32-byte digest of everything
// written so far to b and returns the result, without changing the state.
func (h *Hasher) Sum(b []byte) []byte {
	out := kernel.WordsToBytes(kernel.CompressNode(h.rootNode()))

	return append(b, out[:Size]...)
}

// Reset implements hash.Hash.
func (h *Hasher) Reset() {
	h.counter = 0
	h.buflen = 0
}

// BlockSize implements hash.Hash.
func (*Hasher) BlockSize() int { return BlockSize }

// Size implements hash.Hash: the digest is 32 bytes.
func (*Hasher) Size() int { return Size }

// XOF returns an OutputReader initialized with the current hash state. The
// state is captured at the call: later writes to h do not affect the reader.
func (h *Hasher) XOF() *OutputReader {
	return &OutputReader{
		n: h.rootNode(),
	}
}

// AppendBinary implements encoding.BinaryAppender: it appends the state's
// wire form to b.
func (h *Hasher) AppendBinary(b []byte) ([]byte, error) {
	b = append(b, stateTag...)
	b = binary.LittleEndian.AppendUint32(b, h.flags)

	for _, w := range h.keyWords() {
		b = binary.LittleEndian.AppendUint32(b, w)
	}

	b = binary.LittleEndian.AppendUint64(b, h.counter)

	for i := range h.stack {
		if h.hasSubtreeAtHeight(i) {
			for _, w := range h.stack[i] {
				b = binary.LittleEndian.AppendUint32(b, w)
			}
		}
	}

	b = binary.LittleEndian.AppendUint16(b, uint16(h.buflen)) // #nosec G115 -- at most a group, 16 KiB

	return append(b, h.buf[:h.buflen]...), nil
}

// MarshalBinary implements encoding.BinaryMarshaler: the state's wire form,
// 50 bytes for a hash of nothing, 32 more for each subtree in progress, and
// the input held back; at most 18 KiB in all. The state carries the key, a
// keyed Hasher's or the one a derive-mode Hasher made of its context, so it
// is as secret as the key: not to be stored or sent where the key would not
// be.
//
// The wire form opens with a tag that ends in the format's version.
// UnmarshalBinary restores the version this release writes and refuses
// every other: a state from a release with another version fails to
// restore, and is never read as a different state.
func (h *Hasher) MarshalBinary() ([]byte, error) {
	return h.AppendBinary(nil)
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler: it restores a state
// MarshalBinary produced, on any host, and continues it. A malformed state
// is refused with an error and h is left unchanged. A well-formed one is
// restored as it reads, and nothing authenticates it: the chaining values
// and the key are taken on trust, so a state from a source that is not
// trusted hashes what that source chose.
func (h *Hasher) UnmarshalBinary(data []byte) error {
	if len(data) < stateFixed+stateLenSize || string(data[:len(stateTag)]) != stateTag {
		return errInvalidState
	}

	flags := binary.LittleEndian.Uint32(data[len(stateTag):])
	if flags != 0 && flags != kernel.FlagKeyedHash && flags != kernel.FlagDeriveKeyMaterial {
		return errInvalidState
	}

	r := Hasher{flags: flags}
	for i := range r.key {
		r.key[i] = binary.LittleEndian.Uint32(data[len(stateTag)+4+4*i:])
	}

	counter := binary.LittleEndian.Uint64(data[stateFixed-8:])
	rest := data[stateFixed:]

	for i := range r.stack {
		if counter&(1<<i) == 0 {
			continue
		}

		if len(rest) < 32+stateLenSize {
			return errInvalidState
		}

		for j := range r.stack[i] {
			r.stack[i][j] = binary.LittleEndian.Uint32(rest[4*j:])
		}

		rest = rest[32:]
	}

	held := int(binary.LittleEndian.Uint16(rest))
	rest = rest[stateLenSize:]

	if len(rest) != held || !r.restorable(counter, held) {
		return errInvalidState
	}

	r.counter = counter
	r.Write(rest)
	*h = r

	return nil
}

// keyWords is the key every compression of h runs under.
func (h *Hasher) keyWords() *[8]uint32 {
	if h.flags == 0 {
		return &kernel.IV
	}

	return &h.key
}

// restorable reports whether h's mode and key, with a counter and a count of
// bytes held, are a state a Hasher can be in. The plain mode has no key of
// its own, so its state carries the IV. A counter past zero holds at least
// a byte back (the last chunk of the input is never compressed before Sum),
// and a group is the most any host holds. An input of 2^64-1 bytes, the
// most BLAKE3 hashes, is 2^54 chunks with the last held back: a counter at
// 2^54 or above is no input's, and the tree it describes has more levels
// than the stack.
func (h *Hasher) restorable(counter uint64, held int) bool {
	if h.flags == 0 && h.key != kernel.IV {
		return false
	}

	return held <= len(h.buf) && (counter == 0 || held > 0) && counter < maxChunks
}

func (h *Hasher) hasSubtreeAtHeight(i int) bool {
	return h.counter&(1<<i) != 0
}

func (h *Hasher) pushSubtree(cv [8]uint32, height int) {
	// seek to first open stack slot, merging subtrees as we go
	i := height
	for h.hasSubtreeAtHeight(i) {
		cv = kernel.ChainingValue(kernel.ParentNode(h.stack[i], cv, h.keyWords(), h.flags))
		i++
	}

	h.stack[i] = cv
	h.counter += 1 << height
}

// rootNode computes the root of the Merkle tree: the buffered group's node,
// a chunk or a parent, joined to the stack from the lowest occupied height
// up. It does not modify the stack.
func (h *Hasher) rootNode() kernel.Node {
	n := kernel.CompressBuffer(&h.buf, h.buflen, h.keyWords(), h.counter, h.flags)
	for i := bits.TrailingZeros64(h.counter); i < bits.Len64(h.counter); i++ {
		if h.hasSubtreeAtHeight(i) {
			n = kernel.ParentNode(h.stack[i], kernel.ChainingValue(n), h.keyWords(), h.flags)
		}
	}

	n.Flags |= kernel.FlagRoot

	return n
}

// compressTrees compresses the complete trees of a Write, the held group
// when flush is set and then eigenbuf's eigentrees, and pushes their
// chaining values in tree order. A lone tree runs on the caller, fanning
// out inside CompressEigentree when it is large; several go to goroutines
// once they hold the dispatched path's ParallelFloor between them: 24 KiB
// of trees in pure Go, 192 KiB under the AVX-512 kernels, 256 KiB under the
// AVX2 ones. The floor is above one goroutine's grain for the tail's sake:
// at one grain a write of a group and a few chunks, [16, 4, 2, 1], would
// wake a goroutine for the few, which costs more than compressing them
// here.
func (h *Hasher) compressTrees(flush bool, eigenbuf []byte) {
	// A climb of at most 64 trees to an aligned counter, then a descent of
	// at most 64: a fixed array, so the serial path allocates nothing.
	var heights [128]int

	trees := heights[:0]
	counter := h.counter

	if flush {
		trees = append(trees, groupHeight)
		counter += kernel.MaxSIMD
	}

	trees = kernel.AppendEigentrees(trees, counter, uint64(len(eigenbuf)/kernel.ChunkSize))
	if len(trees) == 0 {
		return
	}

	total := len(eigenbuf)
	if flush {
		total += len(h.buf)
	}

	if len(trees) == 1 || total < kernel.ParallelFloor() {
		h.compressTreesSerial(flush, eigenbuf, trees)
	} else {
		h.compressTreesParallel(flush, eigenbuf, trees)
	}
}

// The held group goes through CompressBuffer, not CompressEigentree: the
// latter's buffer reaches goroutines of its own for a large tree, and
// h.buf there would move a stack Hasher (Sum512's, DeriveKey's) to the heap.
func (h *Hasher) compressTreesSerial(flush bool, eigenbuf []byte, trees []int) {
	counter := h.counter

	for i, height := range trees {
		var n kernel.Node

		if flush && i == 0 {
			n = kernel.CompressBuffer(&h.buf, len(h.buf), h.keyWords(), counter, h.flags)
		} else {
			size := (1 << height) * kernel.ChunkSize
			n = kernel.CompressEigentree(eigenbuf[:size], h.keyWords(), counter, h.flags)
			eigenbuf = eigenbuf[size:]
		}

		h.pushSubtree(kernel.ChainingValue(n), height)
		counter += 1 << height
	}
}

// compressTreesParallel is compressTrees' concurrent half, kept apart so the
// goroutine closure and its captures reach the heap on this path alone. A
// tree of a group or more is a unit of its own; the trees below a group that
// the pure-Go path leaves, holding one chunk back, are batched into runs
// wherever they sit (Eigentrees climbs before it descends, so a run precedes
// the first large tree as readily as it follows the last), each run one
// unit, so that no run executes on the caller after the large trees are
// spawned. A unit recomputes its own heights: the sub-sequence of a tree
// sequence is the sequence of its own range, and handing a unit the shared
// slice would move the caller's array to the heap. The held group is the
// caller's own unit, compressed while the others wake, from the buffer no
// goroutine reads.
func (h *Hasher) compressTreesParallel(flush bool, eigenbuf []byte, trees []int) {
	cvs := make([][8]uint32, len(trees))

	var wg sync.WaitGroup

	unit := func(idx int, buf []byte, counter uint64) {
		wg.Add(1)

		go func(idx int, buf []byte, counter uint64, key [8]uint32, flags uint32) {
			defer wg.Done()

			var heights [128]int

			for i, height := range kernel.AppendEigentrees(heights[:0], counter, uint64(len(buf)/kernel.ChunkSize)) {
				n := (1 << height) * kernel.ChunkSize
				cvs[idx+i] = kernel.ChainingValue(kernel.SpreadEigentree(buf[:n], &key, counter, flags))
				buf = buf[n:]
				counter += 1 << height
			}
		}(idx, buf, counter, *h.keyWords(), h.flags)
	}

	first, counter := 0, h.counter
	if flush {
		first, counter = 1, counter+kernel.MaxSIMD
	}

	dealUnits(trees[first:], first, eigenbuf, counter, unit)

	if flush {
		cvs[0] = kernel.ChainingValue(kernel.CompressBuffer(&h.buf, len(h.buf), h.keyWords(), h.counter, h.flags))
	}

	wg.Wait()

	for i, height := range trees {
		h.pushSubtree(cvs[i], height)
	}
}

// setDeriveKey puts h in key-derivation mode for context: the context,
// hashed in its own mode, is the key the material is hashed under.
func (h *Hasher) setDeriveKey(context string) {
	*h = Hasher{key: contextKey(context), flags: kernel.FlagDeriveKeyMaterial}
}

// contextKey is the key a key derivation under context hashes its material
// with: the hash of the context, in the mode of its own.
func contextKey(context string) [8]uint32 {
	return kernel.ChainingValue(rootOf([]byte(context), &kernel.IV, kernel.FlagDeriveKeyContext))
}

// rootOf is the root node of b under key in the mode flags name, a keyed
// one: what a Hasher of that key and mode holds after one Write of b. An
// input of a chunk or less is compressed as it stands, with no Hasher:
// making one, with the group it holds back, costs more than such an input
// does to hash.
func rootOf(b []byte, key *[8]uint32, flags uint32) kernel.Node {
	if len(b) > kernel.ChunkSize {
		h := Hasher{key: *key, flags: flags}
		h.Write(b)

		return h.rootNode()
	}

	n := kernel.CompressChunk(b, key, 0, flags)
	n.Flags |= kernel.FlagRoot

	return n
}

// groupHeight is the height of a tree of kernel.MaxSIMD chunks, the held
// group's; the array below fails to compile when the constant drifts.
const groupHeight = 4

var _ [0]struct{} = [kernel.MaxSIMD - 1<<groupHeight]struct{}{}

// Hasher is a hash.Hash.
var _ hash.Hash = (*Hasher)(nil)

// dealUnits hands eigenbuf's trees, numbered from idx, to unit: a tree of a
// group or more on its own, a run of smaller trees together.
func dealUnits(trees []int, idx int, eigenbuf []byte, counter uint64, unit func(idx int, buf []byte, counter uint64)) {
	runStart, runBuf, runCounter := -1, eigenbuf, counter

	for i, height := range trees {
		n := (1 << height) * kernel.ChunkSize

		if 1<<height < kernel.MaxSIMD {
			if runStart < 0 {
				runStart, runBuf, runCounter = i, eigenbuf, counter
			}
		} else {
			if runStart >= 0 {
				unit(idx+runStart, runBuf[:len(runBuf)-len(eigenbuf)], runCounter)

				runStart = -1
			}

			unit(idx+i, eigenbuf[:n], counter)
		}

		eigenbuf = eigenbuf[n:]
		counter += 1 << height
	}

	if runStart >= 0 {
		unit(idx+runStart, runBuf[:len(runBuf)-len(eigenbuf)], runCounter)
	}
}

// keyWords is a key as the words the compression function takes.
func keyWords(key [32]byte) (words [8]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(key[i*4:])
	}

	return words
}

// Sum256 returns the BLAKE3 hash of b.
func Sum256(b []byte) (out [32]byte) {
	out512 := Sum512(b)
	copy(out[:], out512[:])

	return out
}

// Sum512 returns the first 64 bytes of the BLAKE3 output of b: the digest,
// then the next 32 bytes of the XOF.
func Sum512(b []byte) (out [64]byte) {
	var n kernel.Node

	switch {
	case len(b) <= kernel.BlockSize:
		// What the next case computes, without CompressChunk: a block is
		// one compression, and this is the faster way to it.
		var block [64]byte
		copy(block[:], b)

		return kernel.WordsToBytes(kernel.CompressNode(kernel.Node{
			CV:       kernel.IV,
			Block:    kernel.BytesToWords(block),
			BlockLen: uint32(len(b)), // #nosec G115 -- this case holds at most BlockSize bytes
			Flags:    kernel.FlagChunkStart | kernel.FlagChunkEnd | kernel.FlagRoot,
		}))
	case len(b) <= kernel.ChunkSize:
		n = kernel.CompressChunk(b, &kernel.IV, 0, 0)
		n.Flags |= kernel.FlagRoot
	default:
		var h Hasher
		h.Write(b)
		n = h.rootNode()
	}

	return kernel.WordsToBytes(kernel.CompressNode(n))
}

// Sum256Keyed returns the keyed BLAKE3 hash of b under key: a MAC.
func Sum256Keyed(key [32]byte, b []byte) (out [32]byte) {
	words := keyWords(key)
	sum := kernel.WordsToBytes(kernel.CompressNode(rootOf(b, &words, kernel.FlagKeyedHash)))
	copy(out[:], sum[:])

	return out
}

// DeriveKey returns n bytes of key derived from material under context, as
// NewDeriveKey describes; the first 32 are the digest, the rest the XOF's.
// It panics if n is negative.
func DeriveKey(context string, material []byte, n int) []byte {
	key := contextKey(context)

	out := make([]byte, n)
	// From offset 0 the read cannot reach the end of the stream, the one
	// case in which Read returns less than asked, or an error.
	r := OutputReader{n: rootOf(material, &key, kernel.FlagDeriveKeyMaterial)}
	_, _ = r.Read(out)

	return out
}

// The hash state's wire form (AppendBinary): a tag, the mode's flags and
// key, the chunk counter, one chaining value per occupied height of the
// stack from the lowest up, and the bytes held back. Nothing in it names
// the host's width: UnmarshalBinary feeds the held bytes through Write,
// which holds back what this host does and realigns the counter when the
// state came from a narrower one. The tag's last byte is the form's
// version, and the only thing that tells one layout from another: a change
// to the layout that keeps the byte lets an older state restore as garbage.
const (
	stateTag     = "b3h\x01"
	stateFixed   = len(stateTag) + 4 + 32 + 8 // tag, flags, key, counter
	stateLenSize = 2                          // the held bytes' count
)

var errInvalidState = errors.New("blake3: invalid hash state")

// maxChunks is the chunks of the longest input BLAKE3 hashes, 2^64-1 bytes.
const maxChunks = 1 << 54

// The errors Seek and ReadAt return for a position outside the stream.
var (
	errSeekNegative = errors.New("blake3: position before the start of the stream")
	errSeekPastEnd  = errors.New("blake3: position past the end of the stream")
)

// errSeekWhence is what Seek returns for a whence it does not know.
var errSeekWhence = errors.New("blake3: Seek: invalid whence")

// minParallelReadBytes is the least output one goroutine is handed when a
// large, block-aligned Read is split across CPUs.
const minParallelReadBytes = 16 * 1024

// outputBufSize is the output an OutputReader compresses at once: one block
// per SIMD lane.
const outputBufSize = kernel.MaxSIMD * kernel.BlockSize

// An OutputReader produces a seekable stream of 2^64 - 1 pseudorandom output
// bytes: the BLAKE3 extendable output of the state it was created from. Read
// and Seek move its position; ReadAt reads anywhere and leaves it alone.
//
// Like a Hasher it is not safe for concurrent use, except that ReadAt calls
// may run concurrently with one another, and it holds no pointers, so a copy
// continues independently from the same position. An OutputReader comes
// from Hasher.XOF: its zero value is no hash's output.
type OutputReader struct {
	n        kernel.Node
	buf      [outputBufSize]byte
	bufStart uint64 // stream offset of buf[0]
	buflen   int    // number of valid bytes in buf
	off      uint64
}

// Read implements io.Reader. Callers may assume that Read returns len(p), nil
// unless the read would extend beyond the end of the stream, in which case it
// returns the bytes that remain with a nil error; once the position is at the
// end, Read returns 0, io.EOF.
func (or *OutputReader) Read(p []byte) (int, error) {
	if or.off == math.MaxUint64 {
		return 0, io.EOF
	} else if rem := math.MaxUint64 - or.off; uint64(len(p)) > rem {
		p = p[:rem]
	}

	lenp := len(p)

	for len(p) > 0 {
		// drain buffered output
		// #nosec G115 -- buflen is a buffer's fill, never negative
		if or.off >= or.bufStart && or.off-or.bufStart < uint64(or.buflen) {
			n := copy(p, or.buf[or.off-or.bufStart:or.buflen])
			p = p[n:]
			or.off += uint64(n) // #nosec G115 -- n is what copy returned, never negative

			continue
		}

		if head := int(or.off % kernel.BlockSize); head != 0 || len(p) < outputBufSize {
			// the read is small or unaligned; compress (only) as many blocks
			// as necessary into our buffer, and serve it from there
			or.bufStart = or.off - uint64(head)
			or.n.Counter = or.bufStart / kernel.BlockSize
			need := min(head+len(p), outputBufSize)
			numBlocks := (need + kernel.BlockSize - 1) / kernel.BlockSize
			or.buflen = kernel.BlockSize * kernel.CompressBlocksN(&or.buf, or.n, numBlocks)

			continue
		}
		// the read is large and block-aligned; compress directly into p
		p = or.readAligned(p)
	}

	return lenp, nil
}

// Seek implements io.Seeker. A position before the start of the stream or
// past its end, and a whence that is not one of io.SeekStart, io.SeekCurrent
// and io.SeekEnd, are rejected with an error and leave the position
// unchanged; io.SeekEnd with a zero offset positions at the end, where Read
// returns io.EOF, and with -1 at the last byte. Positions of 2^63 and above
// are valid but cannot be represented in the int64 return value, which is
// then negative.
func (or *OutputReader) Seek(offset int64, whence int) (int64, error) {
	off := or.off

	switch whence {
	case io.SeekStart:
		if offset < 0 {
			return 0, errSeekNegative
		}

		off = uint64(offset)
	case io.SeekCurrent:
		if offset < 0 {
			if uint64(-offset) > off {
				return 0, errSeekNegative
			}

			off -= uint64(-offset)
		} else if off += uint64(offset); off < uint64(offset) {
			return 0, errSeekPastEnd
		}
	case io.SeekEnd:
		if offset > 0 {
			return 0, errSeekPastEnd
		}

		// #nosec G115 -- offset is at most 0 here: the wrap is the point,
		// counting back from the end of the 2^64 - 1 byte stream.
		off = uint64(offset) - 1
	default:
		return 0, errSeekWhence
	}

	or.off = off
	// NOTE: there is no need to update or invalidate the buffer: it caches an
	// absolute range [bufStart, bufStart+buflen) of the stream, and Read only
	// serves from it when or.off falls within that range.
	// #nosec G115 -- positions past 2^63 - 1 do not fit io.Seeker's int64 and
	// come back negative; the seek is exact.
	return int64(or.off), nil
}

// ReadAt implements io.ReaderAt: the bytes of the stream from off into p,
// leaving the reader's position where it was. Several goroutines may call
// it at once on one reader, as it writes nothing it reads; a Read or Seek
// at the same time is not safe. The stream's end, at 2^64 - 1, lies past
// any offset an int64 names, so a ReadAt always fills p and returns
// len(p), nil; a negative offset is an error.
func (or *OutputReader) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errSeekNegative
	}

	r := *or
	r.off = uint64(off)

	// Read returns a short count with a nil error across the end of the
	// stream, which io.ReaderAt forbids and no call here reaches: off and
	// len(p) are each under 2^63, the end at 2^64 - 1.
	return r.Read(p)
}

// readAligned compresses output straight into p, which starts on a block
// boundary of the stream and holds at least outputBufSize bytes, and returns
// what is left of p.
func (or *OutputReader) readAligned(p []byte) []byte {
	or.n.Counter = or.off / kernel.BlockSize
	numBufs := len(p) / outputBufSize

	const minBufsPerCPU = minParallelReadBytes / outputBufSize

	// GOMAXPROCS, not NumCPU: what the scheduler will run on, a container's
	// quota and the caller's setting included.
	par := min(numBufs/minBufsPerCPU, runtime.GOMAXPROCS(0))
	if par <= 1 {
		kernel.CompressBlocks((*[outputBufSize]byte)(p), or.n)
		or.off += outputBufSize

		return p[outputBufSize:]
	}
	// enough work for each CPU to be worth parallelizing; distribute
	// the buffers evenly among the goroutines
	var wg sync.WaitGroup

	for i := range par {
		bufs := uint64(numBufs / par) // #nosec G115 -- a quotient of two positive counts
		if i < numBufs%par {
			bufs++
		}

		wg.Add(1)
		go func(p []byte, n kernel.Node, bufs uint64) {
			defer wg.Done()

			for i := range bufs {
				kernel.CompressBlocks((*[outputBufSize]byte)(p[i*outputBufSize:]), n)
				n.Counter += outputBufSize / kernel.BlockSize
			}
		}(p, or.n, bufs)

		p = p[bufs*outputBufSize:]
		or.off += bufs * outputBufSize
		or.n.Counter = or.off / kernel.BlockSize
	}

	wg.Wait()

	return p
}
