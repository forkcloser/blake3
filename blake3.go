// Package blake3 implements the BLAKE3 cryptographic hash function.
//
// [New] returns a [Hasher]: a [hash.Hash] with a caller-chosen digest size,
// an optional 32-byte key, and an extendable output ([Hasher.XOF]). [Sum256]
// and [Sum512] hash a byte slice in one call, and [DeriveKey] is BLAKE3's
// key-derivation mode.
//
// A Hasher is not safe for concurrent use; its methods must not be called
// from more than one goroutine at a time. It holds no pointers, so copying a
// Hasher value forks its state: the copy and the original continue
// independently. Write never fails (the error it returns to satisfy
// [hash.Hash] is always nil), and Sum leaves the state untouched, so Sum may
// be called repeatedly and interleaved with Write. Digests longer than 64
// bytes are produced through the XOF.
//
// The subpackages are [github.com/forkcloser/blake3/bao], verified streaming
// over BLAKE3's tree, and [github.com/forkcloser/blake3/guts], the tree-hashing
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
	"slices"
	"sync"

	"github.com/forkcloser/blake3/guts"
)

// Hasher implements hash.Hash. See the package documentation for its
// concurrency and copying rules.
type Hasher struct {
	key   [8]uint32
	flags uint32
	size  int // output size, for Sum

	// log(n) set of Merkle subtree roots, at most one per height.
	stack [64][8]uint32
	// counter is the number of chunks compressed into the stack, and a bit
	// vector of the heights it occupies. Write keeps it a multiple of
	// guts.MaxSIMD.
	counter uint64

	// buf is the input not yet compressed: the tail, held back because the
	// last chunk has to wait for Sum, where it is the root or the tree's
	// right edge. How much is held is the path's: one chunk in pure Go, a
	// whole group of guts.MaxSIMD chunks where the SIMD kernels run, so
	// that counter stays group-aligned, every tree Write compresses is
	// whole groups, and the kernels, which compute all their lanes
	// whatever they are handed, run full.
	buf    [guts.MaxSIMD * guts.ChunkSize]byte
	buflen int
}

// Write implements hash.Hash. It always returns len(p), nil.
func (h *Hasher) Write(p []byte) (int, error) {
	lenp := len(p)
	hold := guts.KernelWidth() * guts.ChunkSize

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
	if flush && hold == guts.ChunkSize {
		h.pushSubtree(guts.ChainingValue(guts.CompressChunk(h.buf[:hold], &h.key, h.counter, h.flags)), 0)

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

// Sum implements hash.Hash: it appends the current digest to b and returns
// the resulting slice, without changing the underlying state. A digest longer
// than 64 bytes is the first Size() bytes of the XOF stream.
func (h *Hasher) Sum(b []byte) (sum []byte) {
	// Append h.Size() bytes to b, reusing b's capacity when it has room.
	sum = slices.Grow(b, h.Size())[:len(b)+h.Size()]
	// Read into the appended portion of sum. Use a low-latency-low-throughput
	// path for small digests (requiring a single compression), and a
	// high-latency-high-throughput path for large digests.
	if dst := sum[len(b):]; len(dst) <= guts.BlockSize {
		out := guts.WordsToBytes(guts.CompressNode(h.rootNode()))
		copy(dst, out[:])
	} else {
		// From offset 0 the read cannot reach the end of the stream, the
		// one case in which Read returns less than asked, or an error.
		or := OutputReader{n: h.rootNode()}
		_, _ = or.Read(dst)
	}

	return sum
}

// Reset implements hash.Hash.
func (h *Hasher) Reset() {
	h.counter = 0
	h.buflen = 0
}

// BlockSize implements hash.Hash.
func (*Hasher) BlockSize() int { return guts.BlockSize }

// Size implements hash.Hash.
func (h *Hasher) Size() int { return h.size }

// XOF returns an OutputReader initialized with the current hash state. The
// state is captured at the call: later writes to h do not affect the reader.
func (h *Hasher) XOF() *OutputReader {
	return &OutputReader{
		n: h.rootNode(),
	}
}

func (h *Hasher) hasSubtreeAtHeight(i int) bool {
	return h.counter&(1<<i) != 0
}

func (h *Hasher) pushSubtree(cv [8]uint32, height int) {
	// seek to first open stack slot, merging subtrees as we go
	i := height
	for h.hasSubtreeAtHeight(i) {
		cv = guts.ChainingValue(guts.ParentNode(h.stack[i], cv, &h.key, h.flags))
		i++
	}

	h.stack[i] = cv
	h.counter += 1 << height
}

// rootNode computes the root of the Merkle tree: the buffered group's node,
// a chunk or a parent, joined to the stack from the lowest occupied height
// up. It does not modify the stack.
func (h *Hasher) rootNode() guts.Node {
	n := guts.CompressBuffer(&h.buf, h.buflen, &h.key, h.counter, h.flags)
	for i := bits.TrailingZeros64(h.counter); i < bits.Len64(h.counter); i++ {
		if h.hasSubtreeAtHeight(i) {
			n = guts.ParentNode(h.stack[i], guts.ChainingValue(n), &h.key, h.flags)
		}
	}

	n.Flags |= guts.FlagRoot

	return n
}

// compressTrees compresses the complete trees of a Write, the held group
// when flush is set and then eigenbuf's eigentrees, and pushes their
// chaining values in tree order. A lone tree runs on the caller, fanning
// out inside CompressEigentree when it is large; several go to goroutines
// once they hold one and a half times the parallel grain between them, the
// grain being what one goroutine is worth on the dispatched path. In pure
// Go that is 24 KiB, measured against upstream's goroutine per tree on
// darwin/arm64: below it the serial path won, by ~45% at 4 KiB writes and
// ~7% at 16 KiB; from io.Copy's 32 KiB, which is 31 KiB of trees after the
// held chunk, the dealing below won by 9–20%; and a 24 KiB write, which is
// [16, 4, 2, 1] chunks, lost 13% to a goroutine for its 7-chunk tail.
func (h *Hasher) compressTrees(flush bool, eigenbuf []byte) {
	// A climb of at most 64 trees to an aligned counter, then a descent of
	// at most 64: a fixed array, so the serial path allocates nothing.
	var heights [128]int

	trees := heights[:0]
	counter := h.counter

	if flush {
		trees = append(trees, groupHeight)
		counter += guts.MaxSIMD
	}

	trees = guts.AppendEigentrees(trees, counter, uint64(len(eigenbuf)/guts.ChunkSize))
	if len(trees) == 0 {
		return
	}

	total := len(eigenbuf)
	if flush {
		total += len(h.buf)
	}

	if grain := guts.ParallelGrain(); len(trees) == 1 || total < grain+grain/2 {
		h.compressTreesSerial(flush, eigenbuf, trees)
	} else {
		h.compressTreesParallel(flush, eigenbuf, trees)
	}
}

// groupHeight is the height of a tree of guts.MaxSIMD chunks, the held
// group's; the array below fails to compile when the constant drifts.
const groupHeight = 4

var _ [0]struct{} = [guts.MaxSIMD - 1<<groupHeight]struct{}{}

// The held group goes through CompressBuffer, not CompressEigentree: the
// latter's buffer reaches goroutines of its own for a large tree, and
// h.buf there would move a stack Hasher (Sum512's, DeriveKey's) to the heap.
func (h *Hasher) compressTreesSerial(flush bool, eigenbuf []byte, trees []int) {
	counter := h.counter

	for i, height := range trees {
		var n guts.Node

		if flush && i == 0 {
			n = guts.CompressBuffer(&h.buf, len(h.buf), &h.key, counter, h.flags)
		} else {
			size := (1 << height) * guts.ChunkSize
			n = guts.CompressEigentree(eigenbuf[:size], &h.key, counter, h.flags)
			eigenbuf = eigenbuf[size:]
		}

		h.pushSubtree(guts.ChainingValue(n), height)
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

			for i, height := range guts.AppendEigentrees(heights[:0], counter, uint64(len(buf)/guts.ChunkSize)) {
				n := (1 << height) * guts.ChunkSize
				cvs[idx+i] = guts.ChainingValue(guts.CompressEigentree(buf[:n], &key, counter, flags))
				buf = buf[n:]
				counter += 1 << height
			}
		}(idx, buf, counter, h.key, h.flags)
	}

	first, counter := 0, h.counter
	if flush {
		first, counter = 1, counter+guts.MaxSIMD
	}

	dealUnits(trees[first:], first, eigenbuf, counter, unit)

	if flush {
		cvs[0] = guts.ChainingValue(guts.CompressBuffer(&h.buf, len(h.buf), &h.key, h.counter, h.flags))
	}

	wg.Wait()

	for i, height := range trees {
		h.pushSubtree(cvs[i], height)
	}
}

// dealUnits hands eigenbuf's trees, numbered from idx, to unit: a tree of a
// group or more on its own, a run of smaller trees together.
func dealUnits(trees []int, idx int, eigenbuf []byte, counter uint64, unit func(idx int, buf []byte, counter uint64)) {
	runStart, runBuf, runCounter := -1, eigenbuf, counter

	for i, height := range trees {
		n := (1 << height) * guts.ChunkSize

		if 1<<height < guts.MaxSIMD {
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

func newHasher(key [8]uint32, flags uint32, size int) *Hasher {
	return &Hasher{
		key:   key,
		flags: flags,
		size:  size,
	}
}

// New returns a Hasher for the specified digest size and key. If key is nil,
// the hash is unkeyed. Otherwise, len(key) must be 32; New panics if size is
// negative or if a key of any other length is provided.
func New(size int, key []byte) *Hasher {
	if size < 0 {
		panic("blake3: digest size cannot be negative")
	}

	if key == nil {
		return newHasher(guts.IV, 0, size)
	}

	if len(key) != 32 {
		panic("blake3: key must be 32 bytes")
	}

	var keyWords [8]uint32
	for i := range keyWords {
		keyWords[i] = binary.LittleEndian.Uint32(key[i*4:])
	}

	return newHasher(keyWords, guts.FlagKeyedHash, size)
}

// Sum256 returns the unkeyed BLAKE3 hash of b, truncated to 256 bits.
func Sum256(b []byte) (out [32]byte) {
	out512 := Sum512(b)
	copy(out[:], out512[:])

	return out
}

// Sum512 returns the unkeyed BLAKE3 hash of b, truncated to 512 bits.
func Sum512(b []byte) (out [64]byte) {
	var n guts.Node

	switch {
	case len(b) <= guts.BlockSize:
		var block [64]byte
		copy(block[:], b)

		return guts.WordsToBytes(guts.CompressNode(guts.Node{
			CV:       guts.IV,
			Block:    guts.BytesToWords(block),
			BlockLen: uint32(len(b)), // #nosec G115 -- this case holds at most BlockSize bytes
			Flags:    guts.FlagChunkStart | guts.FlagChunkEnd | guts.FlagRoot,
		}))
	case len(b) <= guts.ChunkSize:
		n = guts.CompressChunk(b, &guts.IV, 0, 0)
		n.Flags |= guts.FlagRoot
	default:
		h := Hasher{key: guts.IV, size: len(out)}
		h.Write(b)
		n = h.rootNode()
	}

	return guts.WordsToBytes(guts.CompressNode(n))
}

// DeriveKey derives a subkey from ctx and srcKey. ctx should be hardcoded,
// globally unique, and application-specific. A good format for ctx strings is:
//
//	[application] [commit timestamp] [purpose]
//
// e.g.:
//
//	example.com 2019-12-25 16:18:03 session tokens v1
//
// The purpose of these requirements is to ensure that an attacker cannot trick
// two different applications into using the same context string.
func DeriveKey(subKey []byte, ctx string, srcKey []byte) {
	// construct the derivation Hasher
	const derivationIVLen = 32

	// Both hashers live on the stack: a Hasher is a group buffer and a
	// stack, too much to allocate twice for one derivation.
	h := Hasher{key: guts.IV, flags: guts.FlagDeriveKeyContext, size: derivationIVLen}
	h.Write([]byte(ctx))

	var derivationIV [derivationIVLen]byte

	h.Sum(derivationIV[:0])

	var ivWords [8]uint32
	for i := range ivWords {
		ivWords[i] = binary.LittleEndian.Uint32(derivationIV[i*4:])
	}

	h = Hasher{key: ivWords, flags: guts.FlagDeriveKeyMaterial}
	// derive the subKey
	h.Write(srcKey)
	// From offset 0 the read cannot reach the end of the stream; see Sum.
	_, _ = h.XOF().Read(subKey)
}

// The errors Seek returns for a position outside the stream.
var (
	errSeekNegative = errors.New("seek position cannot be negative")
	errSeekPastEnd  = errors.New("seek position cannot exceed end of stream")
)

// minParallelReadBytes is the least output one goroutine is handed when a
// large, block-aligned Read is split across CPUs.
const minParallelReadBytes = 16 * 1024

// outputBufSize is the output an OutputReader compresses at once: one block
// per SIMD lane.
const outputBufSize = guts.MaxSIMD * guts.BlockSize

// An OutputReader produces a seekable stream of 2^64 - 1 pseudorandom output
// bytes: the BLAKE3 extendable output of the state it was created from.
//
// Like a Hasher it is not safe for concurrent use, and it holds no pointers,
// so a copy continues independently from the same position.
type OutputReader struct {
	n        guts.Node
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

		if head := int(or.off % guts.BlockSize); head != 0 || len(p) < outputBufSize {
			// the read is small or unaligned; compress (only) as many blocks
			// as necessary into our buffer, and serve it from there
			or.bufStart = or.off - uint64(head)
			or.n.Counter = or.bufStart / guts.BlockSize
			need := min(head+len(p), outputBufSize)
			numBlocks := (need + guts.BlockSize - 1) / guts.BlockSize
			or.buflen = guts.BlockSize * guts.CompressBlocksN(&or.buf, or.n, numBlocks)

			continue
		}
		// the read is large and block-aligned; compress directly into p
		p = or.readAligned(p)
	}

	return lenp, nil
}

// Seek implements io.Seeker. A position before the start of the stream or
// past its end is rejected with an error and leaves the position unchanged;
// io.SeekEnd with a zero offset positions at the end, where Read returns
// io.EOF, and with -1 at the last byte. Positions of 2^63 and above are
// valid but cannot be represented in the int64 return value, which is then
// negative.
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
		panic("invalid whence")
	}

	or.off = off
	// NOTE: there is no need to update or invalidate the buffer: it caches an
	// absolute range [bufStart, bufStart+buflen) of the stream, and Read only
	// serves from it when or.off falls within that range.
	// #nosec G115 -- positions past 2^63 - 1, reachable only through SeekEnd,
	// do not fit io.Seeker's int64 and come back negative; the seek is exact.
	return int64(or.off), nil
}

// readAligned compresses output straight into p, which starts on a block
// boundary of the stream and holds at least outputBufSize bytes, and returns
// what is left of p.
func (or *OutputReader) readAligned(p []byte) []byte {
	or.n.Counter = or.off / guts.BlockSize
	numBufs := len(p) / outputBufSize

	const minBufsPerCPU = minParallelReadBytes / outputBufSize

	// GOMAXPROCS, not NumCPU: what the scheduler will run on, a container's
	// quota and the caller's setting included.
	par := min(numBufs/minBufsPerCPU, runtime.GOMAXPROCS(0))
	if par <= 1 {
		guts.CompressBlocks((*[outputBufSize]byte)(p), or.n)
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
		go func(p []byte, n guts.Node, bufs uint64) {
			defer wg.Done()

			for i := range bufs {
				guts.CompressBlocks((*[outputBufSize]byte)(p[i*outputBufSize:]), n)
				n.Counter += outputBufSize / guts.BlockSize
			}
		}(p, or.n, bufs)

		p = p[bufs*outputBufSize:]
		or.off += bufs * outputBufSize
		or.n.Counter = or.off / guts.BlockSize
	}

	wg.Wait()

	return p
}

// Hasher is a hash.Hash.
var _ hash.Hash = (*Hasher)(nil)
