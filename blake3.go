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
package blake3 // import "github.com/forkcloser/blake3"

import (
	"encoding/binary"
	"errors"
	"hash"
	"io"
	"math"
	"math/bits"
	"runtime"
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
	stack   [64][8]uint32
	counter uint64 // number of buffers hashed; also serves as a bit vector indicating which stack elems are occupied

	buf    [guts.ChunkSize]byte
	buflen int
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

// rootNode computes the root of the Merkle tree. It does not modify the
// stack.
func (h *Hasher) rootNode() guts.Node {
	n := guts.CompressChunk(h.buf[:h.buflen], &h.key, h.counter, h.flags)
	for i := bits.TrailingZeros64(h.counter); i < bits.Len64(h.counter); i++ {
		if h.hasSubtreeAtHeight(i) {
			n = guts.ParentNode(h.stack[i], guts.ChainingValue(n), &h.key, h.flags)
		}
	}
	n.Flags |= guts.FlagRoot
	return n
}

// Write implements hash.Hash. It always returns len(p), nil.
func (h *Hasher) Write(p []byte) (int, error) {
	lenp := len(p)

	// align to chunk boundary
	if h.buflen > 0 {
		n := copy(h.buf[h.buflen:], p)
		h.buflen += n
		p = p[n:]
	}
	if h.buflen == len(h.buf) && len(p) > 0 {
		n := guts.CompressChunk(h.buf[:], &h.key, h.counter, h.flags)
		h.pushSubtree(guts.ChainingValue(n), 0)
		h.buflen = 0
	}

	// process full chunks
	if len(p) > len(h.buf) {
		rem := len(p) % len(h.buf)
		if rem == 0 {
			rem = len(h.buf) // don't prematurely compress
		}
		eigenbuf := p[:len(p)-rem]
		trees := guts.Eigentrees(h.counter, uint64(len(eigenbuf)/guts.ChunkSize))

		// A Write's eigentrees form a cascade of independent subtrees: a
		// 64 KiB write at counter 0 is [32,16,8,4,2,1] chunks (the last
		// chunk is held back), a 1 MiB one is ten trees. What decides
		// whether threads pay for themselves is the write's total size,
		// not any one tree's: below minParallelWriteBytes it all runs
		// inline, serially — no goroutines, no scratch, nothing for a
		// closure to capture and drag onto the heap. Above it, the trees
		// large enough to fan out internally each get a goroutine, and the
		// tail of small trees — which together are nearly the size of the
		// largest — go to one more, so they overlap the big ones instead of
		// running serially after them. CVs are pushed in tree order once
		// all are in, since the CV stack merges depend on that order.
		if len(eigenbuf) < minParallelWriteBytes {
			counter := h.counter
			for _, height := range trees {
				buf := eigenbuf[:(1<<height)*guts.ChunkSize]
				eigenbuf = eigenbuf[len(buf):]
				h.pushSubtree(guts.ChainingValue(guts.CompressEigentree(buf, &h.key, counter, h.flags)), height)
				counter += 1 << height
			}
		} else {
			h.writeTreesParallel(eigenbuf, trees)
		}
		p = p[len(p)-rem:]
	}

	// buffer remaining partial chunk
	n := copy(h.buf[h.buflen:], p)
	h.buflen += n

	return lenp, nil
}

// minParallelWriteBytes is the smallest run of eigentree bytes compressed
// concurrently. Measured on the generic (non-SIMD) path, darwin/arm64:
// below it the thread wakeups cost more than they overlap (a 32 KiB write
// was slower parallel than serial with a 16 KiB threshold), and above it
// the cascade parallelizes well — 64 KiB writes are 1.7× the serial rate.
// Note a "64 KiB" write is 63 KiB of trees (the last chunk is held back),
// so a threshold at exactly 64 KiB would run it serially.
const minParallelWriteBytes = 32 * 1024

// writeTreesParallel compresses a Write's eigentrees concurrently: each tree
// larger than MaxSIMD chunks gets a goroutine (it fans out further inside
// CompressEigentree), and the run of small trees at the tail shares one.
// Every CV is pushed in tree order once all are in. Kept out of Write so the
// goroutine closure and its captures live only on this path.
func (h *Hasher) writeTreesParallel(eigenbuf []byte, trees []int) {
	cvs := make([][8]uint32, len(trees))
	counter := h.counter
	var wg sync.WaitGroup
	// The small trees form a contiguous tail: Eigentrees climbs (heights
	// increase while the counter is not yet aligned) then descends, and
	// only the descent can hold trees below MaxSIMD chunks after a large
	// one — but a climb of small trees precedes the first large one too.
	// So small trees are grouped into runs wherever they sit, each run
	// one goroutine, so no run of them ever executes on the caller.
	runStart := -1
	flushRun := func(end int, bufStart []byte, ctr uint64) {
		wg.Add(1)
		go func(lo, hi int, buf []byte, counter uint64) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				height := trees[i]
				n := (1 << height) * guts.ChunkSize
				cvs[i] = guts.ChainingValue(guts.CompressEigentree(buf[:n], &h.key, counter, h.flags))
				buf = buf[n:]
				counter += 1 << height
			}
		}(runStart, end, bufStart, ctr)
		runStart = -1
	}
	var runBuf []byte
	var runCounter uint64
	for i, height := range trees {
		buf := eigenbuf[:(1<<height)*guts.ChunkSize]
		if 1<<height <= guts.MaxSIMD {
			if runStart < 0 {
				runStart, runBuf, runCounter = i, eigenbuf, counter
			}
		} else {
			if runStart >= 0 {
				flushRun(i, runBuf, runCounter)
			}
			wg.Add(1)
			go func(i int, buf []byte, counter uint64) {
				defer wg.Done()
				cvs[i] = guts.ChainingValue(guts.CompressEigentree(buf, &h.key, counter, h.flags))
			}(i, buf, counter)
		}
		eigenbuf = eigenbuf[len(buf):]
		counter += 1 << height
	}
	if runStart >= 0 {
		flushRun(len(trees), runBuf, runCounter)
	}
	wg.Wait()
	for i, height := range trees {
		h.pushSubtree(cvs[i], height)
	}
}

// Sum implements hash.Hash: it appends the current digest to b and returns
// the resulting slice, without changing the underlying state. A digest longer
// than 64 bytes is the first Size() bytes of the XOF stream.
func (h *Hasher) Sum(b []byte) (sum []byte) {
	// We need to append h.Size() bytes to b. Reuse b's capacity if possible;
	// otherwise, allocate a new slice.
	if total := len(b) + h.Size(); cap(b) >= total {
		sum = b[:total]
	} else {
		sum = make([]byte, total)
		copy(sum, b)
	}
	// Read into the appended portion of sum. Use a low-latency-low-throughput
	// path for small digests (requiring a single compression), and a
	// high-latency-high-throughput path for large digests.
	if dst := sum[len(b):]; len(dst) <= 64 {
		out := guts.WordsToBytes(guts.CompressNode(h.rootNode()))
		copy(dst, out[:])
	} else {
		or := OutputReader{n: h.rootNode()}
		or.Read(dst)
	}
	return
}

// Reset implements hash.Hash.
func (h *Hasher) Reset() {
	h.counter = 0
	h.buflen = 0
}

// BlockSize implements hash.Hash.
func (h *Hasher) BlockSize() int { return 64 }

// Size implements hash.Hash.
func (h *Hasher) Size() int { return h.size }

// XOF returns an OutputReader initialized with the current hash state. The
// state is captured at the call: later writes to h do not affect the reader.
func (h *Hasher) XOF() *OutputReader {
	return &OutputReader{
		n: h.rootNode(),
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

// Sum256 and Sum512 always use the same hasher state, so we can save some time
// when hashing small inputs by constructing the hasher ahead of time.
var defaultHasher = New(64, nil)

// Sum256 returns the unkeyed BLAKE3 hash of b, truncated to 256 bits.
func Sum256(b []byte) (out [32]byte) {
	out512 := Sum512(b)
	copy(out[:], out512[:])
	return
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
			BlockLen: uint32(len(b)),
			Flags:    guts.FlagChunkStart | guts.FlagChunkEnd | guts.FlagRoot,
		}))
	case len(b) <= guts.ChunkSize:
		n = guts.CompressChunk(b, &guts.IV, 0, 0)
		n.Flags |= guts.FlagRoot
	default:
		h := *defaultHasher
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
	h := newHasher(guts.IV, guts.FlagDeriveKeyContext, 32)
	h.Write([]byte(ctx))
	derivationIV := h.Sum(make([]byte, 0, derivationIVLen))
	var ivWords [8]uint32
	for i := range ivWords {
		ivWords[i] = binary.LittleEndian.Uint32(derivationIV[i*4:])
	}
	h = newHasher(ivWords, guts.FlagDeriveKeyMaterial, 0)
	// derive the subKey
	h.Write(srcKey)
	h.XOF().Read(subKey)
}

// An OutputReader produces a seekable stream of 2^64 - 1 pseudorandom output
// bytes: the BLAKE3 extendable output of the state it was created from.
//
// Like a Hasher it is not safe for concurrent use, and it holds no pointers,
// so a copy continues independently from the same position.
type OutputReader struct {
	n        guts.Node
	buf      [guts.MaxSIMD * guts.BlockSize]byte
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

	const bufsize = guts.MaxSIMD * guts.BlockSize
	for len(p) > 0 {
		// drain buffered output
		if or.off >= or.bufStart && or.off-or.bufStart < uint64(or.buflen) {
			n := copy(p, or.buf[or.off-or.bufStart:or.buflen])
			p = p[n:]
			or.off += uint64(n)
			continue
		}
		if head := int(or.off % guts.BlockSize); head != 0 || len(p) < bufsize {
			// the read is small or unaligned; compress (only) as many blocks
			// as necessary into our buffer, and serve it from there
			or.bufStart = or.off - uint64(head)
			or.n.Counter = or.bufStart / guts.BlockSize
			need := min(head+len(p), bufsize)
			numBlocks := (need + guts.BlockSize - 1) / guts.BlockSize
			or.buflen = guts.BlockSize * guts.CompressBlocksN(&or.buf, or.n, numBlocks)
			continue
		}
		// the read is large and block-aligned; compress directly into p
		or.n.Counter = or.off / guts.BlockSize
		numBufs := len(p) / bufsize
		const minBufsPerCPU = (16 * 1024) / bufsize
		if par := min(numBufs/minBufsPerCPU, runtime.NumCPU()); par > 1 {
			// enough work for each CPU to be worth parallelizing; distribute
			// the buffers evenly among the goroutines
			var wg sync.WaitGroup
			for i := range par {
				bufs := uint64(numBufs / par)
				if i < numBufs%par {
					bufs++
				}
				wg.Add(1)
				go func(p []byte, n guts.Node, bufs uint64) {
					defer wg.Done()
					for i := range bufs {
						guts.CompressBlocks((*[bufsize]byte)(p[i*bufsize:]), n)
						n.Counter += bufsize / guts.BlockSize
					}
				}(p, or.n, bufs)
				p = p[bufs*bufsize:]
				or.off += bufs * bufsize
				or.n.Counter = or.off / guts.BlockSize
			}
			wg.Wait()
		} else {
			guts.CompressBlocks((*[bufsize]byte)(p), or.n)
			p = p[bufsize:]
			or.off += bufsize
		}
	}
	return lenp, nil
}

// Seek implements io.Seeker. A position before the start of the stream or
// past its end is rejected with an error and leaves the position unchanged;
// io.SeekEnd with a zero offset positions at the last byte. Positions of
// 2^63 and above are valid but cannot be represented in the int64 return
// value, which is then negative.
func (or *OutputReader) Seek(offset int64, whence int) (int64, error) {
	off := or.off
	switch whence {
	case io.SeekStart:
		if offset < 0 {
			return 0, errors.New("seek position cannot be negative")
		}
		off = uint64(offset)
	case io.SeekCurrent:
		if offset < 0 {
			if uint64(-offset) > off {
				return 0, errors.New("seek position cannot be negative")
			}
			off -= uint64(-offset)
		} else if off += uint64(offset); off < uint64(offset) {
			return 0, errors.New("seek position cannot exceed end of stream")
		}
	case io.SeekEnd:
		if offset > 0 {
			return 0, errors.New("seek position cannot exceed end of stream")
		}
		off = uint64(offset) - 1
	default:
		panic("invalid whence")
	}
	or.off = off
	// NOTE: there is no need to update or invalidate the buffer: it caches an
	// absolute range [bufStart, bufStart+buflen) of the stream, and Read only
	// serves from it when or.off falls within that range.
	return int64(or.off), nil
}

// ensure that Hasher implements hash.Hash
var _ hash.Hash = (*Hasher)(nil)
