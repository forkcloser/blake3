// Package kernel is the BLAKE3 engine behind packages blake3, bao and guts:
// the compression function, chunk and parent nodes, the SIMD kernels and the
// tree-hashing helpers, with how a tree is dealt out across goroutines. Package
// guts exports the primitives; this package is the module's own.
//
// What every function computes is fixed by the BLAKE3 specification and
// pinned by the official test vectors. On amd64 the
// AVX-512 and AVX2 kernels are selected at runtime from the CPU's features,
// with the pure-Go code as the fallback everywhere; every path produces
// identical results.
//
// Functions that take pointers read (and, where documented, write) through
// them only for the duration of the call. Nothing here is safe for concurrent
// use on shared mutable arguments.
package kernel

import (
	"math/bits"
	"runtime"
	"sync"
)

// Various constants.
const (
	FlagChunkStart = 1 << iota
	FlagChunkEnd
	FlagParent
	FlagRoot
	FlagKeyedHash
	FlagDeriveKeyContext
	FlagDeriveKeyMaterial

	BlockSize = 64
	ChunkSize = 1024

	MaxSIMD = 16 // AVX-512 vectors can store 16 words
)

// Width is the number of chunks CompressBuffer compresses at once on
// this host: MaxSIMD where a SIMD kernel is dispatched, whose lanes all run
// whatever the buffer holds, and 1 in pure Go, which goes chunk by chunk. A
// caller choosing how much input to hold back does best holding a multiple.
func Width() int {
	if kernelsDispatched() {
		return MaxSIMD
	}

	return 1
}

// ParallelGrain is the least input, in bytes, one goroutine is handed when a
// tree is compressed in parallel: the pure-Go group's worth of work on the
// dispatched path. In pure Go that is one MaxSIMD-chunk group, 15–19 µs
// (15 on a Xeon 8370C at 1 MiB, 19 on an M5 Pro at a 16 KiB write; 32 KiB
// per goroutine measured ~10% slower at 64 KiB and 128 KiB trees). Under
// the SIMD kernels it is eight groups: on the 4-vCPU Xeon 8573C runner
// where a pure-Go group is 32 µs, eight kernel groups are 33 µs under
// AVX-512 and 50 under AVX2, and with two groups a grain every write from
// 48 KiB to 128 KiB cost 6–25 µs more than its serial calls, the wake of
// its goroutines, while 1 MiB alone gained. A caller spreading trees of
// its own across goroutines should hand each at least this much.
func ParallelGrain() int {
	if kernelsDispatched() {
		return 8 * MaxSIMD * ChunkSize
	}

	return MaxSIMD * ChunkSize
}

// IV is the BLAKE3 initialization vector.
//
//nolint:gochecknoglobals // exported: an array cannot be a constant, and callers building their own trees read it
var IV = [8]uint32{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
}

// A Node represents a chunk or parent in the BLAKE3 Merkle tree.
type Node struct {
	CV       [8]uint32 // chaining value from previous node
	Block    [16]uint32
	Counter  uint64
	BlockLen uint32
	Flags    uint32
}

// ParentNode returns a Node that incorporates the chaining values of two child
// nodes.
func ParentNode(left, right [8]uint32, key *[8]uint32, flags uint32) Node {
	n := Node{
		CV:       *key,
		Counter:  0,         // counter is reset for parents
		BlockLen: BlockSize, // block is full
		Flags:    flags | FlagParent,
	}
	copy(n.Block[:8], left[:])
	copy(n.Block[8:], right[:])

	return n
}

// Eigentrees returns the sequence of eigentree heights that increment counter
// to counter+chunks.
func Eigentrees(counter, chunks uint64) []int {
	return AppendEigentrees(nil, counter, chunks)
}

// AppendEigentrees appends the sequence of eigentree heights that increment
// counter to counter+chunks to trees, and returns it: Eigentrees into a
// caller's array. The sequence climbs while counter is not yet aligned, at
// most 64 trees, then descends, at most 64 more.
func AppendEigentrees(trees []int, counter, chunks uint64) []int {
	for i := counter; i < counter+chunks; {
		bite := min(bits.TrailingZeros64(i), bits.Len64(counter+chunks-i)-1)
		trees = append(trees, bite)
		i += 1 << bite
	}

	return trees
}

// CompressEigentree compresses a buffer of 2^n whole chunks in parallel,
// returning their root node. It panics on any other length: a tree is a
// power-of-two number of chunks, and a partial chunk belongs to
// CompressChunk or CompressBuffer.
func CompressEigentree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	numChunks := uint64(len(buf) / ChunkSize)
	switch {
	case len(buf)%ChunkSize != 0 || bits.OnesCount64(numChunks) != 1:
		panic("blake3: an eigentree is a power-of-two number of whole chunks")
	case numChunks == 1:
		return CompressChunk(buf, key, counter, flags)
	case numChunks <= MaxSIMD:
		if len(buf) < MaxSIMD*ChunkSize && (simdReadsFullBuffer || cap(buf) < MaxSIMD*ChunkSize) {
			// CompressBuffer takes a full-size array. A short tree could be
			// served by reinterpreting buf's spare capacity as that array,
			// but on amd64 the SIMD kernels read all 16 KiB of it, and the
			// bytes past len(buf) are memory the caller never handed over —
			// another goroutine may be writing them, a race no detector
			// sees through assembly. So on those hosts a short tree is
			// copied into a stack scratch buffer; the generic path reads
			// only buflen and takes the reinterpretation when the capacity
			// is there.
			var tmp [MaxSIMD * ChunkSize]byte
			return CompressBuffer(&tmp, copy(tmp[:], buf), key, counter, flags)
		}

		return CompressBuffer((*[MaxSIMD * ChunkSize]byte)(buf[:MaxSIMD*ChunkSize]), len(buf), key, counter, flags)
	default:
		return compressGroups(buf, key, counter, flags, numChunks/MaxSIMD)
	}
}

// compressGroups compresses an eigentree of more than MaxSIMD chunks, in
// groups of MaxSIMD, and returns its root node.
func compressGroups(buf []byte, key *[8]uint32, counter uint64, flags uint32, groups uint64) Node {
	// One CV per MaxSIMD-chunk group; the merge below is defined over
	// these, so they are computed identically however the work is
	// spread. Spreading it one group per goroutine was the mistake:
	// waking a thread costs about what a group is worth (a 64 KiB tree
	// spent ~95% of its time in the scheduler). Give each goroutine a
	// run of groups worth at least
	// ParallelGrain, and run the whole tree on the caller when it does
	// not split at least two ways. The deal caps a tree at GOMAXPROCS
	// goroutines (a 1 MiB tree spawned 64 before): what the scheduler
	// will actually run on, a container's CPU quota and the caller's
	// setting included, where NumCPU reports the machine.
	cvs := make([][8]uint32, groups)
	// The key is a parameter, and each goroutine below gets its own copy
	// by value: captured, it would leak the caller's, and move a Hasher
	// on the caller's stack to the heap.
	compressRun := func(lo, hi uint64, key *[8]uint32) {
		for i := lo; i < hi; i++ {
			cvs[i] = ChainingValue(
				CompressBuffer(
					(*[MaxSIMD * ChunkSize]byte)(buf[i*MaxSIMD*ChunkSize:]),
					MaxSIMD*ChunkSize,
					key,
					counter+(MaxSIMD*i),
					flags,
				),
			)
		}
	}

	groupsPerGoroutine := uint64(ParallelGrain() / (MaxSIMD * ChunkSize)) // #nosec G115 -- a small positive quotient
	// #nosec G115 -- GOMAXPROCS is a small positive count
	if par := min(groups/groupsPerGoroutine, uint64(runtime.GOMAXPROCS(0))); par > 1 {
		// Deal groups out in par near-equal contiguous runs; the
		// remainder folds into the runs rather than a second spawn.
		per, extra := groups/par, groups%par

		var wg sync.WaitGroup

		for w, lo := uint64(0), uint64(0); w < par; w++ {
			hi := lo + per
			if w < extra {
				hi++
			}

			wg.Add(1)
			go func(lo, hi uint64, key [8]uint32) {
				defer wg.Done()

				compressRun(lo, hi, &key)
			}(lo, hi, *key)

			lo = hi
		}

		wg.Wait()
	} else {
		compressRun(0, groups, key)
	}

	return mergeGroups(cvs, key, flags)
}

// mergeGroups merges the chaining values of a power-of-two number of
// MaxSIMD-chunk groups into their root node.
func mergeGroups(cvs [][8]uint32, key *[8]uint32, flags uint32) Node {
	if len(cvs) == 2 {
		return ParentNode(cvs[0], cvs[1], key, flags)
	} else if len(cvs) == MaxSIMD {
		return mergeSubtrees((*[MaxSIMD][8]uint32)(cvs), MaxSIMD, key, flags)
	}

	return ParentNode(
		ChainingValue(mergeGroups(cvs[:len(cvs)/2], key, flags)),
		ChainingValue(mergeGroups(cvs[len(cvs)/2:], key, flags)),
		key,
		flags,
	)
}
