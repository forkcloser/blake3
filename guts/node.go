// Package guts provides a low-level interface to the BLAKE3 cryptographic hash
// function.
package guts

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

	// minParallelBytes is the least work handed to one goroutine when a
	// large eigentree is compressed in parallel: one MaxSIMD-chunk group,
	// which measured best (32 KiB per goroutine was ~10% slower at 64 KiB
	// and 128 KiB trees). What the run-dealing buys over one goroutine per
	// group is a cap of NumCPU goroutines per tree — a 1 MiB tree spawned
	// 64 before — and a serial path for a tree that does not split.
	minParallelBytes = 16 * 1024
)

// IV is the BLAKE3 initialization vector.
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
func Eigentrees(counter uint64, chunks uint64) (trees []int) {
	for i := counter; i < counter+chunks; {
		bite := min(bits.TrailingZeros64(i), bits.Len64(counter+chunks-i)-1)
		trees = append(trees, bite)
		i += 1 << bite
	}
	return
}

// CompressEigentree compresses a buffer of 2^n chunks in parallel, returning
// their root node.
func CompressEigentree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	numChunks := uint64(len(buf) / ChunkSize)
	switch {
	case bits.OnesCount64(numChunks) != 1:
		panic("non-power-of-two eigentree size")
	case numChunks == 1:
		return CompressChunk(buf, key, counter, flags)
	case numChunks <= MaxSIMD:
		if cap(buf) < MaxSIMD*ChunkSize {
			// CompressBuffer requires a full-size buffer; copy into a
			// stack-allocated one rather than growing buf on the heap
			var tmp [MaxSIMD * ChunkSize]byte
			return CompressBuffer(&tmp, copy(tmp[:], buf), key, counter, flags)
		}
		return CompressBuffer((*[MaxSIMD * ChunkSize]byte)(buf[:MaxSIMD*ChunkSize]), len(buf), key, counter, flags)
	default:
		// One CV per MaxSIMD-chunk group; the merge below is defined over
		// these, so they are computed identically however the work is
		// spread. Spreading it one group per goroutine was the mistake:
		// a group is ~10 µs of work, and waking a thread for it costs
		// more than that (a 64 KiB tree spent ~95% of its time in the
		// scheduler). Give each goroutine a run of groups worth at least
		// minParallelBytes, and run the whole tree on the caller when it
		// does not split at least two ways.
		groups := numChunks / MaxSIMD
		cvs := make([][8]uint32, groups)
		compressGroups := func(lo, hi uint64) {
			for i := lo; i < hi; i++ {
				cvs[i] = ChainingValue(CompressBuffer((*[MaxSIMD * ChunkSize]byte)(buf[i*MaxSIMD*ChunkSize:]), MaxSIMD*ChunkSize, key, counter+(MaxSIMD*i), flags))
			}
		}
		const groupsPerGoroutine = minParallelBytes / (MaxSIMD * ChunkSize)
		if par := min(groups/groupsPerGoroutine, uint64(runtime.NumCPU())); par > 1 {
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
				go func(lo, hi uint64) {
					defer wg.Done()
					compressGroups(lo, hi)
				}(lo, hi)
				lo = hi
			}
			wg.Wait()
		} else {
			compressGroups(0, groups)
		}

		var rec func(cvs [][8]uint32) Node
		rec = func(cvs [][8]uint32) Node {
			if len(cvs) == 2 {
				return ParentNode(cvs[0], cvs[1], key, flags)
			} else if len(cvs) == MaxSIMD {
				return mergeSubtrees((*[MaxSIMD][8]uint32)(cvs), MaxSIMD, key, flags)
			}
			return ParentNode(ChainingValue(rec(cvs[:len(cvs)/2])), ChainingValue(rec(cvs[len(cvs)/2:])), key, flags)
		}
		return rec(cvs)
	}
}
