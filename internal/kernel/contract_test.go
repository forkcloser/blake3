package kernel_test

// The contract of package kernel, from its sources, each rule cited:
//
//   - The BLAKE3 specification fixes what every function computes: the
//     compression function (CompressNode, ChainingValue), a chunk's final
//     node (CompressChunk), a parent node (ParentNode), the tree over a run
//     of chunks (CompressBuffer, CompressEigentree) and the output stream's
//     blocks (CompressBlocks, CompressBlocksN). The specification's reference
//     implementation (internal/refimpl) is the oracle; its tree over n chunks
//     is the left-heavy binary tree whose left subtree holds the largest
//     power of two of chunks strictly below n.
//   - The package doc: every path, SIMD or pure Go, produces identical
//     results; IV and the flag constants are the specification's.
//   - CompressEigentree's doc: a power-of-two number of whole chunks, or a
//     panic. Eigentrees' doc: the heights that increment counter to
//     counter+chunks, each tree the largest aligned power of two that fits.
//   - CompressBlocksN's doc: at least numBlocks blocks, more only when cheap,
//     never more than MaxSIMD.
//   - BytesToWords and WordsToBytes: little-endian, each the other's inverse.

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/bits"
	"math/rand/v2"
	"testing"

	"github.com/forkcloser/blake3/internal/kernel"
	"github.com/forkcloser/blake3/internal/refimpl"
)

// refSubtree is the specification's tree over data as n whole chunks (or a
// final partial one) starting at chunk counter: the reference's chunk
// states, merged left-heavy.
func refSubtree(data []byte, key [8]uint32, counter uint64, flags uint32) refimpl.Output {
	if len(data) <= refimpl.ChunkLen {
		state := refimpl.ChunkState{Cv: key, Counter: counter, Flags: flags}
		state.Update(data)

		return state.Output()
	}

	chunks := (len(data) + refimpl.ChunkLen - 1) / refimpl.ChunkLen
	left := 1 << (bits.Len(uint(chunks-1)) - 1) // the largest power of two below chunks
	mid := left * refimpl.ChunkLen

	l := refSubtree(data[:mid], key, counter, flags).ChainingValue()
	r := refSubtree(data[mid:], key, counter+uint64(left), flags).ChainingValue()

	return refimpl.ParentOutput(l, r, key, flags)
}

func testData(n int) []byte {
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(i*7 + i/251)
	}

	return data
}

var testKey = [8]uint32{1, 2, 3, 4, 5, 6, 7, 8} //nolint:gochecknoglobals // a fixture

// TestContractConstants: IV, the flags and the sizes are the specification's.
func TestContractConstants(t *testing.T) {
	t.Parallel()

	if kernel.IV != refimpl.IV {
		t.Error("IV is not the specification's")
	}

	for name, pair := range map[string][2]uint32{
		"ChunkStart":        {kernel.FlagChunkStart, refimpl.FlagChunkStart},
		"ChunkEnd":          {kernel.FlagChunkEnd, refimpl.FlagChunkEnd},
		"Parent":            {kernel.FlagParent, refimpl.FlagParent},
		"Root":              {kernel.FlagRoot, refimpl.FlagRoot},
		"KeyedHash":         {kernel.FlagKeyedHash, refimpl.FlagKeyedHash},
		"DeriveKeyContext":  {kernel.FlagDeriveKeyContext, refimpl.FlagDeriveKeyContext},
		"DeriveKeyMaterial": {kernel.FlagDeriveKeyMaterial, refimpl.FlagDeriveKeyMaterial},
	} {
		if pair[0] != pair[1] {
			t.Errorf("Flag%s = %d, the specification's is %d", name, pair[0], pair[1])
		}
	}

	if kernel.BlockSize != refimpl.BlockLen || kernel.ChunkSize != refimpl.ChunkLen || kernel.MaxSIMD != 16 {
		t.Errorf("BlockSize %d, ChunkSize %d, MaxSIMD %d", kernel.BlockSize, kernel.ChunkSize, kernel.MaxSIMD)
	}
}

// TestContractCompressNode: the compression function on nodes of every
// shape, counters past 32 bits included; ChainingValue is its first half.
func TestContractCompressNode(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(3, 3))

	for i := range 200 {
		var n kernel.Node

		for j := range n.CV {
			n.CV[j] = rng.Uint32()
		}

		for j := range n.Block {
			n.Block[j] = rng.Uint32()
		}

		switch i % 4 {
		case 0:
			n.Counter = uint64(rng.Uint32())
		case 1:
			n.Counter = rng.Uint64()
		case 2:
			n.Counter = math.MaxUint32 + uint64(i)
		default:
			n.Counter = math.MaxUint64 - uint64(i)
		}

		n.BlockLen = uint32(rng.IntN(kernel.BlockSize + 1))
		n.Flags = rng.Uint32() & 0x7f

		want := refimpl.Compress(&n.CV, n.Block, n.Counter, n.BlockLen, n.Flags)
		if got := kernel.CompressNode(n); got != want {
			t.Fatalf("node %d: CompressNode differs from the specification", i)
		}

		cv := kernel.ChainingValue(n)
		if !bytes.Equal(cvBytes(cv), cvBytes(first8(want))) {
			t.Fatalf("node %d: ChainingValue is not the first eight words", i)
		}
	}
}

func first8(words [16]uint32) (cv [8]uint32) {
	copy(cv[:], words[:8])

	return cv
}

func cvBytes(cv [8]uint32) []byte {
	var b []byte
	for _, w := range cv {
		b = binary.LittleEndian.AppendUint32(b, w)
	}

	return b
}

// TestContractParentNode: a parent's node has the key as its chaining value,
// the two children as its block, counter 0, a full block and the parent flag
// over the caller's.
func TestContractParentNode(t *testing.T) {
	t.Parallel()

	var left, right [8]uint32
	for i := range left {
		left[i], right[i] = uint32(i+100), uint32(i+200)
	}

	for _, flags := range []uint32{0, kernel.FlagKeyedHash, kernel.FlagDeriveKeyMaterial} {
		n := kernel.ParentNode(left, right, &testKey, flags)

		var block [16]uint32
		copy(block[:8], left[:])
		copy(block[8:], right[:])

		if n.CV != testKey || n.Block != block || n.Counter != 0 || n.BlockLen != kernel.BlockSize ||
			n.Flags != flags|kernel.FlagParent {
			t.Fatalf("flags %d: ParentNode = %+v", flags, n)
		}

		want := refimpl.ParentOutput(left, right, testKey, flags)
		if kernel.ChainingValue(n) != want.ChainingValue() {
			t.Fatalf("flags %d: the parent node's output is not the specification's", flags)
		}
	}
}

// TestContractCompressEigentree: every power of two of chunks to 1024, at
// counters on both sides of 2^32 (the last aligned tree below it, where every
// lane's low counter word is at its highest, and trees above it, where the
// high word is set), is the specification's tree; anything else panics.
func TestContractCompressEigentree(t *testing.T) {
	t.Parallel()

	// To 1024 chunks: a tree of 64 SIMD groups, more groups than any host
	// here has CPUs, so the groups are dealt out several per goroutine and
	// the run loops are walked.
	data := testData(1024 * kernel.ChunkSize)

	for chunks := 1; chunks <= 1024; chunks *= 2 {
		for _, counter := range []uint64{
			0, uint64(chunks), 3 * uint64(chunks),
			(1 << 32) - uint64(chunks), 1 << 32, (1 << 40) + 5*uint64(chunks),
		} {
			for _, flags := range []uint32{0, kernel.FlagDeriveKeyMaterial} {
				want := refSubtree(data[:chunks*kernel.ChunkSize], testKey, counter, flags)

				// A slice of a larger buffer, and one with no spare capacity:
				// the SIMD kernels read a whole group, so a short tree must
				// not be reinterpreted past its own end.
				for _, in := range [][]byte{data[:chunks*kernel.ChunkSize], bytes.Clone(data[:chunks*kernel.ChunkSize])} {
					got := kernel.CompressEigentree(in, &testKey, counter, flags)
					if kernel.ChainingValue(got) != want.ChainingValue() {
						t.Fatalf(
							"%d chunks (cap %d), counter %d, flags %d: CompressEigentree is not the specification's tree",
							chunks,
							cap(in),
							counter,
							flags,
						)
					}
				}
			}
		}
	}

	panics := func(n int) (panicked bool) {
		defer func() { panicked = recover() != nil }()

		kernel.CompressEigentree(data[:n], &kernel.IV, 0, 0)

		return false
	}

	for _, n := range []int{0, 1, 1500, 3 * kernel.ChunkSize, 2*kernel.ChunkSize + 1, 100 * kernel.ChunkSize} {
		if !panics(n) {
			t.Errorf("CompressEigentree accepted %d bytes", n)
		}
	}
}

// TestContractWords: BytesToWords is little-endian and WordsToBytes its
// inverse, both ways.
func TestContractWords(t *testing.T) {
	t.Parallel()

	var block [kernel.BlockSize]byte
	for i := range block {
		block[i] = byte(i * 13)
	}

	words := kernel.BytesToWords(block)
	for i, w := range words {
		if w != binary.LittleEndian.Uint32(block[4*i:]) {
			t.Fatalf("word %d = %#x", i, w)
		}
	}

	if kernel.WordsToBytes(words) != block {
		t.Fatal("WordsToBytes(BytesToWords(b)) != b")
	}

	words[3] = 0xdeadbeef
	if kernel.BytesToWords(kernel.WordsToBytes(words)) != words {
		t.Fatal("BytesToWords(WordsToBytes(w)) != w")
	}
}
