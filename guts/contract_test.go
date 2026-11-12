package guts_test

// The contract of package guts: every primitive computes what the BLAKE3
// specification says, which the reference implementation's port
// (internal/refimpl) states independently. Three chunks and a partial
// fourth, hashed three ways, meet the reference's digest.

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/forkcloser/blake3/v2/guts"
	"github.com/forkcloser/blake3/v2/internal/refimpl"
)

func TestContractPrimitives(t *testing.T) {
	t.Parallel()

	in := make([]byte, 3*guts.ChunkSize+500)
	for i := range in {
		in[i] = byte(i * 11)
	}

	want := refimpl.Hash(in, 64)

	// Chunk by chunk, then the tree by hand: ((0, 1), (2, 3)).
	var cvs [4][8]uint32

	for i := range cvs {
		chunk := in[i*guts.ChunkSize : min((i+1)*guts.ChunkSize, len(in))]
		cvs[i] = guts.ChainingValue(guts.CompressChunk(chunk, &guts.IV, uint64(i), 0))
	}

	left := guts.ChainingValue(guts.ParentNode(cvs[0], cvs[1], &guts.IV, 0))
	right := guts.ChainingValue(guts.ParentNode(cvs[2], cvs[3], &guts.IV, 0))
	root := guts.ParentNode(left, right, &guts.IV, guts.FlagRoot)
	out := wordsToBytes(guts.CompressNode(root))

	if !bytes.Equal(out[:], want) {
		t.Errorf("by hand: %x, want %x", out, want)
	}

	// The whole input through CompressBuffer, one group.
	var buf [guts.MaxSIMD * guts.ChunkSize]byte

	n := guts.CompressBuffer(&buf, copy(buf[:], in), &guts.IV, 0, 0)
	n.Flags |= guts.FlagRoot

	if out = wordsToBytes(guts.CompressNode(n)); !bytes.Equal(out[:], want) {
		t.Errorf("CompressBuffer: %x, want %x", out, want)
	}

	// The two whole-chunk eigentrees, then the tail: Eigentrees says [1, 0]
	// for two chunks from an aligned counter and the partial chunk follows.
	if trees := guts.Eigentrees(0, 3); len(trees) != 2 || trees[0] != 1 || trees[1] != 0 {
		t.Fatalf("Eigentrees(0, 3) = %v, want [1 0]", trees)
	}

	pair := guts.ChainingValue(guts.CompressEigentree(in[:2*guts.ChunkSize], &guts.IV, 0, 0))
	third := guts.ChainingValue(guts.CompressEigentree(in[2*guts.ChunkSize:3*guts.ChunkSize], &guts.IV, 2, 0))
	tail := guts.ChainingValue(guts.CompressChunk(in[3*guts.ChunkSize:], &guts.IV, 3, 0))
	root = guts.ParentNode(pair, guts.ChainingValue(guts.ParentNode(third, tail, &guts.IV, 0)), &guts.IV, guts.FlagRoot)

	if out = wordsToBytes(guts.CompressNode(root)); !bytes.Equal(out[:], want) {
		t.Errorf("eigentrees: %x, want %x", out, want)
	}
}

// wordsToBytes is a compression's output as the digest's bytes.
func wordsToBytes(words [16]uint32) (out [64]byte) {
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[4*i:], w)
	}

	return out
}
