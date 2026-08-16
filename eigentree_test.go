package blake3_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/guts"
)

// TestEigentreeWriteEquivalence pins the eigentree fast path in Hasher.Write
// to the chunk-at-a-time reference.
//
// A Write larger than one chunk carves its input into eigentrees, each
// compressed as one subtree; the sequence of tree heights, and therefore
// which subtree CVs get merged in which order, depends on the chunk counter
// at the moment of the Write. Feeding the same bytes one chunk at a time
// never enters that path (each write is <= ChunkSize), so it exercises only
// the plain CompressChunk + pushSubtree tree-builder — the reference the
// eigentree path is an optimization of, and independent of it.
//
// Lengths cover every eigentree shape up to 64 chunks plus a spread beyond
// (single trees, cascades, the SIMD boundary at 16 chunks); the split points
// place the large Write at counters 0..k so the height sequence varies; and
// the keyed flag rides along.
func TestEigentreeWriteEquivalence(t *testing.T) {
	const maxLen = 300 * guts.ChunkSize
	in := make([]byte, maxLen)
	rng := rand.New(rand.NewSource(1))
	rng.Read(in)
	key := in[:32]

	// reference: chunk-at-a-time, never touching the eigentree path
	ref := func(h *blake3.Hasher, b []byte) []byte {
		for len(b) > 0 {
			n := min(len(b), guts.ChunkSize)
			h.Write(b[:n])
			b = b[n:]
		}
		return h.Sum(nil)
	}

	var lens []int
	for c := 1; c <= 64; c++ { // every chunk count to 64, plus off-by-one bytes
		lens = append(lens, c*guts.ChunkSize, c*guts.ChunkSize+1, c*guts.ChunkSize-1)
	}
	for _, c := range []int{65, 96, 127, 128, 129, 200, 255, 256, 257, 300} {
		lens = append(lens, c*guts.ChunkSize, c*guts.ChunkSize+7)
	}

	names := []string{"plain", "keyed"}
	ctors := []func() *blake3.Hasher{
		func() *blake3.Hasher { return blake3.New(64, nil) },
		func() *blake3.Hasher { return blake3.New(64, key) },
	}
	for _, l := range lens {
		if l > maxLen || l <= 0 {
			continue
		}
		data := in[:l]
		for hi, ctor := range ctors {
			want := ref(ctor(), data)

			// one-shot Write: the whole eigentree cascade at counter 0
			got := ctor()
			got.Write(data)
			if s := got.Sum(nil); !bytes.Equal(s, want) {
				t.Fatalf("%s len=%d one-shot: mismatch", names[hi], l)
			}

			// prefix of k chunks (+/- a few bytes) written first, so the
			// large Write starts at a non-zero, non-power-of-two counter and
			// possibly mid-chunk; then the remainder in one Write.
			for _, split := range []int{
				1, guts.ChunkSize - 1, guts.ChunkSize, guts.ChunkSize + 1,
				3 * guts.ChunkSize, 5*guts.ChunkSize + 100, 7 * guts.ChunkSize,
				15*guts.ChunkSize + 3, 17 * guts.ChunkSize, 33 * guts.ChunkSize,
			} {
				if split >= l {
					continue
				}
				h := ctor()
				h.Write(data[:split])
				h.Write(data[split:])
				if s := h.Sum(nil); !bytes.Equal(s, want) {
					t.Fatalf("%s len=%d split=%d: mismatch", names[hi], l, split)
				}
			}
			// and a few random splits per length
			for range 3 {
				split := rng.Intn(l)
				h := ctor()
				h.Write(data[:split])
				h.Write(data[split:])
				if s := h.Sum(nil); !bytes.Equal(s, want) {
					t.Fatalf("%s len=%d rand split=%d: mismatch", names[hi], l, split)
				}
			}
		}
	}
}
