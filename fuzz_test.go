package blake3_test

import (
	"bytes"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/guts"
)

// FuzzWriteEquivalence is the fuzz form of TestEigentreeWriteEquivalence.
//
// A hash has no untrusted input format to parse, so the only bug class a
// fuzzer can find in one is an implementation divergence — here, between the
// eigentree fast path Hasher.Write takes for writes larger than a chunk and
// the chunk-at-a-time reference that never enters it. The deterministic test
// pins that at fixed lengths, split points and starting counters; this
// target hands the split *pattern* to the engine — how many writes, where
// each lands relative to chunk and tree boundaries — which is the axis the
// enumeration cannot exhaust. Content is irrelevant to the tree shape, so
// data comes from a fixed pseudo-random buffer and the fuzzed bytes decide
// only length and splits; a large input to the engine buys nothing.
func FuzzWriteEquivalence(f *testing.F) {
	const maxLen = 300 * guts.ChunkSize
	data := make([]byte, maxLen)
	// LCG-filled: deterministic, cheap, no math/rand dependency in the seed.
	x := uint32(0x9E3779B9)
	for i := range data {
		x = x*1664525 + 1013904223
		data[i] = byte(x >> 24)
	}
	key := data[:32]

	f.Add(uint32(64*guts.ChunkSize), []byte{})
	f.Add(uint32(63*guts.ChunkSize+1), []byte{1, 2})
	f.Add(uint32(200*guts.ChunkSize), []byte{7, 0, 15, 3})
	f.Add(uint32(17*guts.ChunkSize+5), []byte{4, 4, 4, 4, 4, 4})

	f.Fuzz(func(t *testing.T, length uint32, splits []byte) {
		n := int(length % (maxLen + 1))
		in := data[:n]

		for _, keyed := range []bool{false, true} {
			var k []byte
			if keyed {
				k = key
			}
			// reference: chunk-at-a-time, never the eigentree path
			ref := blake3.New(64, k)
			for b := in; len(b) > 0; {
				m := min(len(b), guts.ChunkSize)
				ref.Write(b[:m])
				b = b[m:]
			}
			want := ref.Sum(nil)

			// candidate: writes cut where the fuzzed pattern says. Each
			// split byte is a fraction of what remains, in 1/16ths of a
			// chunk-multiple, so the engine can steer cuts onto and around
			// chunk and tree boundaries; a run of zeros degenerates to a
			// single write, a long pattern to many small ones.
			h := blake3.New(64, k)
			rem := in
			for _, s := range splits {
				if len(rem) == 0 {
					break
				}
				cut := (int(s) * guts.ChunkSize) / 16
				if cut == 0 || cut >= len(rem) {
					continue
				}
				h.Write(rem[:cut])
				rem = rem[cut:]
			}
			h.Write(rem)
			if got := h.Sum(nil); !bytes.Equal(got, want) {
				t.Fatalf("keyed=%v len=%d splits=%v: eigentree path diverges from reference", keyed, n, splits)
			}
		}
	})
}
