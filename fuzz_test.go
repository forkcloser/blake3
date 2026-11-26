package blake3_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/bits"
	"testing"

	"github.com/forkcloser/go-blake3/v2"
	"github.com/forkcloser/go-blake3/v2/internal/kernel"
	"github.com/forkcloser/go-blake3/v2/internal/refimpl"
)

// FuzzWriteEquivalence: however an input is cut into writes, the digest is
// the specification's. The oracle is the reference port, which shares
// nothing with the package: a Hasher written a chunk at a time goes through
// the same tree code as the one under test, and a fault in what they share
// would be invisible. The bounded tests pin fixed lengths and splits; this
// target hands the split pattern to the engine, how many writes and where
// each lands against chunk and tree boundaries, which no enumeration
// exhausts. Content does not shape the tree, so the data is a fixed
// pseudo-random buffer and the fuzzed bytes decide length and splits only.
func FuzzWriteEquivalence(f *testing.F) {
	const maxLen = 300 * kernel.ChunkSize

	data := make([]byte, maxLen)
	// LCG-filled: deterministic, cheap, no math/rand dependency in the seed.
	x := uint32(0x9E3779B9)
	for i := range data {
		x = x*1664525 + 1013904223
		data[i] = byte(x >> 24)
	}

	var key [32]byte
	copy(key[:], data)

	f.Add(uint32(64*kernel.ChunkSize), []byte{})
	f.Add(uint32(63*kernel.ChunkSize+1), []byte{1, 2})
	f.Add(uint32(200*kernel.ChunkSize), []byte{7, 0, 15, 3})
	f.Add(uint32(17*kernel.ChunkSize+5), []byte{4, 4, 4, 4, 4, 4})

	f.Fuzz(func(t *testing.T, length uint32, splits []byte) {
		n := int(length % (maxLen + 1))
		in := data[:n]

		for _, keyed := range []bool{false, true} {
			newHasher := blake3.New
			if keyed {
				newHasher = func() *blake3.Hasher { return blake3.NewKeyed(key) }
			}

			ref := refimpl.New()
			if keyed {
				ref = refimpl.NewKeyed(key)
			}

			ref.Update(in)

			want := ref.Finalize(0, blake3.Size)

			// candidate: writes cut where the fuzzed pattern says. Each
			// split byte is a fraction of what remains, in 1/16ths of a
			// chunk-multiple, so the engine can steer cuts onto and around
			// chunk and tree boundaries; a run of zeros degenerates to a
			// single write, a long pattern to many small ones.
			h := newHasher()

			rem := in
			for _, s := range splits {
				if len(rem) == 0 {
					break
				}

				cut := (int(s) * kernel.ChunkSize) / 16
				if cut == 0 || cut >= len(rem) {
					continue
				}

				h.Write(rem[:cut])
				rem = rem[cut:]
			}

			h.Write(rem)

			if got := h.Sum(nil); !bytes.Equal(got, want) {
				t.Fatalf("keyed=%v len=%d splits=%v: the digest is not the reference's", keyed, n, splits)
			}
		}
	})
}

// FuzzUnmarshalBinary: UnmarshalBinary is the package's one parser of bytes
// it did not write. Whatever it is given, it does not panic; a state it
// refuses leaves the Hasher as it was; and a state it accepts is a Hasher:
// it takes more input and sums without panicking, and what it marshals
// restores to the same digest.
func FuzzUnmarshalBinary(f *testing.F) {
	var key [32]byte
	copy(key[:], "the thirty-two bytes of this key")

	for _, h := range []*blake3.Hasher{blake3.New(), blake3.NewKeyed(key), blake3.NewDeriveKey("fuzz")} {
		for _, n := range []int{0, 1, kernel.ChunkSize, 5*kernel.ChunkSize + 7, 40 * kernel.ChunkSize} {
			h.Reset()
			h.Write(make([]byte, n))

			state, err := h.MarshalBinary()
			if err != nil {
				f.Fatal(err)
			}

			f.Add(state, uint16(n))
		}
	}

	// States that are refused, so that the refusal runs with the seeds and
	// not only under the fuzzer: one cut short, and one whose counter no
	// input reaches, every bit set, with a zero subtree for each and a
	// chunk held. Taken for a state, that one runs the tree past its
	// levels at the next Write.
	fresh, err := blake3.New().MarshalBinary()
	if err != nil {
		f.Fatal(err)
	}

	f.Add(fresh[:len(fresh)-1], uint16(0))

	const counter = math.MaxUint64

	over := binary.LittleEndian.AppendUint64(bytes.Clone(fresh[:len(fresh)-10]), counter)
	over = append(over, make([]byte, 32*bits.OnesCount64(counter))...)
	over = append(binary.LittleEndian.AppendUint16(over, kernel.ChunkSize), make([]byte, kernel.ChunkSize)...)

	f.Add(over, uint16(1))

	f.Fuzz(func(t *testing.T, state []byte, more uint16) {
		h := blake3.New()
		h.Write([]byte("abc"))

		before := h.Sum(nil)

		if err := h.UnmarshalBinary(state); err != nil {
			if got := h.Sum(nil); !bytes.Equal(got, before) {
				t.Fatal("a refused state changed the Hasher")
			}

			return
		}

		h.Write(make([]byte, int(more)))

		want := h.Sum(nil)

		again, err := h.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}

		var restored blake3.Hasher
		if err := restored.UnmarshalBinary(again); err != nil {
			t.Fatalf("the state of an accepted Hasher is refused: %v", err)
		}

		if got := restored.Sum(nil); !bytes.Equal(got, want) {
			t.Fatal("an accepted Hasher's own state restores to another digest")
		}
	})
}
