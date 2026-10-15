package blake3_test

// The bounded walk: every input length around every boundary the
// implementation has (a block, a chunk, the SIMD group of 16 chunks, the
// 24 KiB parallel-write threshold, powers of two of chunks and their
// neighbours), in each of the three modes, written whole and split at every
// kind of boundary; and every output offset and read size around the
// boundaries of the XOF (a block, the 1 KiB output buffer, the 16 KiB
// parallel-read grain). The oracle is the reference port at every point.

import (
	"bytes"
	"io"
	"math"
	"math/rand/v2"
	"strconv"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/guts"
	"github.com/forkcloser/blake3/internal/refimpl"
)

// inputLengths are the lengths walked: each boundary, one step either side.
func inputLengths() []int {
	var lengths []int

	add := func(n int) {
		for _, d := range []int{-1, 0, 1} {
			if n+d >= 0 {
				lengths = append(lengths, n+d)
			}
		}
	}

	add(0)
	add(guts.BlockSize)
	add(2 * guts.BlockSize)
	add(guts.ChunkSize)
	add(2 * guts.ChunkSize)
	add(3 * guts.ChunkSize)

	for c := 4; c <= 128; c *= 2 { // powers of two of chunks: 4 .. 128
		add(c * guts.ChunkSize)
		add((c + 1) * guts.ChunkSize)
		add((c - 1) * guts.ChunkSize)
	}

	add(guts.MaxSIMD * guts.ChunkSize)     // the SIMD group
	add(24 * 1024)                         // the parallel-write threshold
	add(32 * 1024)                         // io.Copy's buffer
	add(100 * guts.ChunkSize)              // not a power of two, a cascade of trees
	add(2*guts.MaxSIMD*guts.ChunkSize + 1) // two groups and a byte
	// A tree of 64 groups: more groups than any host here has CPUs, so the
	// groups are dealt out in runs of several and the run loops are walked,
	// which a tree of one group per CPU never does.
	add(1024 * guts.ChunkSize)
	add(1025 * guts.ChunkSize)

	return lengths
}

// modes are the three hashing modes, each as the package and the reference
// build it.
type mode struct {
	name string
	new  func() *blake3.Hasher
	ref  func() *refimpl.Hasher
	sum  func(in []byte, n int) []byte // the one-shot form, nil where there is none
}

func modes() []mode {
	var key [32]byte
	for i := range key {
		key[i] = byte(i * 7)
	}

	const ctx = "forkcloser blake3 2026-10-14 bounded walk"

	return []mode{
		{
			name: "plain",
			new:  func() *blake3.Hasher { return blake3.New(64, nil) },
			ref:  refimpl.New,
			sum: func(in []byte, n int) []byte {
				if n == 32 {
					s := blake3.Sum256(in)

					return s[:]
				}

				s := blake3.Sum512(in)

				return s[:]
			},
		},
		{
			name: "keyed",
			new:  func() *blake3.Hasher { return blake3.New(64, key[:]) },
			ref:  func() *refimpl.Hasher { return refimpl.NewKeyed(key) },
		},
		{
			name: "derive",
			ref:  func() *refimpl.Hasher { return refimpl.NewDeriveKey(ctx) },
			sum: func(in []byte, n int) []byte {
				out := make([]byte, n)
				blake3.DeriveKey(out, ctx, in)

				return out
			},
		},
	}
}

// splits are where a length is cut into writes: at chunk boundaries, one
// byte either side of them, at block boundaries, in halves, and in a long
// run of small writes.
func splits(n int) [][]int {
	cuts := [][]int{nil} // one write

	for _, at := range []int{1, guts.BlockSize, guts.ChunkSize - 1, guts.ChunkSize, guts.ChunkSize + 1, 3*guts.ChunkSize + 5, 16 * guts.ChunkSize, n / 2} {
		if at > 0 && at < n {
			cuts = append(cuts, []int{at})
		}
	}

	if n > 2*guts.ChunkSize+2 {
		cuts = append(cuts, []int{guts.ChunkSize - 1, 2 * guts.ChunkSize, 2*guts.ChunkSize + 2})
	}

	if n > 300 {
		var every []int
		for at := 100; at < n; at += 100 {
			every = append(every, at)
		}

		cuts = append(cuts, every)
	}

	return cuts
}

// write feeds in to h cut at the given points.
func write(h *blake3.Hasher, in []byte, cuts []int) {
	prev := 0
	for _, at := range cuts {
		h.Write(in[prev:at])
		prev = at
	}

	h.Write(in[prev:])
}

// TestBoundedLengths: every length, every mode, every split: the 64-byte
// digest, the 32-byte one-shot, and the first 200 bytes of the XOF, each the
// reference's.
func TestBoundedLengths(t *testing.T) {
	t.Parallel()

	data := make([]byte, 1026*guts.ChunkSize)
	if _, err := rand.NewChaCha8([32]byte{2}).Read(data); err != nil {
		t.Fatal(err)
	}

	for _, m := range modes() {
		t.Run(m.name, func(t *testing.T) {
			t.Parallel()

			for _, n := range inputLengths() {
				in := data[:n]

				ref := m.ref()
				ref.Update(in)

				want64 := ref.Finalize(0, 64)
				want200 := ref.Finalize(0, 200)

				if m.sum != nil {
					if got := m.sum(in, 32); !bytes.Equal(got, want64[:32]) {
						t.Fatalf("len %d: one-shot 32 = %x, want %x", n, got[:8], want64[:8])
					}

					if got := m.sum(in, 64); !bytes.Equal(got, want64) {
						t.Fatalf("len %d: one-shot 64 = %x, want %x", n, got[:8], want64[:8])
					}
				}

				if m.new == nil {
					continue
				}

				for _, cuts := range splits(n) {
					h := m.new()
					write(h, in, cuts)

					if got := h.Sum(nil); !bytes.Equal(got, want64) {
						t.Fatalf("len %d, cuts %v: Sum = %x, want %x", n, cuts, got[:8], want64[:8])
					}

					got := make([]byte, 200)
					if _, err := io.ReadFull(h.XOF(), got); err != nil || !bytes.Equal(got, want200) {
						t.Fatalf("len %d, cuts %v: XOF = %x, %v, want %x", n, cuts, got[:8], err, want200[:8])
					}
				}
			}
		})
	}
}

// outputOffsets are the XOF positions walked: each boundary of the output
// path, one step either side, and the same near the end of the stream.
func outputOffsets() []uint64 {
	var offsets []uint64

	for _, b := range []uint64{0, guts.BlockSize, 2 * guts.BlockSize, 1024, 2048, 16 * 1024, 32 * 1024, 64 * 1024} {
		for _, d := range []int64{-1, 0, 1} {
			if int64(b)+d >= 0 {
				offsets = append(offsets, uint64(int64(b)+d))
			}
		}
	}

	offsets = append(
		offsets,
		1<<32*guts.BlockSize-1,
		1<<32*guts.BlockSize,
		1<<32*guts.BlockSize+1,
	) // a block counter past 32 bits

	return offsets
}

// readSizes are the read lengths walked: around a block, around the 1 KiB
// output buffer, around the 16 KiB grain, and a read big enough to split
// across CPUs.
func readSizes() []int {
	return []int{1, 63, 64, 65, 1023, 1024, 1025, 2047, 2048, 16*1024 - 1, 16 * 1024, 16*1024 + 1, 64*1024 + 3}
}

// TestBoundedXOFReads: from every offset, every read size, one Read, the
// bytes are the reference's; then a second Read continues the stream.
func TestBoundedXOFReads(t *testing.T) {
	t.Parallel()

	in := cInput(guts.ChunkSize + 1)
	ref := refimpl.New()
	ref.Update(in)

	h := blake3.New(32, nil)
	h.Write(in)

	for _, off := range outputOffsets() {
		for _, size := range readSizes() {
			xof := h.XOF()
			if _, err := xof.Seek(int64(off), io.SeekStart); err != nil {
				t.Fatal(err)
			}

			got := make([]byte, size)
			if _, err := io.ReadFull(xof, got); err != nil {
				t.Fatalf("offset %d, size %d: %v", off, size, err)
			}

			if !bytes.Equal(got, ref.Finalize(off, size)) {
				t.Fatalf("offset %d, size %d: the read is not the stream", off, size)
			}

			next := make([]byte, 100)
			if _, err := io.ReadFull(xof, next); err != nil {
				t.Fatal(err)
			}

			if !bytes.Equal(
				next,
				ref.Finalize(off+uint64(size), 100),
			) {
				t.Fatalf("offset %d, size %d: the following read is not the stream", off, size)
			}
		}
	}
}

// TestBoundedXOFEnd: reads of every size that end at the end of the stream
// return what is left, then io.EOF.
func TestBoundedXOFEnd(t *testing.T) {
	t.Parallel()

	ref := refimpl.New()

	for _, rem := range []int{1, 63, 64, 65, 1023, 1024, 1025, 20000} {
		xof := blake3.New(0, nil).XOF()
		if _, err := xof.Seek(-int64(rem), io.SeekEnd); err != nil {
			t.Fatal(err)
		}

		got := make([]byte, rem+10)

		n, err := xof.Read(got)
		if n != rem || err != nil {
			t.Fatalf("rem %d: Read = %d, %v", rem, n, err)
		}

		if !bytes.Equal(
			got[:n],
			ref.Finalize(math.MaxUint64-uint64(rem), rem),
		) {
			t.Fatalf("rem %d: the last bytes are not the stream's", rem)
		}

		if n, err := xof.Read(
			got,
		); n != 0 ||
			err != io.EOF { //nolint:errorlint // io.EOF is returned bare, as io.Reader requires
			t.Fatalf("rem %d: Read at the end = %d, %v", rem, n, err)
		}
	}
}

// TestBoundedXOFSeekSequences: every sequence of up to three seeks from a
// small set, each followed by a short read, held to the reference, so the
// buffer's window is tested against every way of moving around it.
func TestBoundedXOFSeekSequences(t *testing.T) {
	t.Parallel()

	ref := refimpl.New()

	type seek struct {
		off    int64
		whence int
	}

	seeks := []seek{
		{0, io.SeekStart},
		{63, io.SeekStart},
		{1000, io.SeekStart},
		{1024, io.SeekStart},
		{5000, io.SeekStart},
		{1, io.SeekCurrent},
		{-1, io.SeekCurrent},
		{700, io.SeekCurrent},
		{-700, io.SeekCurrent},
		{-100, io.SeekEnd},
	}

	var walk func(xof *blake3.OutputReader, pos uint64, depth int, path string)

	walk = func(xof *blake3.OutputReader, pos uint64, depth int, path string) {
		if depth == 0 {
			return
		}

		for _, s := range seeks {
			copied := *xof

			var want uint64

			switch s.whence {
			case io.SeekStart:
				want = uint64(s.off)
			case io.SeekCurrent:
				if (s.off < 0 && uint64(-s.off) > pos) || (s.off > 0 && pos > math.MaxUint64-uint64(s.off)) {
					continue // before the start or past the end: TestContractXOFSeek's refusals
				}

				want = pos + uint64(s.off)
			default:
				want = math.MaxUint64 - uint64(-s.off)
			}

			if want > math.MaxUint64-70 {
				continue // the read after the seek would cross the end, TestBoundedXOFEnd's case
			}

			name := path + "/" + strconv.FormatInt(s.off, 10) + ":" + strconv.Itoa(s.whence)

			got, err := copied.Seek(s.off, s.whence)
			if err != nil || uint64(got) != want {
				t.Fatalf("%s: Seek = %d, %v; want %d", name, got, err, want)
			}

			buf := make([]byte, 70)
			if _, err := io.ReadFull(&copied, buf); err != nil {
				t.Fatalf("%s: %v", name, err)
			}

			if !bytes.Equal(buf, ref.Finalize(want, 70)) {
				t.Fatalf("%s: the read after the seek is not the stream at %d", name, want)
			}

			walk(&copied, want+70, depth-1, name)
		}
	}

	walk(blake3.New(0, nil).XOF(), 0, 3, "")
}
