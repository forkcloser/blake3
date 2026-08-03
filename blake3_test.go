package blake3_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/guts"
)

func toHex(data []byte) string { return hex.EncodeToString(data) }

var testVectors = func() (vecs struct {
	Key   string
	Cases []struct {
		InputLen  int    `json:"input_len"`
		Hash      string `json:"hash"`
		KeyedHash string `json:"keyed_hash"`
		DeriveKey string `json:"derive_key"`
	}
}) {
	data, err := os.ReadFile("testdata/vectors.json")
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(data, &vecs); err != nil {
		panic(err)
	}
	return
}()

var testInput = func() []byte {
	input := make([]byte, 1e6)
	for i := range input {
		input[i] = byte(i % 251)
	}
	return input
}()

func TestVectors(t *testing.T) {
	for _, vec := range testVectors.Cases {
		in := testInput[:vec.InputLen]

		// regular
		h := blake3.New(len(vec.Hash)/2, nil)
		h.Write(in)
		if out := toHex(h.Sum(nil)); out != vec.Hash {
			t.Errorf("output did not match test vector:\n\texpected: %v...\n\t     got: %v...", vec.Hash[:10], out[:10])
		}

		// keyed
		h = blake3.New(len(vec.KeyedHash)/2, []byte(testVectors.Key))
		h.Write(in)
		if out := toHex(h.Sum(nil)); out != vec.KeyedHash {
			t.Errorf("output did not match test vector:\n\texpected: %v...\n\t     got: %v...", vec.KeyedHash[:10], out[:10])
		}

		// derive key
		const ctx = "BLAKE3 2019-12-27 16:29:52 test vectors context"
		subKey := make([]byte, len(vec.DeriveKey)/2)
		blake3.DeriveKey(subKey, ctx, in)
		if out := toHex(subKey); out != vec.DeriveKey {
			t.Errorf("output did not match test vector:\n\texpected: %v...\n\t     got: %v...", vec.DeriveKey[:10], out[:10])
		}
	}
}

func TestXOF(t *testing.T) {
	for _, vec := range testVectors.Cases {
		in := testInput[:vec.InputLen]

		// XOF should produce same output as Sum, even when outputting 7 bytes at a time.
		// Read well past the digest length, so that the seek tests below stay
		// within the reference buffer.
		h := blake3.New(len(vec.Hash)/2, nil)
		h.Write(in)
		var xofBuf bytes.Buffer
		io.CopyBuffer(&xofBuf, io.LimitReader(h.XOF(), 4096), make([]byte, 7))
		if out := toHex(xofBuf.Bytes()[:len(vec.Hash)/2]); out != vec.Hash {
			t.Errorf("XOF output did not match test vector:\n\texpected: %v...\n\t     got: %v...", vec.Hash[:10], out[:10])
		}

		// Should be able to Seek around in the output stream without affecting correctness
		seeks := []struct {
			offset int64
			whence int
		}{
			{0, io.SeekStart},
			{17, io.SeekCurrent},
			{-5, io.SeekCurrent},
			{int64(h.Size()), io.SeekStart},
			{int64(h.Size()), io.SeekCurrent},
		}
		xof := h.XOF()
		outR := bytes.NewReader(xofBuf.Bytes())
		for _, s := range seeks {
			outRead := make([]byte, 10)
			xofRead := make([]byte, 10)
			offset, _ := outR.Seek(s.offset, s.whence)
			n, _ := outR.Read(outRead)
			xof.Seek(s.offset, s.whence)
			xof.Read(xofRead[:n])
			if !bytes.Equal(outRead[:n], xofRead[:n]) {
				t.Errorf("XOF output did not match test vector at offset %v:\n\texpected: %x...\n\t     got: %x...", offset, outRead[:10], xofRead[:10])
			}
		}
	}

	{
		// test multiple-buffer output
		golden := make([]byte, 1<<20)
		n := guts.CompressChunk(nil, &guts.IV, 0, 0)
		n.Flags |= guts.FlagRoot
		for i := 0; i < len(golden); i += 64 {
			block := guts.WordsToBytes(guts.CompressNode(n))
			copy(golden[i:], block[:])
			n.Counter++
		}
		got := make([]byte, 1<<20)
		blake3.New(0, nil).XOF().Read(got)
		if !bytes.Equal(golden, got) {
			t.Error("XOF output did not match golden output")
		}
	}

	// test behavior at end of stream
	xof := blake3.New(0, nil).XOF()
	buf := make([]byte, 1024)
	xof.Seek(-1000, io.SeekEnd)
	n, err := xof.Read(buf)
	if n != 1000 || err != nil {
		t.Errorf("expected (1000, nil) when reading near end of stream, got (%v, %v)", n, err)
	}
	n, err = xof.Read(buf)
	if n != 0 || !errors.Is(err, io.EOF) {
		t.Errorf("expected (0, EOF) when reading past end of stream, got (%v, %v)", n, err)
	}

	// test invalid seek offsets
	_, err = xof.Seek(-1, io.SeekStart)
	if err == nil {
		t.Error("expected invalid offset error, got nil")
	}
	xof.Seek(0, io.SeekStart)
	_, err = xof.Seek(-1, io.SeekCurrent)
	if err == nil {
		t.Error("expected invalid offset error, got nil")
	}
	_, err = xof.Seek(1, io.SeekEnd)
	if err == nil {
		t.Error("expected past-end error, got nil")
	}
	xof.Seek(-10, io.SeekEnd)
	_, err = xof.Seek(math.MaxInt64, io.SeekCurrent)
	if err == nil {
		t.Error("expected past-end error, got nil")
	}

	// test invalid seek whence
	didPanic := func() (p bool) {
		defer func() { p = recover() != nil }()
		xof.Seek(0, 17)
		return
	}()
	if !didPanic {
		t.Error("expected panic when seeking with invalid whence")
	}
}

func TestXOFSeek(t *testing.T) {
	// generate golden output, one block at a time
	golden := make([]byte, 1<<16)
	n := guts.CompressChunk(nil, &guts.IV, 0, 0)
	n.Flags |= guts.FlagRoot
	for i := 0; i < len(golden); i += guts.BlockSize {
		block := guts.WordsToBytes(guts.CompressNode(n))
		copy(golden[i:], block[:])
		n.Counter++
	}

	// seeking to any offset should produce the same output as the golden
	// stream, in particular offsets that are not aligned to the XOF's internal
	// buffer
	xof := blake3.New(0, nil).XOF()
	buf := make([]byte, 100)
	for _, off := range []int{0, 1, 63, 64, 65, 100, 131, 1000, 1023, 1024, 1025, 1100, 2047, 2048, 3000, len(golden) - len(buf)} {
		if _, err := xof.Seek(int64(off), io.SeekStart); err != nil {
			t.Fatal(err)
		} else if _, err := io.ReadFull(xof, buf); err != nil {
			t.Fatal(err)
		}
		if exp := golden[off:][:len(buf)]; !bytes.Equal(buf, exp) {
			t.Errorf("Seek(%v, io.SeekStart): expected %x..., got %x...", off, exp[:8], buf[:8])
		}
	}
	xof.Seek(0, io.SeekStart)
	io.ReadFull(xof, buf) // off = 100
	xof.Seek(100, io.SeekCurrent)
	io.ReadFull(xof, buf) // off = 300
	if exp := golden[200:][:len(buf)]; !bytes.Equal(buf, exp) {
		t.Errorf("Seek(100, io.SeekCurrent): expected %x..., got %x...", exp[:8], buf[:8])
	}

	// seek near the end of the stream, to a buffer-unaligned offset; this also
	// exercises block counters beyond 2^32
	const rem = 1500
	off := uint64(math.MaxUint64) - rem // stream ends at 2^64 - 1
	n = guts.CompressChunk(nil, &guts.IV, 0, 0)
	n.Flags |= guts.FlagRoot
	n.Counter = off / guts.BlockSize
	var endGolden []byte
	for len(endGolden) < rem+guts.BlockSize {
		block := guts.WordsToBytes(guts.CompressNode(n))
		endGolden = append(endGolden, block[:]...)
		n.Counter++
	}
	endGolden = endGolden[off%guts.BlockSize:][:rem]
	xof.Seek(-rem, io.SeekEnd)
	end := make([]byte, rem)
	if _, err := io.ReadFull(xof, end); err != nil {
		t.Fatal(err)
	} else if !bytes.Equal(end, endGolden) {
		t.Errorf("Seek(-%v, io.SeekEnd): expected %x..., got %x...", rem, endGolden[:8], end[:8])
	}
}

func TestXOFReadPatterns(t *testing.T) {
	// generate golden output, one block at a time
	golden := make([]byte, 1<<20)
	n := guts.CompressChunk(nil, &guts.IV, 0, 0)
	n.Flags |= guts.FlagRoot
	for i := 0; i < len(golden); i += guts.BlockSize {
		block := guts.WordsToBytes(guts.CompressNode(n))
		copy(golden[i:], block[:])
		n.Counter++
	}

	// interleave reads of various sizes (crossing the buffered, direct, and
	// parallel paths) with seeks, and confirm that the output always matches
	// the golden stream
	rng := rand.New(rand.NewSource(0))
	xof := blake3.New(0, nil).XOF()
	off := 0
	for range 500 {
		if rng.Intn(4) == 0 {
			off = rng.Intn(len(golden) / 2)
			xof.Seek(int64(off), io.SeekStart)
		}
		var readSize int
		switch rng.Intn(4) {
		case 0:
			readSize = 1 + rng.Intn(64)
		case 1:
			readSize = 1 + rng.Intn(2048)
		case 2:
			readSize = 1 + rng.Intn(1<<15)
		case 3:
			readSize = 1 + rng.Intn(1<<19)
		}
		readSize = min(readSize, len(golden)-off)
		buf := make([]byte, readSize)
		if _, err := io.ReadFull(xof, buf); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(buf, golden[off:][:readSize]) {
			t.Fatalf("read of %v bytes at offset %v did not match golden output", readSize, off)
		}
		off += readSize
	}
}

func TestNewValidation(t *testing.T) {
	expectPanic := func(desc string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("expected panic from %v", desc)
			}
		}()
		fn()
	}
	expectPanic("negative size", func() { blake3.New(-1, nil) })
	expectPanic("short key", func() { blake3.New(32, make([]byte, 16)) })
	expectPanic("long key", func() { blake3.New(32, make([]byte, 33)) })
}

func TestSum(t *testing.T) {
	for _, vec := range testVectors.Cases {
		in := testInput[:vec.InputLen]

		var exp256 [32]byte
		h := blake3.New(32, nil)
		h.Write(in)
		h.Sum(exp256[:0])
		if got256 := blake3.Sum256(in); exp256 != got256 {
			t.Errorf("Sum256 output did not match Sum output:\n\texpected: %x...\n\t     got: %x...", exp256[:5], got256[:5])
		}

		var exp512 [64]byte
		h = blake3.New(64, nil)
		h.Write(in)
		h.Sum(exp512[:0])
		if got512 := blake3.Sum512(in); exp512 != got512 {
			t.Errorf("Sum512 output did not match Sum output:\n\texpected: %x...\n\t     got: %x...", exp512[:5], got512[:5])
		}
	}
}

func TestReset(t *testing.T) {
	for _, vec := range testVectors.Cases {
		in := testInput[:vec.InputLen]

		h := blake3.New(32, nil)
		h.Write(in)
		out1 := h.Sum(nil)
		h.Reset()
		h.Write(in)
		out2 := h.Sum(nil)
		if !bytes.Equal(out1, out2) {
			t.Error("Reset did not reset Hasher state properly")
		}
	}

	// gotta have 100% test coverage...
	if blake3.New(0, nil).BlockSize() != 64 {
		t.Error("incorrect block size")
	}
}

func TestEigentrees(t *testing.T) {
	for i := range uint64(64) {
		for j := range uint64(64) {
			trees := guts.Eigentrees(i, j)
			x := i
			for _, tree := range trees {
				x += 1 << tree
			}
			if x != i+j {
				t.Errorf("Wrong eigentrees for %v, %v: %v", i, j, trees)
			}
		}
	}
}

func TestSplitWrite(t *testing.T) {
	in := make([]byte, 2048)
	for i := range in {
		in[i] = byte(i)
	}
	exp := blake3.Sum256(in)
	for i := range in {
		h := blake3.New(32, nil)
		h.Write(in[:i])
		h.Write(in[i:])
		if !bytes.Equal(h.Sum(nil), exp[:]) {
			t.Fatalf("split write failed at position %v", i)
		}
	}
}

type nopReader struct{}

func (nopReader) Read(p []byte) (int, error) { return len(p), nil }

func BenchmarkWrite(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(1024)
	io.CopyN(blake3.New(0, nil), nopReader{}, int64(b.N*1024))
}

func BenchmarkXOF(b *testing.B) {
	for _, size := range []int64{64, 1024, 65536, 1048576} {
		b.Run(strconv.FormatInt(size, 10), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(size)
			buf := make([]byte, size)
			xof := blake3.New(0, nil).XOF()
			for b.Loop() {
				xof.Seek(0, 0)
				xof.Read(buf)
			}
		})
	}
}

func BenchmarkSum256(b *testing.B) {
	for _, size := range []int64{64, 1024, 65536, 1048576} {
		b.Run(strconv.FormatInt(size, 10), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(size)
			buf := make([]byte, size)
			for b.Loop() {
				blake3.Sum256(buf)
			}
		})
	}
}
