package blake3_test

// The contract of package blake3, from its sources, each rule cited:
//
//   - The BLAKE3 specification (BLAKE3-team/BLAKE3-specs) fixes what every
//     digest, keyed digest, derived key and output stream byte is. The
//     official cVectors (testdata/vectors.json) pin it at the lengths they
//     cover; the specification's reference implementation (reference_test.go)
//     is the oracle at every other length and output offset.
//   - hash.Hash (Go's hash package): Write never returns an error here (the
//     package doc says so); Sum appends the digest to its argument and does
//     not change the state; Reset returns to the initial state; Size and
//     BlockSize are the digest size asked for and 64.
//   - The package doc: a Hasher holds no pointers, so a copy forks its state;
//     New panics on a negative size or a key not 32 bytes long; a digest
//     longer than 64 bytes is the first Size bytes of the XOF.
//   - Hasher.XOF: the state captured at the call; later writes do not reach
//     the reader. OutputReader.Read returns len(p), nil except across the
//     end of the 2^64-1 byte stream, where it returns what remains and then
//     0, io.EOF. Seek: before the start or past the end is an error that
//     leaves the position alone; io.SeekEnd with 0 is the end, with -n the
//     last n bytes; an unknown whence panics.
//   - DeriveKey: the specification's derive_key mode, any output length.

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/guts"
	"github.com/forkcloser/blake3/internal/refimpl"
)

// cVectors is testdata/vectors.json, the reference implementation's test
// cVectors: for the first InputLen bytes of the counting input, 131 bytes
// of hash, keyed hash under Key, and derived key under vectorsContext, the
// context string the reference's test_vectors.json leaves implicit.
type cVectors struct {
	Key   string        `json:"key"`
	Cases []cVectorCase `json:"cases"`
}

//nolint:tagliatelle // the reference file's field names
type cVectorCase struct {
	InputLen  int    `json:"input_len"`
	Hash      string `json:"hash"`
	KeyedHash string `json:"keyed_hash"`
	DeriveKey string `json:"derive_key"`
}

const vectorsContext = "BLAKE3 2019-12-27 16:29:52 test vectors context"

func cLoadVectors(t *testing.T) cVectors {
	t.Helper()

	data, err := os.ReadFile("testdata/vectors.json")
	if err != nil {
		t.Fatal(err)
	}

	var vecs cVectors
	if err := json.Unmarshal(data, &vecs); err != nil {
		t.Fatal(err)
	}

	if len(vecs.Cases) == 0 || len(vecs.Key) != 32 || vectorsContext == "" {
		t.Fatalf("cVectors.json: %d cases, key %q, context %q", len(vecs.Cases), vecs.Key, vectorsContext)
	}

	return vecs
}

// cInput is the cVectors' input: n bytes counting 0 to 250 and around.
func cInput(n int) []byte {
	input := make([]byte, n)
	for i := range input {
		input[i] = byte(i % 251)
	}

	return input
}

func fromHex(t *testing.T, s string) []byte {
	t.Helper()

	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}

	return b
}

// TestContractVectors: the official cVectors, through every entry point that
// produces them, and through the reference port, which is checked here so
// that the bounded walk can trust it.
func TestContractVectors(t *testing.T) {
	t.Parallel()

	vecs := cLoadVectors(t)
	key := []byte(vecs.Key)

	var key32 [32]byte
	copy(key32[:], key)

	for _, vec := range vecs.Cases {
		in := cInput(vec.InputLen)
		wantHash, wantKeyed, wantDerived := fromHex(t, vec.Hash), fromHex(t, vec.KeyedHash), fromHex(t, vec.DeriveKey)

		// the reference port
		ref := refimpl.New()
		ref.Update(in)

		if got := ref.Finalize(0, len(wantHash)); !bytes.Equal(got, wantHash) {
			t.Fatalf("len %d: the reference port disagrees with the cVectors", vec.InputLen)
		}

		ref = refimpl.NewKeyed(key32)
		ref.Update(in)

		if got := ref.Finalize(0, len(wantKeyed)); !bytes.Equal(got, wantKeyed) {
			t.Fatalf("len %d: the reference port's keyed hash disagrees with the cVectors", vec.InputLen)
		}

		ref = refimpl.NewDeriveKey(vectorsContext)
		ref.Update(in)

		if got := ref.Finalize(0, len(wantDerived)); !bytes.Equal(got, wantDerived) {
			t.Fatalf("len %d: the reference port's derived key disagrees with the cVectors", vec.InputLen)
		}

		// the package: a 131-byte digest goes through the XOF path of Sum
		h := blake3.New(len(wantHash), nil)
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantHash) {
			t.Errorf("len %d: Sum(%d) = %x, want %x", vec.InputLen, len(wantHash), got[:8], wantHash[:8])
		}

		if got := blake3.Sum256(in); !bytes.Equal(got[:], wantHash[:32]) {
			t.Errorf("len %d: Sum256 = %x, want %x", vec.InputLen, got[:8], wantHash[:8])
		}

		if got := blake3.Sum512(in); !bytes.Equal(got[:], wantHash[:64]) {
			t.Errorf("len %d: Sum512 = %x, want %x", vec.InputLen, got[:8], wantHash[:8])
		}

		xof := make([]byte, len(wantHash))
		if _, err := io.ReadFull(h.XOF(), xof); err != nil || !bytes.Equal(xof, wantHash) {
			t.Errorf("len %d: XOF = %x, %v, want %x", vec.InputLen, xof[:8], err, wantHash[:8])
		}

		h = blake3.New(len(wantKeyed), key)
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantKeyed) {
			t.Errorf("len %d: keyed Sum = %x, want %x", vec.InputLen, got[:8], wantKeyed[:8])
		}

		derived := make([]byte, len(wantDerived))
		blake3.DeriveKey(derived, vectorsContext, in)

		if !bytes.Equal(derived, wantDerived) {
			t.Errorf("len %d: DeriveKey = %x, want %x", vec.InputLen, derived[:8], wantDerived[:8])
		}
	}
}

// TestContractHashHash: the hash.Hash rules. Write takes everything with a
// nil error; Sum appends to its argument, leaves the state alone and may be
// interleaved with Write; Reset is a fresh Hasher; Size and BlockSize are
// what they say.
func TestContractHashHash(t *testing.T) {
	t.Parallel()

	in := cInput(3*guts.ChunkSize + 7)
	want := refimpl.Hash(in, 32)
	wantHalf := refimpl.Hash(in[:guts.ChunkSize+1], 32)

	h := blake3.New(32, nil)
	if h.Size() != 32 || h.BlockSize() != guts.BlockSize {
		t.Fatalf("Size %d, BlockSize %d", h.Size(), h.BlockSize())
	}

	n, err := h.Write(in[:guts.ChunkSize+1])
	if n != guts.ChunkSize+1 || err != nil {
		t.Fatalf("Write = %d, %v", n, err)
	}

	// Sum appends, does not disturb, and repeats
	prefix := []byte("prefix")

	sum := h.Sum(prefix)
	if !bytes.Equal(sum[:len(prefix)], prefix) || !bytes.Equal(sum[len(prefix):], wantHalf) {
		t.Fatalf("Sum(prefix) = %x", sum)
	}

	if again := h.Sum(nil); !bytes.Equal(again, wantHalf) {
		t.Fatalf("second Sum = %x, want %x", again, wantHalf)
	}

	// the write after the Sums continues the same stream
	n, err = h.Write(in[guts.ChunkSize+1:])
	if n != len(in)-guts.ChunkSize-1 || err != nil {
		t.Fatalf("Write = %d, %v", n, err)
	}

	if got := h.Sum(nil); !bytes.Equal(got, want) {
		t.Fatalf("Sum after interleaved Sum = %x, want %x", got, want)
	}

	// Sum reuses its argument's capacity
	buf := make([]byte, 0, 64)
	if got := h.Sum(buf); &got[0] != &buf[:1][0] {
		t.Error("Sum allocated although its argument had room")
	}

	// Reset
	h.Reset()

	if got := h.Sum(nil); !bytes.Equal(got, refimpl.Hash(nil, 32)) {
		t.Fatalf("Sum after Reset = %x, want the empty hash", got)
	}

	h.Write(in)

	if got := h.Sum(nil); !bytes.Equal(got, want) {
		t.Fatalf("Sum after Reset and Write = %x, want %x", got, want)
	}

	// an empty Write changes nothing
	h.Write([]byte{})

	if got := h.Sum(nil); !bytes.Equal(got, want) {
		t.Fatalf("Sum after an empty Write = %x, want %x", got, want)
	}

	// Size 0: Sum appends nothing
	if got := blake3.New(0, nil).Sum([]byte("x")); string(got) != "x" {
		t.Fatalf("Sum of a size-0 Hasher = %q", got)
	}
}

// TestContractCopyForks: a Hasher and an OutputReader hold no pointers, so a
// copy continues on its own.
func TestContractCopyForks(t *testing.T) {
	t.Parallel()

	in := cInput(5*guts.ChunkSize + 100)

	h := blake3.New(32, nil)
	h.Write(in[:2*guts.ChunkSize+3])

	fork := *h

	h.Write(in[2*guts.ChunkSize+3:])
	fork.Write([]byte("other"))

	if got := h.Sum(nil); !bytes.Equal(got, refimpl.Hash(in, 32)) {
		t.Fatalf("original after the fork wrote: %x", got)
	}

	other := append(bytes.Clone(in[:2*guts.ChunkSize+3]), "other"...)
	if got := fork.Sum(nil); !bytes.Equal(got, refimpl.Hash(other, 32)) {
		t.Fatalf("fork after the original wrote: %x", got)
	}

	xof := blake3.New(0, nil).XOF()
	buf := make([]byte, 100)

	if _, err := io.ReadFull(xof, buf); err != nil {
		t.Fatal(err)
	}

	xofCopy := *xof

	if _, err := io.ReadFull(xof, buf); err != nil {
		t.Fatal(err)
	}

	copyBuf := make([]byte, 100)
	if _, err := io.ReadFull(&xofCopy, copyBuf); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(buf, copyBuf) {
		t.Fatal("the copied OutputReader did not continue from the same position")
	}

	if !bytes.Equal(buf, refimpl.New().Finalize(100, 100)) {
		t.Fatal("the OutputReader's bytes 100..200 are not the stream's")
	}
}

// TestContractNewPanics: New refuses what the doc says it refuses, and takes
// the rest: nil key, 32-byte key, size 0.
func TestContractNewPanics(t *testing.T) {
	t.Parallel()

	panics := func(f func()) (panicked bool) {
		defer func() { panicked = recover() != nil }()

		f()

		return false
	}

	if !panics(func() { blake3.New(-1, nil) }) {
		t.Error("New(-1, nil) did not panic")
	}

	for _, n := range []int{0, 1, 16, 31, 33, 64} {
		if !panics(func() { blake3.New(32, make([]byte, n)) }) {
			t.Errorf("New with a %d-byte key did not panic", n)
		}
	}

	if panics(func() { blake3.New(0, nil) }) || panics(func() { blake3.New(32, make([]byte, 32)) }) {
		t.Error("New refused a valid argument")
	}
}

// TestContractXOFCaptured: the XOF is the state at the call.
func TestContractXOFCaptured(t *testing.T) {
	t.Parallel()

	in := cInput(2*guts.ChunkSize + 1)

	h := blake3.New(32, nil)
	h.Write(in[:guts.ChunkSize])

	xof := h.XOF()
	h.Write(in[guts.ChunkSize:])

	got := make([]byte, 64)
	if _, err := io.ReadFull(xof, got); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(got, refimpl.Hash(in[:guts.ChunkSize], 64)) {
		t.Fatal("the XOF saw a write made after it was taken")
	}
}

// TestContractXOFEnd: the stream is 2^64-1 bytes; a read across its end
// returns what is left, then 0, io.EOF.
func TestContractXOFEnd(t *testing.T) {
	t.Parallel()

	xof := blake3.New(0, nil).XOF()

	const rem = 1000

	if _, err := xof.Seek(-rem, io.SeekEnd); err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, 2*rem)

	n, err := xof.Read(buf)
	if n != rem || err != nil {
		t.Fatalf("Read across the end = %d, %v; want %d, nil", n, err, rem)
	}

	if want := refimpl.New().Finalize(math.MaxUint64-rem, rem); !bytes.Equal(buf[:n], want) {
		t.Fatal("the last bytes of the stream are not the reference's")
	}

	if n, err = xof.Read(buf); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("Read at the end = %d, %v; want 0, io.EOF", n, err)
	}

	if n, err = xof.Read(nil); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("empty Read at the end = %d, %v; want 0, io.EOF", n, err)
	}
}

// TestContractXOFSeek: every Seek rule, with the position checked after each
// refusal, and the position returned checked against a read.
func TestContractXOFSeek(t *testing.T) {
	t.Parallel()

	xof := blake3.New(0, nil).XOF()
	ref := refimpl.New()

	// position 0 after construction
	readAt := func(want uint64) {
		t.Helper()

		// 16 bytes, or what is left of the stream before its end
		n := int(min(16, math.MaxUint64-want))

		got := make([]byte, n)
		if _, err := io.ReadFull(xof, got); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(got, ref.Finalize(want, n)) {
			t.Fatalf("the read is not the stream at %d", want)
		}

		if _, err := xof.Seek(-int64(n), io.SeekCurrent); err != nil {
			t.Fatal(err)
		}
	}

	readAt(0)

	for _, seek := range []struct {
		off    int64
		whence int
		want   uint64
	}{
		{1000, io.SeekStart, 1000},
		{17, io.SeekCurrent, 1017},
		{-5, io.SeekCurrent, 1012},
		{0, io.SeekCurrent, 1012},
		{-1, io.SeekEnd, math.MaxUint64 - 1},
		{-1000, io.SeekEnd, math.MaxUint64 - 1000},
		{0, io.SeekStart, 0},
	} {
		pos, err := xof.Seek(seek.off, seek.whence)
		if err != nil || uint64(pos) != seek.want {
			t.Fatalf("Seek(%d, %d) = %d, %v; want %d", seek.off, seek.whence, pos, err, seek.want)
		}

		readAt(seek.want)
	}

	// refusals leave the position alone (at 0 here)
	if _, err := xof.Seek(-1, io.SeekStart); err == nil {
		t.Error("Seek before the start was accepted")
	}

	if _, err := xof.Seek(-1, io.SeekCurrent); err == nil {
		t.Error("Seek before the start, relative, was accepted")
	}

	if _, err := xof.Seek(1, io.SeekEnd); err == nil {
		t.Error("Seek past the end was accepted")
	}

	if _, err := xof.Seek(-2, io.SeekEnd); err != nil {
		t.Fatal(err)
	}

	if _, err := xof.Seek(math.MaxInt64, io.SeekCurrent); err == nil {
		t.Error("Seek that wraps was accepted")
	}

	readAt(math.MaxUint64 - 2)

	// SeekEnd with 0 is the end: EOF
	if _, err := xof.Seek(0, io.SeekEnd); err != nil {
		t.Fatal(err)
	}

	if n, err := xof.Read(make([]byte, 1)); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("Read at SeekEnd 0 = %d, %v; want 0, io.EOF", n, err)
	}

	// an unknown whence panics
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Seek with whence 17 did not panic")
			}
		}()

		_, _ = xof.Seek(0, 17)
	}()
}

// TestContractDeriveKey: any output length, the prefix property of the XOF,
// and a different context or key material gives a different key.
func TestContractDeriveKey(t *testing.T) {
	t.Parallel()

	const ctx = "forkcloser blake3 2026-10-14 contract test"

	material := cInput(1500)

	ref := refimpl.NewDeriveKey(ctx)
	ref.Update(material)

	for _, n := range []int{0, 1, 31, 32, 33, 64, 65, 1000, 1025} {
		got := make([]byte, n)
		blake3.DeriveKey(got, ctx, material)

		if !bytes.Equal(got, ref.Finalize(0, n)) {
			t.Fatalf("DeriveKey of %d bytes is not the reference's", n)
		}
	}

	other := make([]byte, 32)
	blake3.DeriveKey(other, ctx+" other", material)

	same := make([]byte, 32)
	blake3.DeriveKey(same, ctx, material)

	if bytes.Equal(other, same) {
		t.Error("two contexts gave the same key")
	}

	blake3.DeriveKey(other, ctx, append(bytes.Clone(material), 1))

	if bytes.Equal(other, same) {
		t.Error("two materials gave the same key")
	}
}

// TestContractEigentreePrecondition: a length that is not a power-of-two
// number of whole chunks panics, a partial chunk included; before the check
// covered it, 1500 bytes passed as one chunk and came back as a value that
// is not BLAKE3.
func TestContractEigentreePrecondition(t *testing.T) {
	t.Parallel()

	panics := func(n int) (panicked bool) {
		defer func() { panicked = recover() != nil }()

		guts.CompressEigentree(make([]byte, n), &guts.IV, 0, 0)

		return false
	}

	for _, n := range []int{0, 1, 1500, 3 * guts.ChunkSize, 2*guts.ChunkSize + 1} {
		if !panics(n) {
			t.Errorf("CompressEigentree accepted %d bytes", n)
		}
	}

	for _, n := range []int{guts.ChunkSize, 2 * guts.ChunkSize, 16 * guts.ChunkSize, 32 * guts.ChunkSize} {
		if panics(n) {
			t.Errorf("CompressEigentree refused %d bytes", n)
		}
	}
}
