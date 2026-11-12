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
//   - The package doc: a Hasher holds no pointers, so a copy forks its state,
//     and MarshalBinary and UnmarshalBinary carry one across processes and
//     hosts; the digest is 32 bytes, longer output is the XOF's.
//   - Hasher.XOF: the state captured at the call; later writes do not reach
//     the reader. OutputReader.Read returns len(p), nil except across the
//     end of the 2^64-1 byte stream, where it returns what remains and then
//     0, io.EOF. Seek: before the start or past the end is an error that
//     leaves the position alone; io.SeekEnd with 0 is the end, with -n the
//     last n bytes; an unknown whence panics.
//   - NewDeriveKey and DeriveKey: the specification's derive_key mode, any
//     output length. OutputReader.ReadAt: the stream at any offset, the
//     position untouched, concurrent calls allowed.

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/bits"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/forkcloser/blake3/v2"
	"github.com/forkcloser/blake3/v2/internal/kernel"
	"github.com/forkcloser/blake3/v2/internal/refimpl"
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

	var key32 [32]byte
	copy(key32[:], vecs.Key)

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

		// the package: the 32-byte digest, the one-shots, the 131 bytes
		// through the XOF
		h := blake3.New()
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantHash[:32]) {
			t.Errorf("len %d: Sum = %x, want %x", vec.InputLen, got[:8], wantHash[:8])
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

		h = blake3.NewKeyed(key32)
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantKeyed[:32]) {
			t.Errorf("len %d: keyed Sum = %x, want %x", vec.InputLen, got[:8], wantKeyed[:8])
		}

		if _, err := io.ReadFull(h.XOF(), xof); err != nil || !bytes.Equal(xof, wantKeyed) {
			t.Errorf("len %d: keyed XOF = %x, %v, want %x", vec.InputLen, xof[:8], err, wantKeyed[:8])
		}

		if got := blake3.Sum256Keyed(key32, in); !bytes.Equal(got[:], wantKeyed[:32]) {
			t.Errorf("len %d: Sum256Keyed = %x, want %x", vec.InputLen, got[:8], wantKeyed[:8])
		}

		if got := blake3.DeriveKey(vectorsContext, in, len(wantDerived)); !bytes.Equal(got, wantDerived) {
			t.Errorf("len %d: DeriveKey = %x, want %x", vec.InputLen, got[:8], wantDerived[:8])
		}

		h = blake3.NewDeriveKey(vectorsContext)
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantDerived[:32]) {
			t.Errorf("len %d: NewDeriveKey Sum = %x, want %x", vec.InputLen, got[:8], wantDerived[:8])
		}

		if _, err := io.ReadFull(h.XOF(), xof); err != nil || !bytes.Equal(xof, wantDerived) {
			t.Errorf("len %d: NewDeriveKey XOF = %x, %v, want %x", vec.InputLen, xof[:8], err, wantDerived[:8])
		}
	}
}

// TestContractHashHash: the hash.Hash rules. Write takes everything with a
// nil error; Sum appends to its argument, leaves the state alone and may be
// interleaved with Write; Reset is a fresh Hasher; Size and BlockSize are
// what they say.
func TestContractHashHash(t *testing.T) {
	t.Parallel()

	in := cInput(3*kernel.ChunkSize + 7)
	want := refimpl.Hash(in, 32)
	wantHalf := refimpl.Hash(in[:kernel.ChunkSize+1], 32)

	h := blake3.New()
	if h.Size() != blake3.Size || blake3.Size != 32 || h.BlockSize() != blake3.BlockSize || blake3.BlockSize != 64 {
		t.Fatalf("Size %d, BlockSize %d", h.Size(), h.BlockSize())
	}

	n, err := h.Write(in[:kernel.ChunkSize+1])
	if n != kernel.ChunkSize+1 || err != nil {
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
	n, err = h.Write(in[kernel.ChunkSize+1:])
	if n != len(in)-kernel.ChunkSize-1 || err != nil {
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
}

// TestContractCopyForks: a Hasher and an OutputReader hold no pointers, so a
// copy continues on its own.
func TestContractCopyForks(t *testing.T) {
	t.Parallel()

	in := cInput(5*kernel.ChunkSize + 100)

	h := blake3.New()
	h.Write(in[:2*kernel.ChunkSize+3])

	fork := *h

	h.Write(in[2*kernel.ChunkSize+3:])
	fork.Write([]byte("other"))

	if got := h.Sum(nil); !bytes.Equal(got, refimpl.Hash(in, 32)) {
		t.Fatalf("original after the fork wrote: %x", got)
	}

	other := append(bytes.Clone(in[:2*kernel.ChunkSize+3]), "other"...)
	if got := fork.Sum(nil); !bytes.Equal(got, refimpl.Hash(other, 32)) {
		t.Fatalf("fork after the original wrote: %x", got)
	}

	xof := blake3.New().XOF()
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

// TestContractXOFCaptured: the XOF is the state at the call.
func TestContractXOFCaptured(t *testing.T) {
	t.Parallel()

	in := cInput(2*kernel.ChunkSize + 1)

	h := blake3.New()
	h.Write(in[:kernel.ChunkSize])

	xof := h.XOF()
	h.Write(in[kernel.ChunkSize:])

	got := make([]byte, 64)
	if _, err := io.ReadFull(xof, got); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(got, refimpl.Hash(in[:kernel.ChunkSize], 64)) {
		t.Fatal("the XOF saw a write made after it was taken")
	}
}

// TestContractXOFEnd: the stream is 2^64-1 bytes; a read across its end
// returns what is left, then 0, io.EOF.
func TestContractXOFEnd(t *testing.T) {
	t.Parallel()

	xof := blake3.New().XOF()

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

	xof := blake3.New().XOF()
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
		if got := blake3.DeriveKey(ctx, material, n); len(got) != n || !bytes.Equal(got, ref.Finalize(0, n)) {
			t.Fatalf("DeriveKey of %d bytes is not the reference's", n)
		}

		h := blake3.NewDeriveKey(ctx)
		h.Write(material[:700])
		h.Write(material[700:])

		got := make([]byte, n)
		if _, err := io.ReadFull(h.XOF(), got); err != nil || !bytes.Equal(got, ref.Finalize(0, n)) {
			t.Fatalf("NewDeriveKey's %d bytes are not the reference's: %v", n, err)
		}
	}

	same := blake3.DeriveKey(ctx, material, 32)

	if other := blake3.DeriveKey(ctx+" other", material, 32); bytes.Equal(other, same) {
		t.Error("two contexts gave the same key")
	}

	if other := blake3.DeriveKey(ctx, append(bytes.Clone(material), 1), 32); bytes.Equal(other, same) {
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

		kernel.CompressEigentree(make([]byte, n), &kernel.IV, 0, 0)

		return false
	}

	for _, n := range []int{0, 1, 1500, 3 * kernel.ChunkSize, 2*kernel.ChunkSize + 1} {
		if !panics(n) {
			t.Errorf("CompressEigentree accepted %d bytes", n)
		}
	}

	for _, n := range []int{kernel.ChunkSize, 2 * kernel.ChunkSize, 16 * kernel.ChunkSize, 32 * kernel.ChunkSize} {
		if panics(n) {
			t.Errorf("CompressEigentree refused %d bytes", n)
		}
	}
}

// TestContractGOMAXPROCS: the digest and the output stream do not depend on
// how many CPUs the scheduler has. GOMAXPROCS decides how a large write's
// tree and a large aligned read are dealt out: at 1 both run serially, at 3
// the work does not divide evenly and the remainder folds into the first
// goroutines, and the default on a many-core machine takes neither path.
// Not parallel: it sets a process-wide value, and restores it.
func TestContractGOMAXPROCS(t *testing.T) { //nolint:paralleltest // sets GOMAXPROCS for the process
	in := make([]byte, 1<<20)
	for i := range in {
		in[i] = byte(i * 7)
	}

	const out = 64 << 10

	wantSum := refimpl.Hash(in, 32)
	wantXOF := refimpl.Hash(in, out)
	prev := runtime.GOMAXPROCS(0)

	t.Cleanup(func() { runtime.GOMAXPROCS(prev) })

	for _, procs := range []int{1, 3} {
		runtime.GOMAXPROCS(procs)

		h := blake3.New()
		if _, err := h.Write(in); err != nil {
			t.Fatalf("GOMAXPROCS=%d: Write: %v", procs, err)
		}

		if got := h.Sum(nil); !bytes.Equal(got, wantSum) {
			t.Errorf("GOMAXPROCS=%d: Sum = %x, want %x", procs, got, wantSum)
		}

		buf := make([]byte, out)
		if _, err := io.ReadFull(h.XOF(), buf); err != nil {
			t.Fatalf("GOMAXPROCS=%d: XOF read: %v", procs, err)
		}

		if !bytes.Equal(buf, wantXOF) {
			t.Errorf("GOMAXPROCS=%d: XOF output differs from the reference", procs)
		}
	}
}

// TestContractMarshal: the state round-trips through MarshalBinary and
// UnmarshalBinary in every mode and at every kind of length, the restored
// Hasher continuing the stream; AppendBinary appends; a state that is not
// one MarshalBinary produced is refused and leaves the Hasher alone.
func TestContractMarshal(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 300)

	var key [32]byte
	copy(key[:], "the thirty-two bytes of this key")

	const ctx = "forkcloser blake3 2026-11-12 marshal"

	for _, m := range []struct {
		name string
		new  func() *blake3.Hasher
		ref  func() *refimpl.Hasher
	}{
		{"plain", blake3.New, refimpl.New},
		{"keyed", func() *blake3.Hasher { return blake3.NewKeyed(key) }, func() *refimpl.Hasher { return refimpl.NewKeyed(key) }},
		{"derive", func() *blake3.Hasher { return blake3.NewDeriveKey(ctx) }, func() *refimpl.Hasher { return refimpl.NewDeriveKey(ctx) }},
	} {
		for _, at := range []int{0, 1, kernel.ChunkSize, kernel.ChunkSize + 1, 5*kernel.ChunkSize + 7, 16 * kernel.ChunkSize, 16*kernel.ChunkSize + 1, 33 * kernel.ChunkSize, len(in)} {
			ref := m.ref()
			ref.Update(in)
			want := ref.Finalize(0, 32)

			h := m.new()
			h.Write(in[:at])

			state, err := h.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}

			withPrefix, err := h.AppendBinary([]byte("prefix"))
			if err != nil || !bytes.Equal(withPrefix, append([]byte("prefix"), state...)) {
				t.Fatalf("%s at %d: AppendBinary did not append the state: %v", m.name, at, err)
			}

			var restored blake3.Hasher
			if err := restored.UnmarshalBinary(state); err != nil {
				t.Fatalf("%s at %d: UnmarshalBinary: %v", m.name, at, err)
			}

			if got := restored.Sum(nil); !bytes.Equal(got, h.Sum(nil)) {
				t.Fatalf("%s at %d: the restored digest differs", m.name, at)
			}

			h.Write(in[at:])
			restored.Write(in[at:])

			if got := restored.Sum(nil); !bytes.Equal(got, want) || !bytes.Equal(h.Sum(nil), want) {
				t.Fatalf("%s at %d: continued after restore = %x, want %x", m.name, at, got[:8], want[:8])
			}
		}
	}

	// Refusals: each leaves the Hasher as it was.
	h := blake3.New()
	h.Write(in[:3000])

	before := h.Sum(nil)

	state, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	// The empty state is the fixed fields and a zero count, on any host.
	fresh, err := blake3.New().MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	for name, bad := range map[string][]byte{
		"empty":           {},
		"short":           state[:20],
		"tag":             append([]byte("x"), state[1:]...),
		"flags":           append(append(bytes.Clone(state[:4]), 4, 0, 0, 0), state[8:]...), // FlagParent: no mode
		"trailing byte":   append(bytes.Clone(state), 0),
		"truncated held":  state[:len(state)-1],
		"held past group": append(append(bytes.Clone(fresh[:len(fresh)-2]), 0xff, 0xff), make([]byte, 65535)...),
	} {
		if h.UnmarshalBinary(bad) == nil {
			t.Errorf("UnmarshalBinary accepted a %s state", name)
		}

		if got := h.Sum(nil); !bytes.Equal(got, before) {
			t.Fatalf("a refused %s state changed the Hasher", name)
		}
	}

	// A counter past zero with nothing held is no input's state: the state
	// after more than a group, so that the counter is past zero on a host
	// of any width (a wide one holds 3000 bytes whole, at counter zero),
	// with its held bytes cut and their count zeroed, every other field as
	// written.
	past := blake3.New()
	past.Write(in[:20*kernel.ChunkSize+300])

	pastSum := past.Sum(nil)

	pastState, err := past.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	counter := binary.LittleEndian.Uint64(pastState[4+4+32:])
	if counter == 0 {
		t.Fatal("premise: a write past a group left the counter at zero")
	}

	stackEnd := 4 + 4 + 32 + 8 + 32*bits.OnesCount64(counter)
	cut := append(bytes.Clone(pastState[:stackEnd]), 0, 0)

	if past.UnmarshalBinary(cut) == nil {
		t.Error("UnmarshalBinary accepted a counter past zero with nothing held")
	}

	if got := past.Sum(nil); !bytes.Equal(got, pastSum) {
		t.Fatal("a refused state with nothing held changed the Hasher")
	}

	// A counter whose stack entries are missing.
	empty := blake3.New()

	state, err = empty.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	state[4+4+32] = 16 // counter 16, one stack entry announced, none present, none held

	if err := empty.UnmarshalBinary(state); err == nil {
		t.Error("UnmarshalBinary accepted a counter with no stack entry")
	}
}

// TestContractMarshalAcrossHosts: a state marshaled on a host of one kernel
// width restores on a host of another. The wire form carries no width: a
// narrow host's state stands at any chunk counter and holds under a chunk,
// a wide host's at a multiple of sixteen and holds up to a group. Both are
// built here from the primitives, as the other host would have written
// them, and restored on this one, whichever it is.
func TestContractMarshalAcrossHosts(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 11)

	state := func(counter uint64, cvs [][8]uint32, held []byte) []byte {
		b := []byte("b3h\x01")
		b = append(b, 0, 0, 0, 0) // unkeyed

		for _, w := range kernel.IV {
			b = binary.LittleEndian.AppendUint32(b, w)
		}

		b = binary.LittleEndian.AppendUint64(b, counter)

		for _, cv := range cvs {
			for _, w := range cv {
				b = binary.LittleEndian.AppendUint32(b, w)
			}
		}

		b = binary.LittleEndian.AppendUint16(b, uint16(len(held)))

		return append(b, held...)
	}

	cv := func(buf []byte, counter uint64) [8]uint32 {
		return kernel.ChainingValue(kernel.CompressEigentree(buf, &kernel.IV, counter, 0))
	}

	for _, tc := range []struct {
		name  string
		state []byte
		done  int // bytes of in the state stands for
	}{
		{
			// A narrow host after 5 chunks and 700 bytes: heights 0 and 2
			// occupied, the lower written first.
			name: "narrow, counter 5",
			state: state(5, [][8]uint32{
				cv(in[4*kernel.ChunkSize:5*kernel.ChunkSize], 4),
				cv(in[:4*kernel.ChunkSize], 0),
			}, in[5*kernel.ChunkSize:5*kernel.ChunkSize+700]),
			done: 5*kernel.ChunkSize + 700,
		},
		{
			// A narrow host after 17 chunks: heights 0 and 4.
			name: "narrow, counter 17",
			state: state(17, [][8]uint32{
				cv(in[16*kernel.ChunkSize:17*kernel.ChunkSize], 16),
				cv(in[:16*kernel.ChunkSize], 0),
			}, in[17*kernel.ChunkSize:17*kernel.ChunkSize+1]),
			done: 17*kernel.ChunkSize + 1,
		},
		{
			// A wide host after a group and 5000 bytes: height 4, five
			// chunks held, more than a narrow host ever holds.
			name:  "wide, counter 16",
			state: state(16, [][8]uint32{cv(in[:16*kernel.ChunkSize], 0)}, in[16*kernel.ChunkSize:16*kernel.ChunkSize+5000]),
			done:  16*kernel.ChunkSize + 5000,
		},
		{
			// A wide host holding a whole group at counter 0.
			name:  "wide, a full group held",
			state: state(0, nil, in[:16*kernel.ChunkSize]),
			done:  16 * kernel.ChunkSize,
		},
	} {
		var h blake3.Hasher
		if err := h.UnmarshalBinary(tc.state); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}

		if got := h.Sum(nil); !bytes.Equal(got, refimpl.Hash(in[:tc.done], 32)) {
			t.Fatalf("%s: digest of the restored state is not the reference's", tc.name)
		}

		// Continue in two writes, one small, so the realignment a wide host
		// does chunk by chunk runs across them.
		h.Write(in[tc.done : tc.done+300])
		h.Write(in[tc.done+300:])

		if got := h.Sum(nil); !bytes.Equal(got, refimpl.Hash(in, 32)) {
			t.Fatalf("%s: continued digest is not the reference's", tc.name)
		}
	}
}

// TestContractReadAt: the stream at any offset, the position untouched, and
// concurrent calls on one reader agree with the reference.
func TestContractReadAt(t *testing.T) {
	t.Parallel()

	in := cInput(2*kernel.ChunkSize + 9)
	ref := refimpl.New()
	ref.Update(in)

	h := blake3.New()
	h.Write(in)
	xof := h.XOF()

	if _, err := xof.Seek(77, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	for _, off := range []int64{0, 1, 63, 64, 65, 1000, 1024, 1 << 20, 1<<32*kernel.BlockSize - 1, math.MaxInt64 - 100} {
		for _, n := range []int{1, 64, 100, 1025} {
			got := make([]byte, n)

			m, err := xof.ReadAt(got, off)
			if m != n || err != nil {
				t.Fatalf("ReadAt(%d, %d) = %d, %v", n, off, m, err)
			}

			if !bytes.Equal(got, ref.Finalize(uint64(off), n)) {
				t.Fatalf("ReadAt(%d, %d) is not the stream", n, off)
			}
		}
	}

	if pos, err := xof.Seek(0, io.SeekCurrent); err != nil || pos != 77 {
		t.Fatalf("ReadAt moved the position to %d, %v", pos, err)
	}

	// A negative offset is refused outright: not read as 2^64 - 1000, which
	// would be a short read at the end of the stream and io.EOF.
	if n, err := xof.ReadAt(make([]byte, 1), -1000); n != 0 || err == nil || errors.Is(err, io.EOF) {
		t.Errorf("ReadAt at -1000 = %d, %v; want 0 and an error that is not io.EOF", n, err)
	}

	var wg sync.WaitGroup

	for i := range 8 {
		wg.Go(func() {
			got := make([]byte, 3000)

			off := int64(i * 1000)
			if _, err := xof.ReadAt(got, off); err != nil || !bytes.Equal(got, ref.Finalize(uint64(off), 3000)) {
				t.Errorf("concurrent ReadAt at %d is not the stream: %v", off, err)
			}
		})
	}

	wg.Wait()
}
