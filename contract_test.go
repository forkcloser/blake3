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
	"strconv"
	"sync"
	"testing"

	"github.com/forkcloser/blake3/v2"
	"github.com/forkcloser/blake3/v2/guts"
	"github.com/forkcloser/blake3/v2/internal/kernel"
	"github.com/forkcloser/blake3/v2/internal/refimpl"
)

// cVectors is testdata/vectors.json, the BLAKE3 team's test vectors as they
// publish them (BLAKE3-team/BLAKE3, test_vectors/test_vectors.json at
// ae3e8e6b3a5ae3190ca5d62820789b17886a0038): for the first InputLen bytes
// of the counting input, 131 bytes of hash, keyed hash under Key, and
// derived key under Context.
//
//nolint:tagliatelle // the reference file's field names
type cVectors struct {
	Key     string        `json:"key"`
	Context string        `json:"context_string"`
	Cases   []cVectorCase `json:"cases"`
}

//nolint:tagliatelle // the reference file's field names
type cVectorCase struct {
	InputLen  int    `json:"input_len"`
	Hash      string `json:"hash"`
	KeyedHash string `json:"keyed_hash"`
	DeriveKey string `json:"derive_key"`
}

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

	// The published file has 35 cases: a file with fewer is not it.
	if len(vecs.Cases) != 35 || len(vecs.Key) != 32 || vecs.Context == "" {
		t.Fatalf("vectors.json: %d cases, key %q, context %q", len(vecs.Cases), vecs.Key, vecs.Context)
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

// marshal is h's state, as MarshalBinary gives it.
func marshal(t *testing.T, h *blake3.Hasher) []byte {
	t.Helper()

	state, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	return state
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

		ref = refimpl.NewDeriveKey(vecs.Context)
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

		if got := blake3.DeriveKey(vecs.Context, in, len(wantDerived)); !bytes.Equal(got, wantDerived) {
			t.Errorf("len %d: DeriveKey = %x, want %x", vec.InputLen, got[:8], wantDerived[:8])
		}

		h = blake3.NewDeriveKey(vecs.Context)
		h.Write(in)

		if got := h.Sum(nil); !bytes.Equal(got, wantDerived[:32]) {
			t.Errorf("len %d: NewDeriveKey Sum = %x, want %x", vec.InputLen, got[:8], wantDerived[:8])
		}

		if _, err := io.ReadFull(h.XOF(), xof); err != nil || !bytes.Equal(xof, wantDerived) {
			t.Errorf("len %d: NewDeriveKey XOF = %x, %v, want %x", vec.InputLen, xof[:8], err, wantDerived[:8])
		}
	}
}

// TestContractZeroHasher: "the zero value is the plain hash, as New returns
// it": written to, summed, read as an XOF, marshalled, reset, and as the
// target of a restore, it is New's Hasher at every length.
func TestContractZeroHasher(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 300)

	for _, n := range []int{0, 1, kernel.ChunkSize, kernel.ChunkSize + 1, 16*kernel.ChunkSize + 1, len(in)} {
		want := refimpl.Hash(in[:n], 100)

		var zero blake3.Hasher

		zero.Write(in[:n])

		if got := zero.Sum(nil); !bytes.Equal(got, want[:32]) {
			t.Fatalf("%d bytes: the zero Hasher's digest is not BLAKE3's", n)
		}

		xof := make([]byte, 100)
		if _, err := io.ReadFull(zero.XOF(), xof); err != nil || !bytes.Equal(xof, want) {
			t.Fatalf("%d bytes: the zero Hasher's output is not BLAKE3's: %v", n, err)
		}

		fromNew := blake3.New()
		fromNew.Write(in[:n])

		zeroState, zerr := zero.MarshalBinary()
		newState, nerr := fromNew.MarshalBinary()

		if zerr != nil || nerr != nil || !bytes.Equal(zeroState, newState) {
			t.Fatalf("%d bytes: the zero Hasher's state is not New's: %v, %v", n, zerr, nerr)
		}

		zero.Reset()
		zero.Write([]byte("abc"))

		if got := zero.Sum(nil); !bytes.Equal(got, refimpl.Hash([]byte("abc"), 32)) {
			t.Fatalf("%d bytes: the zero Hasher after Reset is not the plain hash", n)
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

// TestContractReset: hash.Hash's "Reset resets the Hash to its initial
// state", which for a keyed or a key-derivation Hasher is that mode under
// that key, whatever was written before: nothing, less than a chunk, a
// stack of subtrees with a tail held. A Hasher restored from a state resets
// to the mode and key the state carried.
func TestContractReset(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 300)
	after := []byte("abc")

	for _, m := range modes() {
		wantEmpty := m.ref().Finalize(0, 32)

		ref := m.ref()
		ref.Update(after)
		want := ref.Finalize(0, 32)

		fresh := marshal(t, m.new())

		for _, n := range []int{0, 1, kernel.ChunkSize, 16*kernel.ChunkSize + 1, len(in)} {
			constructed := m.new()
			constructed.Write(in[:n])

			var restored blake3.Hasher
			if err := restored.UnmarshalBinary(marshal(t, constructed)); err != nil {
				t.Fatalf("%s after %d bytes: UnmarshalBinary: %v", m.name, n, err)
			}

			for _, c := range []struct {
				name string
				h    *blake3.Hasher
			}{{"constructed", constructed}, {"restored", &restored}} {
				c.h.Reset()

				if got := c.h.Sum(nil); !bytes.Equal(got, wantEmpty) {
					t.Fatalf("%s, %s after %d bytes: Reset is not the mode's hash of nothing", m.name, c.name, n)
				}

				if got := marshal(t, c.h); !bytes.Equal(got, fresh) {
					t.Fatalf("%s, %s after %d bytes: Reset is not the constructor's state", m.name, c.name, n)
				}

				c.h.Write(after)

				if got := c.h.Sum(nil); !bytes.Equal(got, want) {
					t.Fatalf("%s, %s after %d bytes: a write after Reset is not hashed in the mode", m.name, c.name, n)
				}
			}
		}
	}
}

// TestContractArguments: io.Writer's "Write must not modify the slice data,
// even temporarily" and "must not retain p", where Write holds input back
// and so has a reason to; and encoding.BinaryUnmarshaler's "UnmarshalBinary
// must copy the data if it wishes to retain the data after returning". The
// argument is the same after the call, and overwriting it afterwards
// changes nothing of the Hasher: at lengths that are held whole, compressed
// serially, and dealt out to goroutines.
func TestContractArguments(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 300)
	rest := []byte("and the rest")

	scribble := func(b []byte) {
		for i := range b {
			b[i] ^= 0xa5
		}
	}

	const group = kernel.MaxSIMD * kernel.ChunkSize

	for _, m := range modes() {
		for _, n := range []int{1, kernel.ChunkSize, kernel.ChunkSize + 1, group, group + 1, len(in)} {
			ref := m.ref()
			ref.Update(in[:n])
			want := ref.Finalize(0, 32)
			ref.Update(rest)
			wantRest := ref.Finalize(0, 32)

			arg := bytes.Clone(in[:n])

			h := m.new()
			h.Write(arg)

			if !bytes.Equal(arg, in[:n]) {
				t.Fatalf("%s: Write changed its %d-byte argument", m.name, n)
			}

			scribble(arg)

			if got := h.Sum(nil); !bytes.Equal(got, want) {
				t.Fatalf("%s: overwriting Write's %d-byte argument after the call changed the digest", m.name, n)
			}

			state := marshal(t, h)
			data := bytes.Clone(state)

			var restored blake3.Hasher
			if err := restored.UnmarshalBinary(data); err != nil {
				t.Fatalf("%s at %d: UnmarshalBinary: %v", m.name, n, err)
			}

			if !bytes.Equal(data, state) {
				t.Fatalf("%s at %d: UnmarshalBinary changed its argument", m.name, n)
			}

			scribble(data)

			for name, c := range map[string]*blake3.Hasher{"written": h, "restored": &restored} {
				if got := c.Sum(nil); !bytes.Equal(got, want) {
					t.Fatalf("%s at %d: the %s Hasher follows its argument", m.name, n, name)
				}

				c.Write(rest)

				if got := c.Sum(nil); !bytes.Equal(got, wantRest) {
					t.Fatalf("%s at %d: the %s Hasher, continued, follows its argument", m.name, n, name)
				}
			}
		}
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

// TestContractXOFLeavesHasher: XOF captures the state and leaves the Hasher
// as it was. In each mode the digest, the state and the writes that follow
// are those of a Hasher XOF was never called on, and neither those writes
// nor a Reset reach the reader.
func TestContractXOFLeavesHasher(t *testing.T) {
	t.Parallel()

	in := cInput(18*kernel.ChunkSize + 9)

	for _, m := range modes() {
		for _, n := range []int{0, 5, kernel.ChunkSize, 17*kernel.ChunkSize + 1} {
			ref := m.ref()
			ref.Update(in[:n])
			wantAt := ref.Finalize(0, 100)
			ref.Update(in[n:])
			want := ref.Finalize(0, 32)

			h := m.new()
			h.Write(in[:n])

			untouched := *h
			xof := h.XOF()

			if got := h.Sum(nil); !bytes.Equal(got, wantAt[:32]) {
				t.Fatalf("%s at %d: Sum after XOF is not the digest", m.name, n)
			}

			if !bytes.Equal(marshal(t, h), marshal(t, &untouched)) {
				t.Fatalf("%s at %d: XOF changed the Hasher's state", m.name, n)
			}

			h.Write(in[n:])

			if got := h.Sum(nil); !bytes.Equal(got, want) {
				t.Fatalf("%s at %d: a write after XOF is not hashed as one before it", m.name, n)
			}

			h.Reset()

			got := make([]byte, 100)
			if _, err := io.ReadFull(xof, got); err != nil || !bytes.Equal(got, wantAt) {
				t.Fatalf("%s at %d: the reader followed its Hasher: %v", m.name, n, err)
			}
		}
	}
}

// TestContractXOFEmptyRead: io.Reader allows 0, nil for a Read of no bytes
// and nowhere else, and a caller must not take it for the end of the
// stream. Away from the end an empty Read is 0, nil, never io.EOF, and does
// not move the position: at the start, inside a block, on a block and on a
// buffer boundary, and with one byte left.
func TestContractXOFEmptyRead(t *testing.T) {
	t.Parallel()

	ref := refimpl.New()

	for _, seek := range []struct {
		off    int64
		whence int
	}{
		{0, io.SeekStart},
		{1, io.SeekStart},
		{kernel.BlockSize, io.SeekStart},
		{100, io.SeekStart},
		{16 * kernel.BlockSize, io.SeekStart},
		{1 << 40, io.SeekStart},
		{-1, io.SeekEnd},
	} {
		xof := blake3.New().XOF()

		pos, err := xof.Seek(seek.off, seek.whence)
		if err != nil {
			t.Fatal(err)
		}

		at := uint64(pos)

		// Before anything is read, and again with the output buffer filled.
		for range 2 {
			for _, empty := range [][]byte{nil, {}, make([]byte, 0, 8)} {
				if n, err := xof.Read(empty); n != 0 || err != nil {
					t.Fatalf("at %d: an empty Read = %d, %v; want 0, nil", at, n, err)
				}
			}

			one := make([]byte, 1)
			if n, err := xof.Read(one); n != 1 || err != nil || one[0] != ref.Finalize(at, 1)[0] {
				t.Fatalf("at %d: the Read after the empty ones = %d, %v, and is not the stream there", at, n, err)
			}

			if _, err := xof.Seek(-1, io.SeekCurrent); err != nil {
				t.Fatal(err)
			}
		}
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

// TestContractCompressBufferCarry: guts.CompressBuffer compresses "up to
// MaxSIMD chunks starting at counter", and the chunks count on from there
// in 64 bits: a group whose counters cross a multiple of 2^32 carries into
// the high word part-way through. A SIMD kernel adds each lane's index to
// the low word and carries by hand, and a Hasher never takes one there, its
// groups starting at multiples of sixteen. The carry at each place in a
// group, on its last chunk and inside a full one, whole chunks and a
// partial last one, plain and keyed, is the reference's tree of chunks
// numbered one by one. The row is here, not with package guts' own, so that
// it runs wherever this package's tests do: the AVX-512 kernel's carry is
// reached only on a host that dispatches it.
func TestContractCompressBufferCarry(t *testing.T) {
	t.Parallel()

	var buf [guts.MaxSIMD * guts.ChunkSize]byte
	copy(buf[:], cInput(len(buf)))

	key := guts.IV
	key[3] ^= 0x5a5a5a5a

	for back := uint64(1); back < guts.MaxSIMD; back++ {
		for _, counter := range []uint64{1<<32 - back, 5<<32 - back} {
			for _, chunks := range []int{int(back) + 1, guts.MaxSIMD} {
				for _, n := range []int{chunks * guts.ChunkSize, chunks*guts.ChunkSize - 700} {
					for _, flags := range []uint32{0, guts.FlagKeyedHash} {
						got := guts.ChainingValue(guts.CompressBuffer(&buf, n, &key, counter, flags))

						if want := refSubtree(buf[:n], key, counter, flags).ChainingValue(); got != want {
							t.Fatalf("%d bytes from chunk %#x, flags %d: not the reference's tree", n, counter, flags)
						}
					}
				}
			}
		}
	}
}

// refSubtree is the specification's tree over data, whose first chunk is
// number counter: the reference's chunk states, each given its own counter,
// merged with the left subtree the largest power of two of chunks that
// leaves the right one something.
func refSubtree(data []byte, key [8]uint32, counter uint64, flags uint32) refimpl.Output {
	if len(data) <= guts.ChunkSize {
		state := refimpl.ChunkState{Cv: key, Counter: counter, Flags: flags}
		state.Update(data)

		return state.Output()
	}

	left := guts.ChunkSize
	for 2*left < len(data) {
		left *= 2
	}

	return refimpl.ParentOutput(
		refSubtree(data[:left], key, counter, flags).ChainingValue(),
		refSubtree(data[left:], key, counter+uint64(left/guts.ChunkSize), flags).ChainingValue(),
		key,
		flags,
	)
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
		"empty":            {},
		"short":            state[:20],
		"tag":              append([]byte("x"), state[1:]...),
		"flags":            append(append(bytes.Clone(state[:4]), 4, 0, 0, 0), state[8:]...), // FlagParent: no mode
		"trailing byte":    append(bytes.Clone(state), 0),
		"plain with a key": append(append(bytes.Clone(state[:8]), state[8]^1), state[9:]...), // the plain mode's is the IV
		"truncated held":   state[:len(state)-1],
		"held past group":  append(append(bytes.Clone(fresh[:len(fresh)-2]), 0xff, 0xff), make([]byte, 65535)...),
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

// TestContractMarshalCounter: "a state that is not one of those is refused".
// An input of 2^64-1 bytes, the most BLAKE3 hashes, is 2^54 chunks, the last
// of them held back: a counter of 2^54 and above is no input's state, and
// one restored would run the tree past its sixty-four levels. The states
// are a plain Hasher's with the counter written in, a zero subtree for each
// of its bits, and bytes held as a host of either width holds them.
func TestContractMarshalCounter(t *testing.T) {
	t.Parallel()

	fresh, err := blake3.New().MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	crafted := func(counter uint64, held int) []byte {
		b := binary.LittleEndian.AppendUint64(bytes.Clone(fresh[:4+4+32]), counter)
		b = append(b, make([]byte, 32*bits.OnesCount64(counter))...)
		b = binary.LittleEndian.AppendUint16(b, uint16(held))

		return append(b, make([]byte, held)...)
	}

	const limit = 1 << 54

	for _, counter := range []uint64{limit, limit + 1, 1 << 63, math.MaxUint64 - 15, math.MaxUint64} {
		for _, held := range []int{1, kernel.ChunkSize, kernel.ChunkSize + 1, 5000, 16 * kernel.ChunkSize} {
			h := blake3.New()
			h.Write([]byte("abc"))

			before := h.Sum(nil)

			if h.UnmarshalBinary(crafted(counter, held)) == nil {
				t.Errorf("UnmarshalBinary accepted a counter of %#x with %d bytes held", counter, held)
			}

			if got := h.Sum(nil); !bytes.Equal(got, before) {
				t.Fatalf("a refused counter of %#x changed the Hasher", counter)
			}
		}
	}

	// The last counter an input can reach restores, and hashes on.
	var h blake3.Hasher
	if err := h.UnmarshalBinary(crafted(limit-1, 1)); err != nil {
		t.Fatalf("UnmarshalBinary refused a counter of 2^54-1: %v", err)
	}

	h.Write(make([]byte, 3*kernel.ChunkSize))

	if got := h.Sum(nil); len(got) != blake3.Size {
		t.Fatalf("a Hasher restored at the last counter gave %d bytes", len(got))
	}
}

// TestContractMarshalOver: UnmarshalBinary replaces everything its Hasher
// was. Restored over a Hasher of each mode and key, with subtrees on its
// stack and bytes held, a state is the Hasher it was taken from: the same
// state again, the same digest, and the rest of the input hashed on, with
// nothing of what the target held.
func TestContractMarshalOver(t *testing.T) {
	t.Parallel()

	in := cInput(40*kernel.ChunkSize + 300)

	other := make([]byte, 37*kernel.ChunkSize+5)
	for i := range other {
		other[i] = byte(i*13 + 1)
	}

	for _, src := range modes() {
		ref := src.ref()
		ref.Update(in)
		want := ref.Finalize(0, 32)

		for _, at := range []int{0, 1, kernel.ChunkSize + 1, 16*kernel.ChunkSize + 1, 33 * kernel.ChunkSize} {
			h := src.new()
			h.Write(in[:at])

			wantAt := h.Sum(nil)
			state := marshal(t, h)

			for _, dst := range modes() {
				for _, used := range []int{1, 3*kernel.ChunkSize + 700, 16 * kernel.ChunkSize, len(other)} {
					target := dst.new()
					target.Write(other[:used])

					name := src.name + " at " + strconv.Itoa(at) + " over " + dst.name + " after " + strconv.Itoa(used)

					if err := target.UnmarshalBinary(state); err != nil {
						t.Fatalf("%s: UnmarshalBinary: %v", name, err)
					}

					if got := marshal(t, target); !bytes.Equal(got, state) {
						t.Fatalf("%s: the state restored is not the state marshaled", name)
					}

					if got := target.Sum(nil); !bytes.Equal(got, wantAt) {
						t.Fatalf("%s: the restored digest differs", name)
					}

					target.Write(in[at:])

					if got := target.Sum(nil); !bytes.Equal(got, want) {
						t.Fatalf("%s: continued after restore = %x, want %x", name, got[:8], want[:8])
					}
				}
			}
		}
	}
}

// TestContractMarshalSize: MarshalBinary's "at most 18 KiB", at the largest
// state this host makes. A state is 50 bytes, 32 more for each subtree on
// the stack, and the bytes held back. The stack is fullest just under 2^54
// chunks, the longest input's count, which no test writes its way to: the
// state is restored there, from the format, and written on until this host
// holds all it ever does, a chunk with 54 subtrees in pure Go, a group of
// sixteen with 50 under the SIMD kernels. Every state on the way is within
// the bound, and the largest restores.
func TestContractMarshalSize(t *testing.T) {
	t.Parallel()

	const (
		bound = 18 << 10
		empty = 4 + 4 + 32 + 8 + 2 // tag, flags, key, counter, held count
		start = 1<<54 - kernel.MaxSIMD
	)

	fresh := marshal(t, blake3.New())
	if len(fresh) != empty {
		t.Fatalf("the state of a hash of nothing is %d bytes, want %d", len(fresh), empty)
	}

	// A counter a host of either width stands at, one byte held.
	state := binary.LittleEndian.AppendUint64(bytes.Clone(fresh[:empty-8-2]), start)
	state = append(state, make([]byte, 32*bits.OnesCount64(start))...)
	state = append(binary.LittleEndian.AppendUint16(state, 1), 0)

	var h blake3.Hasher
	if err := h.UnmarshalBinary(state); err != nil {
		t.Fatalf("premise: a state at 2^54-16 chunks is refused: %v", err)
	}

	// A group in all with the byte restored, in writes that end by turns
	// inside a chunk and on its boundary; one byte more would complete a
	// chunk past the last counter.
	const half = kernel.ChunkSize / 2

	largest := 0

	for written := 1; written < kernel.MaxSIMD*kernel.ChunkSize; {
		n := half - written%half
		h.Write(make([]byte, n))
		written += n

		if state = marshal(t, &h); len(state) > bound {
			t.Fatalf("%d bytes on: a state of %d bytes; the bound is %d", written, len(state), bound)
		}

		largest = max(largest, len(state))
	}

	// What this host holds at most, and the stack that goes with it.
	want := empty + 32*54 + kernel.ChunkSize
	if kernel.Width() > 1 {
		want = empty + 32*50 + kernel.MaxSIMD*kernel.ChunkSize
	}

	if len(state) != want || largest != want {
		t.Fatalf("the largest state is %d bytes, the last %d; want %d on this host", largest, len(state), want)
	}

	var restored blake3.Hasher
	if err := restored.UnmarshalBinary(state); err != nil {
		t.Fatalf("the largest state does not restore: %v", err)
	}

	if again := marshal(t, &restored); !bytes.Equal(again, state) {
		t.Fatal("the largest state restored is not itself")
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

		// The same state continued in writes that each end on a chunk
		// boundary, the first completing the chunk the state held. A wide
		// host realigning a narrow state goes chunk by chunk, and the chunk
		// a write completes is still the last: it waits for Sum, where it
		// is the tree's right edge, and a state taken there holds it. The
		// boundaries run past the next multiple of sixteen chunks, where
		// the realignment ends.
		aligned := h
		at := tc.done

		for end := at + kernel.ChunkSize - at%kernel.ChunkSize; end <= 33*kernel.ChunkSize; end += kernel.ChunkSize {
			aligned.Write(in[at:end])
			at = end

			if got := aligned.Sum(nil); !bytes.Equal(got, refimpl.Hash(in[:end], 32)) {
				t.Fatalf("%s: digest at a write ending at %d is not the reference's", tc.name, end)
			}

			var again blake3.Hasher
			if err := again.UnmarshalBinary(marshal(t, &aligned)); err != nil {
				t.Fatalf("%s: the state at a write ending at %d: %v", tc.name, end, err)
			}

			again.Write(in[end:])

			if got := again.Sum(nil); !bytes.Equal(got, refimpl.Hash(in, 32)) {
				t.Fatalf("%s: the state at a write ending at %d, continued, is not the reference's", tc.name, end)
			}
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
