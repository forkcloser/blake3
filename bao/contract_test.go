package bao_test

// The contract of package bao, from its sources, each rule cited:
//
//   - The Bao specification fixes the combined and outboard encodings, the
//     slice format and what a decoder must do: verify every chaining value it
//     reads, verify the final chunk (the empty chunk of an empty encoding
//     included) before reporting a length, write no byte it has not verified.
//     The reference encoder in reference_test.go is the specification as
//     written; the reference implementation's golden files and the abao
//     project's chunk-group vectors (testdata) pin it independently.
//   - The package doc: every function validates group, [0, MaxGroup], and
//     panics on any other; a negative length panics in Encode and EncodedSize.
//   - Encode's doc: dst is written by offset, and the result's root is the
//     BLAKE3 hash of the data. EncodedSize's doc: the encoding's size exactly.
//   - Decode's doc: false on a verification failure, with the error the
//     readers' or the writer's own when one failed.
//   - ExtractSlice's and DecodeSlice's docs: ErrSliceLength for a request that
//     does not fit the encoding; the spec's zero-length rule.
//   - VerifyChunk's doc: false with no chunks over non-empty content.

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/bao"
)

// cData is n bytes of BLAKE3 output, the data these tests encode.
func cData(tb testing.TB, n int) []byte {
	tb.Helper()

	data := make([]byte, n)
	if _, err := blake3.New().XOF().Read(data); err != nil {
		tb.Fatal(err)
	}

	return data
}

func groupSize(group int) int { return 1024 << group }

// TestContractReferenceEncoder: the reference encoder written here agrees
// with the reference implementation's golden files and the abao vectors
// before anything leans on it, and the package agrees with both.
func TestContractReferenceEncoder(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../testdata/vectors.json")
	if err != nil {
		t.Fatal(err)
	}

	goldenCombined, err := os.ReadFile("../testdata/bao-golden.bao")
	if err != nil {
		t.Fatal(err)
	}

	goldenOutboard, err := os.ReadFile("../testdata/bao-golden.obao")
	if err != nil {
		t.Fatal(err)
	}

	const goldenRoot = "6654fbd1836b531b25e2782c9cc9b792c80abb36b024f59db5d5f6bd3187ddfe"

	combined, outboard, root := refEncode(data, groupSize(0))
	if hex.EncodeToString(root[:]) != goldenRoot || !bytes.Equal(combined, goldenCombined) ||
		!bytes.Equal(outboard, goldenOutboard) {
		t.Fatal("the reference encoder disagrees with the golden files")
	}

	if got, gotRoot := (bao.Encoding{}).EncodeBuf(data); gotRoot != root || !bytes.Equal(got, goldenCombined) {
		t.Fatal("EncodeBuf disagrees with the golden combined encoding")
	}

	if got, gotRoot := (bao.Encoding{Outboard: true}).EncodeBuf(
		data,
	); gotRoot != root ||
		!bytes.Equal(got, goldenOutboard) {
		t.Fatal("EncodeBuf disagrees with the golden outboard encoding")
	}

	// abao's chunk-group vectors: group 4, the input a run of little-endian
	// counters.
	input := func(n int) (in []byte) {
		for i := uint32(1); len(in) < n; i++ {
			in = binary.LittleEndian.AppendUint32(in, i)
		}

		return in[:n]
	}

	for _, test := range []struct {
		inputLen int
		root     string
	}{
		{0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
		{1, "48fc721fbbc172e0925fa27af1671de225ba927134802998b10a1568a188652b"},
		{1023, "15f8c1ae1049fe7e837186612c8ce732e66835841a4569b71e4ac3e3d3411b90"},
		{1024, "f749c19181983b839cd97fe121cebaf076bc951e8c8e6d64accfedad5951ec22"},
		{1025, "3613596275c4ea790774dedf20835b2daf86cacc892feef6ce720c121572f1f9"},
		{16383, "f0970fbfe2f1c5145fa6aa31833779803d5c53743a8443ed1395218f511834ba"},
		{16384, "b318758645c4467406c829a5f3da7cab00010fccccf4b7c314525cd85e2d0af8"},
		{16385, "12a6a6b0554e7f3eed485f668bfd3b37382a2beee5e7ed5594c4a91c4c70f4aa"},
		{32768, "8008de557073cab60f851191359ad9dc1afe9dc6152668ee01825c56ac5a754e"},
		{49152, "91823357fefc308b57bb85ebed1d1edeba3c355e804dc63fa98fcb82554b1566"},
		{180224, "4742cbae9485ce1b86ab359c1a84e203f819795d018b22a5c70c5c4577dd732e"},
		{212992, "760c549edfe95c734b1d6a9b846d81692ed3ca022b541442949a0e42fe570df2"},
	} {
		in := input(test.inputLen)

		if root := refRoot(in, groupSize(4)); hex.EncodeToString(root[:]) != test.root {
			t.Fatalf("len %d: the reference root is not abao's", test.inputLen)
		}

		if _, root := (bao.Encoding{Group: 4}).EncodeBuf(in); hex.EncodeToString(root[:]) != test.root {
			t.Fatalf("len %d: EncodeBuf's root is not abao's", test.inputLen)
		}
	}
}

// TestContractRootIsHash: an encoding's root is the BLAKE3 hash of the data,
// whatever the group.
func TestContractRootIsHash(t *testing.T) {
	t.Parallel()

	data := cData(t, 150000) // more than two of the largest groups below

	for group := range 7 {
		_, root := (bao.Encoding{Group: group, Outboard: true}).EncodeBuf(data)
		if root != blake3.Sum256(data) {
			t.Fatalf("group %d: the root is not Sum256", group)
		}
	}
}

// failingAt is an io.WriterAt that refuses every write.
type failingAt struct{ err error }

func (f failingAt) WriteAt([]byte, int64) (int, error) { return 0, f.err }

// failingReader fails on its first Read.
type failingReader struct{ err error }

func (f failingReader) Read([]byte) (int, error) { return 0, f.err }

// failingWriter refuses every write.
type failingWriter struct{ err error }

func (f failingWriter) Write([]byte) (int, error) { return 0, f.err }

var errBoom = errors.New("boom")

// TestContractErrors: a destination or a source that fails is its own error,
// wrapped, from Encode, Decode, ExtractSlice and DecodeSlice; a slice request
// past the content is ErrSliceLength; content that does not verify is
// ErrVerification.
func TestContractErrors(t *testing.T) {
	t.Parallel()

	data := cData(t, 4096)
	enc, root := (bao.Encoding{}).EncodeBuf(data)

	if _, err := (bao.Encoding{}).Encode(failingAt{errBoom}, bytes.NewReader(data), int64(len(data))); !errors.Is(
		err,
		errBoom,
	) {
		t.Errorf("Encode with a failing destination: %v", err)
	}

	if _, err := (bao.Encoding{}).Encode(&cDiscardAt{}, failingReader{errBoom}, int64(len(data))); !errors.Is(
		err,
		errBoom,
	) {
		t.Errorf("Encode with a failing source: %v", err)
	}

	if err := (bao.Encoding{}).Decode(io.Discard, failingReader{errBoom}, nil, root); !errors.Is(err, errBoom) {
		t.Errorf("Decode with a failing source: %v", err)
	}

	if err := (bao.Encoding{}).Decode(
		failingWriter{errBoom},
		bytes.NewReader(enc),
		nil,
		root,
	); !errors.Is(
		err,
		errBoom,
	) {
		t.Errorf("Decode with a failing destination: %v", err)
	}

	if err := (bao.Encoding{}).Decode(
		io.Discard,
		bytes.NewReader(enc),
		nil,
		[32]byte{},
	); !errors.Is(
		err,
		bao.ErrVerification,
	) {
		t.Errorf("Decode with a wrong root: %v, want ErrVerification", err)
	}

	if err := (bao.Encoding{}).ExtractSlice(
		failingWriter{errBoom},
		bytes.NewReader(enc),
		nil,
		0,
		10,
	); !errors.Is(
		err,
		errBoom,
	) {
		t.Errorf("ExtractSlice with a failing destination: %v", err)
	}

	// The whole content as a slice: four groups, so the walk goes on past
	// the failed write and stops on the error it recorded.
	var slice bytes.Buffer
	if err := (bao.Encoding{}).ExtractSlice(&slice, bytes.NewReader(enc), nil, 0, uint64(len(data))); err != nil {
		t.Fatal(err)
	}

	if err := (bao.Encoding{}).DecodeSlice(
		failingWriter{errBoom},
		&slice,
		0,
		uint64(len(data)),
		root,
	); !errors.Is(
		err,
		errBoom,
	) {
		t.Errorf("DecodeSlice with a failing destination: %v", err)
	}

	if err := (bao.Encoding{}).ExtractSlice(io.Discard, failingReader{errBoom}, nil, 0, 10); !errors.Is(err, errBoom) {
		t.Errorf("ExtractSlice with a failing source: %v", err)
	}

	for _, test := range []struct{ off, len uint64 }{
		{0, 4097}, {4096, 1}, {1 << 63, 1<<63 + 10}, {math.MaxUint64, math.MaxUint64},
	} {
		err := (bao.Encoding{}).ExtractSlice(io.Discard, bytes.NewReader(enc), nil, test.off, test.len)
		if !errors.Is(err, bao.ErrSliceLength) {
			t.Errorf("ExtractSlice (%d, %d): %v, want ErrSliceLength", test.off, test.len, err)
		}

		if err := (bao.Encoding{}).DecodeSlice(
			io.Discard,
			bytes.NewReader(enc),
			test.off,
			test.len,
			root,
		); !errors.Is(
			err,
			bao.ErrSliceLength,
		) {
			t.Errorf("DecodeSlice (%d, %d): %v, want ErrSliceLength", test.off, test.len, err)
		}

		if _, err := (bao.Encoding{}).VerifySlice(enc, test.off, test.len, root); !errors.Is(err, bao.ErrSliceLength) {
			t.Errorf("VerifySlice (%d, %d): %v, want ErrSliceLength", test.off, test.len, err)
		}
	}

	for _, short := range [][]byte{nil, {1, 2, 3}} {
		want := io.ErrUnexpectedEOF
		if len(short) == 0 {
			want = io.EOF
		}

		if err := (bao.Encoding{}).ExtractSlice(io.Discard, bytes.NewReader(short), nil, 0, 1); !errors.Is(err, want) {
			t.Errorf("ExtractSlice on %d header bytes: %v, want %v", len(short), err, want)
		}

		if err := (bao.Encoding{}).DecodeSlice(io.Discard, bytes.NewReader(short), 0, 1, root); !errors.Is(err, want) {
			t.Errorf("DecodeSlice on %d header bytes: %v, want %v", len(short), err, want)
		}
	}
}

// cDiscardAt accepts every write.
type cDiscardAt struct{}

func (*cDiscardAt) WriteAt(p []byte, _ int64) (int, error) { return len(p), nil }

// TestContractArguments: an out-of-range group panics with a message that
// names it at every entry point, the bounds themselves are taken, and a
// negative length panics in Encode before anything is written and in
// EncodedSize.
func TestContractArguments(t *testing.T) {
	t.Parallel()

	data := cData(t, 3000)
	enc, root := (bao.Encoding{}).EncodeBuf(data)
	obd, _ := (bao.Encoding{Outboard: true}).EncodeBuf(data)

	mustPanic := func(name string, f func()) {
		t.Helper()

		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("%s: no panic", name)

				return
			}

			if msg, ok := r.(string); !ok || !strings.Contains(msg, "Group") && !strings.Contains(msg, "negative") {
				t.Errorf("%s: panicked with %v, want the named message", name, r)
			}
		}()

		f()
	}

	for _, group := range []int{-1, bao.MaxGroup + 1, math.MinInt, math.MaxInt} {
		mustPanic("EncodedSize", func() { (bao.Encoding{Group: group}).EncodedSize(10) })
		mustPanic(
			"Encode",
			func() {
				_, _ = (bao.Encoding{Group: group}).Encode(&cDiscardAt{}, bytes.NewReader(data), int64(len(data)))
			},
		)
		mustPanic(
			"Decode",
			func() { _ = (bao.Encoding{Group: group}).Decode(io.Discard, bytes.NewReader(enc), nil, root) },
		)
		mustPanic("EncodeBuf", func() { (bao.Encoding{Group: group}).EncodeBuf(data) })
		mustPanic("VerifyBuf", func() { _ = (bao.Encoding{Group: group}).VerifyBuf(enc, nil, root) })
		mustPanic(
			"ExtractSlice",
			func() { _ = (bao.Encoding{Group: group}).ExtractSlice(io.Discard, bytes.NewReader(enc), nil, 0, 1) },
		)
		mustPanic(
			"DecodeSlice",
			func() { _ = (bao.Encoding{Group: group}).DecodeSlice(io.Discard, bytes.NewReader(enc), 0, 1, root) },
		)
		mustPanic("VerifySlice", func() { _, _ = (bao.Encoding{Group: group}).VerifySlice(enc, 0, 1, root) })
		mustPanic("VerifyChunk", func() { _ = (bao.Encoding{Group: group}).VerifyChunk(data, obd, 0, root) })
	}

	if got := (bao.Encoding{Group: bao.MaxGroup, Outboard: true}).EncodedSize(10); got != 8 {
		t.Errorf("EncodedSize at MaxGroup = %d, want 8", got)
	}

	var written cCountingAt

	mustPanic("Encode negative", func() { _, _ = (bao.Encoding{}).Encode(&written, bytes.NewReader(nil), -1) })
	mustPanic("EncodedSize negative", func() { (bao.Encoding{}).EncodedSize(-1) })

	if written != 0 {
		t.Errorf("Encode wrote %d bytes before refusing a negative length", written)
	}
}

// cCountingAt counts the bytes written to it.
type cCountingAt int

func (c *cCountingAt) WriteAt(p []byte, _ int64) (int, error) {
	*c += cCountingAt(len(p))

	return len(p), nil
}

// TestContractDecoderWritesVerifiedOnly: a decoder writes nothing it has not
// verified. A corrupt byte anywhere in a combined encoding, or in the
// outboard tree, is refused, and what reached the writer is a prefix of the
// data that was verified before the corruption. Every byte position of a
// three-chunk encoding is tried.
func TestContractDecoderWritesVerifiedOnly(t *testing.T) {
	t.Parallel()

	data := cData(t, 3*1024+100)
	enc, root := (bao.Encoding{}).EncodeBuf(data)

	for i := range enc {
		bad := bytes.Clone(enc)
		bad[i] ^= 0x01

		var out bytes.Buffer

		err := (bao.Encoding{}).Decode(&out, bytes.NewReader(bad), nil, root)
		if err == nil {
			t.Fatalf("byte %d flipped: Decode accepted", i)
		}

		if !errors.Is(err, bao.ErrVerification) && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
			t.Fatalf("byte %d flipped: %v", i, err)
		}

		if !bytes.HasPrefix(data, out.Bytes()) {
			t.Fatalf("byte %d flipped: %d unverified bytes written", i, out.Len())
		}

		if (bao.Encoding{}).VerifyBuf(bad, nil, root) == nil {
			t.Fatalf("byte %d flipped: VerifyBuf accepted", i)
		}
	}

	// truncated at every length
	for n := range enc {
		var out bytes.Buffer
		if err := (bao.Encoding{}).Decode(&out, bytes.NewReader(enc[:n]), nil, root); err == nil {
			t.Fatalf("truncated to %d: Decode accepted", n)
		} else if !bytes.HasPrefix(data, out.Bytes()) {
			t.Fatalf("truncated to %d: unverified bytes written", n)
		}

		if (bao.Encoding{}).VerifyBuf(enc[:n], nil, root) == nil {
			t.Fatalf("truncated to %d: VerifyBuf accepted", n)
		}
	}

	// trailing bytes
	if err := (bao.Encoding{}).VerifyBuf(append(bytes.Clone(enc), 0), nil, root); !errors.Is(err, bao.ErrVerification) {
		t.Fatalf("VerifyBuf with trailing data = %v, want ErrVerification", err)
	}

	// the empty encoding verifies its one empty chunk: a wrong root is refused
	empty, emptyRoot := (bao.Encoding{}).EncodeBuf(nil)
	if err := (bao.Encoding{}).VerifyBuf(empty, nil, emptyRoot); err != nil {
		t.Fatalf("the empty encoding is refused: %v", err)
	}

	if err := (bao.Encoding{}).VerifyBuf(empty, nil, [32]byte{}); !errors.Is(err, bao.ErrVerification) {
		t.Fatalf("the empty encoding is not held to its root: %v", err)
	}

	if err := (bao.Encoding{}).VerifyChunk(nil, empty, 0, emptyRoot); err != nil {
		t.Fatalf("VerifyChunk refuses the empty encoding: %v", err)
	}

	if err := (bao.Encoding{}).VerifyChunk(nil, empty, 0, [32]byte{}); !errors.Is(err, bao.ErrVerification) {
		t.Fatalf("VerifyChunk does not hold the empty encoding to its root: %v", err)
	}
}
