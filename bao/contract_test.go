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
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/forkcloser/go-blake3/v2"
	"github.com/forkcloser/go-blake3/v2/bao"
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

	// The content the two golden encodings are of: any bytes would do, and
	// these are the ones the reference tool was run on.
	data, err := os.ReadFile("../testdata/bao-golden.in")
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

// endsWith serves its bytes, then ends with err.
type endsWith struct {
	data []byte
	err  error
}

func (e *endsWith) Read(p []byte) (int, error) {
	if len(e.data) == 0 {
		return 0, e.err
	}

	n := copy(p, e.data)
	e.data = e.data[n:]

	return n, nil
}

// TestContractCutShort: "an input that ends before its encoding does, with
// io.EOF or an error wrapping it, is io.ErrUnexpectedEOF", from Decode,
// DecodeSlice, ExtractSlice and Encode, wherever the cut falls, a node's
// boundary included, and Decode has by then written "the content that
// verified before the cut" and no more. To the buffer verifiers an encoding
// "too short to" verify is ErrVerification.
func TestContractCutShort(t *testing.T) {
	t.Parallel()

	data := cData(t, 3*1024+500)
	enc, root := (bao.Encoding{}).EncodeBuf(data)

	var slice bytes.Buffer
	if err := (bao.Encoding{}).ExtractSlice(&slice, bytes.NewReader(enc), nil, 1024, 2000); err != nil {
		t.Fatal(err)
	}

	ends := map[string]error{"io.EOF": io.EOF, "an error wrapping io.EOF": fmt.Errorf("gone: %w", io.EOF)}

	for name, end := range ends {
		for n := range enc {
			var out bytes.Buffer

			err := (bao.Encoding{}).Decode(&out, &endsWith{enc[:n], end}, nil, root)
			if !errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, bao.ErrVerification) {
				t.Fatalf("Decode cut at %d of %d, ending with %s: %v, want io.ErrUnexpectedEOF", n, len(enc), name, err)
			}

			if !bytes.HasPrefix(data, out.Bytes()) || out.Len() == len(data) {
				t.Fatalf("Decode cut at %d: wrote %d bytes that are not a strict start of the content", n, out.Len())
			}

			err = (bao.Encoding{}).ExtractSlice(io.Discard, &endsWith{enc[:n], end}, nil, 0, uint64(len(data)))
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("ExtractSlice cut at %d, ending with %s: %v, want io.ErrUnexpectedEOF", n, name, err)
			}
		}

		for n := range slice.Len() {
			err := (bao.Encoding{}).DecodeSlice(io.Discard, &endsWith{slice.Bytes()[:n], end}, 1024, 2000, root)
			if !errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, bao.ErrVerification) {
				t.Fatalf("DecodeSlice cut at %d, ending with %s: %v, want io.ErrUnexpectedEOF", n, name, err)
			}
		}

		for _, n := range []int{0, 1, 1024, 2048, 2049, len(data) - 1} {
			_, err := (bao.Encoding{}).Encode(discardAt{}, &endsWith{data[:n], end}, int64(len(data)))
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("Encode with %d of %d bytes, ending with %s: %v, want io.ErrUnexpectedEOF",
					n, len(data), name, err)
			}
		}
	}

	for n := range enc {
		if err := (bao.Encoding{}).VerifyBuf(enc[:n], nil, root); !errors.Is(err, bao.ErrVerification) {
			t.Fatalf("VerifyBuf of %d of %d bytes: %v, want ErrVerification", n, len(enc), err)
		}
	}

	for n := range slice.Len() {
		if _, err := (bao.Encoding{}).VerifySlice(
			slice.Bytes()[:n],
			1024,
			2000,
			root,
		); !errors.Is(
			err,
			bao.ErrVerification,
		) {
			t.Fatalf("VerifySlice of %d of %d bytes: %v, want ErrVerification", n, slice.Len(), err)
		}
	}

	obd, obRoot := (bao.Encoding{Outboard: true}).EncodeBuf(data)
	for _, cut := range []struct{ data, tree []byte }{{data, nil}, {data[:len(data)-1], obd}, {data, obd[:len(obd)-1]}} {
		if err := (bao.Encoding{Outboard: true}).VerifyBuf(
			cut.data,
			cut.tree,
			obRoot,
		); !errors.Is(
			err,
			bao.ErrVerification,
		) {
			t.Fatalf(
				"outboard VerifyBuf of %d and %d bytes: %v, want ErrVerification",
				len(cut.data),
				len(cut.tree),
				err,
			)
		}
	}
}

// discardAt is a destination that keeps nothing.
type discardAt struct{}

func (discardAt) WriteAt(p []byte, _ int64) (int, error) { return len(p), nil }

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

// TestContractSliceEveryByte: "a decoder writes nothing it has not
// verified", and to DecodeSlice "content that does not verify against root is
// ErrVerification". Every byte of a slice that spans several groups is
// flipped in turn, parent nodes and chunk groups alike: the slice is refused,
// and what reached the writer is a start of the bytes asked for. The header
// is the one exception, and a bounded one: its length is declared, not
// verified, so a flip there that changes no node the request reads is
// accepted, with exactly the bytes asked for.
func TestContractSliceEveryByte(t *testing.T) {
	t.Parallel()

	const offset, length = 1500, 2600

	data := cData(t, 5*1024+300)
	want := data[offset : offset+length]

	for _, group := range []int{0, 1} {
		format := bao.Encoding{Group: group}
		enc, root := format.EncodeBuf(data)

		var slice bytes.Buffer
		if err := format.ExtractSlice(&slice, bytes.NewReader(enc), nil, offset, length); err != nil {
			t.Fatal(err)
		}

		for i := range slice.Bytes() {
			bad := bytes.Clone(slice.Bytes())
			bad[i] ^= 1

			var out bytes.Buffer

			err := format.DecodeSlice(&out, bytes.NewReader(bad), offset, length, root)

			switch {
			case !bytes.HasPrefix(want, out.Bytes()):
				t.Fatalf("group %d, slice byte %d flipped: %d unverified bytes written", group, i, out.Len())
			case err == nil && out.Len() != len(want):
				t.Fatalf("group %d, slice byte %d flipped: accepted with %d bytes of %d", group, i, out.Len(), length)
			case i >= refHeader && !errors.Is(err, bao.ErrVerification):
				t.Fatalf("group %d, slice byte %d flipped: %v, want ErrVerification", group, i, err)
			}
		}
	}
}

// TestContractVerifyChunkRoot: "VerifyChunk verifies chunks, whole chunk
// groups of content starting at offset, against root with the full outboard
// tree in tree". The tree vouches for the chunks and nothing but the root
// vouches for the tree: each group of a content of several is refused with
// its own tree against another root, and so are another content's group and
// tree, which agree with each other, against this content's root.
func TestContractVerifyChunkRoot(t *testing.T) {
	t.Parallel()

	data := cData(t, 2*(5*1024+300))
	data, other := data[:len(data)/2], data[len(data)/2:]

	for _, group := range []int{0, 1} {
		format := bao.Encoding{Group: group}
		tree, root := (bao.Encoding{Group: group, Outboard: true}).EncodeBuf(data)
		otherTree, _ := (bao.Encoding{Group: group, Outboard: true}).EncodeBuf(other)
		wrong := root
		wrong[0] ^= 1

		for off := 0; off < len(data); off += groupSize(group) {
			end := min(off+groupSize(group), len(data))

			if err := format.VerifyChunk(data[off:end], tree, uint64(off), root); err != nil {
				t.Fatalf("group %d, offset %d: VerifyChunk refused: %v", group, off, err)
			}

			err := format.VerifyChunk(data[off:end], tree, uint64(off), wrong)
			if !errors.Is(err, bao.ErrVerification) {
				t.Fatalf("group %d, offset %d: another root: %v, want ErrVerification", group, off, err)
			}

			err = format.VerifyChunk(other[off:end], otherTree, uint64(off), root)
			if !errors.Is(err, bao.ErrVerification) {
				t.Fatalf("group %d, offset %d: a forged group and tree: %v, want ErrVerification", group, off, err)
			}
		}
	}
}

// countingReader counts the bytes read through it.
type countingReader struct {
	r io.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n

	return n, err
}

// TestContractExtractSliceReads: ExtractSlice "reads the encoding from its
// start to the end of the slice, and no further". A slice from the start of
// a large content is the start of its encoding, the tree's left edge and
// the groups under it: as many bytes are read, from data and tree together,
// as the slice holds, however much content follows. That holds for the
// first byte, and for a range that ends where the next node starts, which
// is not read: a group, a parent after two groups, the tree's whole right
// half after the left.
func TestContractExtractSliceReads(t *testing.T) {
	t.Parallel()

	data := cData(t, 1<<20)

	for _, test := range []struct {
		name   string
		format bao.Encoding
	}{
		{"combined", bao.Encoding{}},
		{"outboard", bao.Encoding{Outboard: true}},
		{"combined, group 2", bao.Encoding{Group: 2}},
		{"outboard, group 2", bao.Encoding{Group: 2, Outboard: true}},
	} {
		enc, root := test.format.EncodeBuf(data)
		leaf := uint64(groupSize(test.format.Group))

		for _, length := range []uint64{1, leaf, 2 * leaf, uint64(len(data)) / 2} {
			// Combined, the encoding is the data reader and the tree reader is
			// ignored; outboard, the content and the tree each have their own.
			content, tree := &countingReader{r: bytes.NewReader(enc)}, &countingReader{r: bytes.NewReader(nil)}

			if test.format.Outboard {
				content.r, tree.r = bytes.NewReader(data), bytes.NewReader(enc)
			}

			var slice bytes.Buffer
			if err := test.format.ExtractSlice(&slice, content, tree, 0, length); err != nil {
				t.Fatalf("%s, %d bytes: %v", test.name, length, err)
			}

			if out, err := test.format.VerifySlice(
				slice.Bytes(),
				0,
				length,
				root,
			); err != nil ||
				!bytes.Equal(out, data[:length]) {
				t.Fatalf("%s, %d bytes: the slice does not verify: %d bytes, %v", test.name, length, len(out), err)
			}

			if read := content.n + tree.n; read != slice.Len() {
				t.Errorf("%s, %d bytes: read %d bytes (%d of content, %d of tree) for a slice of %d",
					test.name, length, read, content.n, tree.n, slice.Len())
			}
		}
	}
}

// TestContractExtractSliceRefused: "A range past the length the encoding
// declares is ErrSliceLength, and nothing is written to dst": not the header
// either, though it is read before the request can be judged.
func TestContractExtractSliceRefused(t *testing.T) {
	t.Parallel()

	data := cData(t, 4096)

	for _, format := range []bao.Encoding{{}, {Outboard: true}} {
		enc, _ := format.EncodeBuf(data)

		for _, test := range []struct{ off, len uint64 }{
			{0, 4097}, {4096, 1}, {4097, 0}, {1 << 63, 1<<63 + 10}, {math.MaxUint64, math.MaxUint64},
		} {
			content := enc

			if format.Outboard {
				content = data
			}

			var dst bytes.Buffer

			err := format.ExtractSlice(&dst, bytes.NewReader(content), bytes.NewReader(enc), test.off, test.len)
			if !errors.Is(err, bao.ErrSliceLength) || dst.Len() != 0 {
				t.Errorf("outboard %t, (%d, %d): %v with %d bytes written, want ErrSliceLength and none",
					format.Outboard, test.off, test.len, err, dst.Len())
			}
		}
	}
}

// TestContractVerifyBufTree: to VerifyBuf "tree is ignored otherwise, as in
// Decode", otherwise being a combined Encoding: its encoding verifies
// whatever tree holds, and the bytes left over that refuse it are data's.
func TestContractVerifyBufTree(t *testing.T) {
	t.Parallel()

	data := cData(t, 3*1024+100)
	enc, root := (bao.Encoding{}).EncodeBuf(data)
	obd, _ := (bao.Encoding{Outboard: true}).EncodeBuf(data)

	for _, tree := range [][]byte{{1, 2, 3}, obd, enc} {
		if err := (bao.Encoding{}).Decode(io.Discard, bytes.NewReader(enc), bytes.NewReader(tree), root); err != nil {
			t.Errorf("Decode of a combined encoding beside %d bytes of tree: %v", len(tree), err)
		}

		if err := (bao.Encoding{}).VerifyBuf(enc, tree, root); err != nil {
			t.Errorf("VerifyBuf of a combined encoding beside %d bytes of tree: %v", len(tree), err)
		}
	}

	// When the encoding is outboard the tree is part of it, and a byte left
	// over after it does not verify.
	outboard := bao.Encoding{Outboard: true}
	if err := outboard.VerifyBuf(data, obd, root); err != nil {
		t.Fatalf("VerifyBuf of an outboard encoding: %v", err)
	}

	if err := outboard.VerifyBuf(data, append(bytes.Clone(obd), 0), root); !errors.Is(err, bao.ErrVerification) {
		t.Errorf("VerifyBuf of an outboard encoding with a byte after its tree: %v, want ErrVerification", err)
	}
}

// miscount is a destination that keeps nothing and answers every write with
// no error and a count of its own, made from the count of bytes it was given.
type miscount func(given int) int

func (m miscount) Write(p []byte) (int, error) { return m(len(p)), nil }

func (m miscount) WriteAt(p []byte, _ int64) (int, error) { return m(len(p)), nil }

// TestContractWriteCounts: "a writer that reports no error but a count other
// than the bytes it was given is an error too: io.ErrShortWrite, wrapped,
// when the count is short", from Decode, DecodeSlice, ExtractSlice and
// Encode. A destination that drops bytes without a word is never taken for
// one that holds the content.
func TestContractWriteCounts(t *testing.T) {
	t.Parallel()

	data := cData(t, 4096)
	enc, root := (bao.Encoding{}).EncodeBuf(data)

	var slice bytes.Buffer
	if err := (bao.Encoding{}).ExtractSlice(&slice, bytes.NewReader(enc), nil, 0, uint64(len(data))); err != nil {
		t.Fatal(err)
	}

	for _, op := range []struct {
		name string
		run  func(dst miscount) error
	}{
		{"Decode", func(dst miscount) error {
			return (bao.Encoding{}).Decode(dst, bytes.NewReader(enc), nil, root)
		}},
		{"DecodeSlice", func(dst miscount) error {
			return (bao.Encoding{}).DecodeSlice(dst, bytes.NewReader(slice.Bytes()), 0, uint64(len(data)), root)
		}},
		{"ExtractSlice", func(dst miscount) error {
			return (bao.Encoding{}).ExtractSlice(dst, bytes.NewReader(enc), nil, 0, uint64(len(data)))
		}},
		{"Encode", func(dst miscount) error {
			_, err := (bao.Encoding{}).Encode(dst, bytes.NewReader(data), int64(len(data)))

			return err
		}},
	} {
		for _, test := range []struct {
			name  string
			count miscount
			short bool
		}{
			{"one byte short", func(given int) int { return given - 1 }, true},
			{"nothing", func(int) int { return 0 }, true},
			{"one byte more", func(given int) int { return given + 1 }, false},
			{"a negative count", func(int) int { return -1 }, false},
		} {
			err := op.run(test.count)
			if err == nil || errors.Is(err, bao.ErrVerification) || test.short != errors.Is(err, io.ErrShortWrite) {
				t.Errorf("%s into a destination that reports %s: %v", op.name, test.name, err)
			}
		}
	}
}

// failingAfter is an io.WriterAt that takes its first writes whole and
// refuses the rest.
type failingAfter struct {
	writes int
	err    error
}

func (f *failingAfter) WriteAt(p []byte, _ int64) (int, error) {
	if f.writes == 0 {
		return 0, f.err
	}

	f.writes--

	return len(p), nil
}

// TestContractEncodeErrorRoot: Encode "returns the tree's root, the BLAKE3
// hash of the data, and the zero root beside an error". Whichever side
// failed and however far the walk had gone, what it hashed is the root of no
// data and is not returned: a caller that drops the error holds no root.
func TestContractEncodeErrorRoot(t *testing.T) {
	t.Parallel()

	data := cData(t, 4096)

	// Four groups: a header and three parent nodes to write, and the four
	// groups among them when the encoding is combined.
	for _, format := range []struct {
		outboard bool
		writes   int
	}{{false, 8}, {true, 4}} {
		lastWrite := &failingAfter{format.writes - 1, errBoom}

		for _, test := range []struct {
			name string
			dst  io.WriterAt
			src  io.Reader
		}{
			{"a source that ends at once", discardAt{}, &endsWith{nil, io.EOF}},
			{"a source that ends between groups", discardAt{}, &endsWith{data[:2048], io.EOF}},
			{"a source that ends inside a group", discardAt{}, &endsWith{data[:2049], io.EOF}},
			{"a source that fails", discardAt{}, failingReader{errBoom}},
			{"a destination that fails after the header", &failingAfter{1, errBoom}, bytes.NewReader(data)},
			{"a destination that fails on its last write", lastWrite, bytes.NewReader(data)},
		} {
			root, err := (bao.Encoding{Outboard: format.outboard}).Encode(test.dst, test.src, int64(len(data)))
			if err == nil || root != ([32]byte{}) {
				t.Errorf("outboard %t, %s: root %x beside %v, want the zero root and an error",
					format.outboard, test.name, root[:4], err)
			}
		}
	}
}

// encodedSize is EncodedSize, and what it panicked with when it did.
func encodedSize(format bao.Encoding, dataLen int64) (size int64, panicked string) {
	defer func() {
		if r := recover(); r != nil {
			panicked = fmt.Sprint(r)
		}
	}()

	return format.EncodedSize(dataLen), ""
}

// TestContractEncodedSizeRange: EncodedSize "panics if dataLen is negative,
// or if the size does not fit an int64: a combined encoding's does not once
// data and tree together pass math.MaxInt64 bytes", and is the encoding's
// size exactly up to there. An outboard tree is 8 + 64*(groups - 1) bytes
// and always fits; a combined encoding adds the data, fits to the length
// whose encoding is math.MaxInt64 bytes to the byte, and panics from the
// next with a message that names the data length.
func TestContractEncodedSizeRange(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		format  bao.Encoding
		dataLen int64
		want    int64 // 0: the size does not fit, and EncodedSize panics
	}{
		{bao.Encoding{Outboard: true}, math.MaxInt64, 8 + 64*(1<<53-1)},
		{bao.Encoding{Group: bao.MaxGroup, Outboard: true}, math.MaxInt64, 8 + 64*(1<<33-1)},
		{bao.Encoding{}, 8680820740569200759, math.MaxInt64},
		{bao.Encoding{}, 8680820740569200760, 0},
		{bao.Encoding{}, math.MaxInt64, 0},
		{bao.Encoding{Group: bao.MaxGroup}, 9223371487098994679, math.MaxInt64},
		{bao.Encoding{Group: bao.MaxGroup}, 9223371487098994680, 0},
		{bao.Encoding{Group: bao.MaxGroup}, math.MaxInt64, 0},
	} {
		size, panicked := encodedSize(test.format, test.dataLen)

		switch {
		case test.want == 0 && !strings.Contains(panicked, "data length"):
			t.Errorf("%+v, %d bytes: size %d, panic %q, want a panic naming the data length",
				test.format, test.dataLen, size, panicked)
		case test.want != 0 && (panicked != "" || size != test.want):
			t.Errorf("%+v, %d bytes: size %d, panic %q, want %d",
				test.format, test.dataLen, size, panicked, test.want)
		}
	}
}
