package bao_test

// The bounded walk: every data length one step either side of each chunk and
// group boundary, at every group size to 16 KiB, the combined and outboard
// encodings and the root against the reference encoder, EncodedSize exact,
// and both encodings decoded whole; every slice request around every chunk
// boundary, zero lengths and the end included, against the reference slice
// and decoded to the bytes asked for; every chunk group verified alone.

import (
	"bytes"
	"io"
	"strconv"
	"testing"

	"github.com/forkcloser/blake3/bao"
)

// dataLengths are the lengths walked for a group size: each chunk boundary
// to four chunks, each group boundary to four groups, one step either side.
func dataLengths(groupSize int) []int {
	seen := map[int]bool{}

	var lengths []int

	add := func(n int) {
		for _, d := range []int{-1, 0, 1} {
			if n+d >= 0 && !seen[n+d] {
				seen[n+d] = true
				lengths = append(lengths, n+d)
			}
		}
	}

	add(0)

	for c := 1; c <= 4; c++ {
		add(c * 1024)
	}

	for g := 1; g <= 4; g++ {
		add(g * groupSize)
	}

	add(5*groupSize + 1024/2)

	// Five SIMD buffers: a partial group whose merge stack holds a gap (101),
	// so a stale entry is skipped, not merged. Short of a group only from
	// 128 chunks.
	add(5 * 16 * 1024)

	return lengths
}

// TestBoundedEncodings: every length at every group, both encodings and the
// root the reference's, EncodedSize exact, and each encoding decodes whole.
func TestBoundedEncodings(t *testing.T) {
	t.Parallel()

	// Groups to 128 KiB: from 32 chunks a group is more than one SIMD buffer,
	// and the encoder's own merge stack within a group is walked; at 128
	// chunks a partial group of five buffers leaves a gap in that stack.
	data := cData(t, 6*groupSize(7))

	for group := range 8 {
		t.Run(strconv.Itoa(group), func(t *testing.T) {
			t.Parallel()

			for _, n := range dataLengths(groupSize(group)) {
				in := data[:n]
				wantCombined, wantOutboard, wantRoot := refEncode(in, groupSize(group))

				combined, root := bao.EncodeBuf(in, group, false)
				if root != wantRoot || !bytes.Equal(combined, wantCombined) {
					t.Fatalf("len %d: the combined encoding is not the specification's", n)
				}

				outboard, root := bao.EncodeBuf(in, group, true)
				if root != wantRoot || !bytes.Equal(outboard, wantOutboard) {
					t.Fatalf("len %d: the outboard encoding is not the specification's", n)
				}

				if bao.EncodedSize(n, group, false) != len(combined) ||
					bao.EncodedSize(n, group, true) != len(outboard) {
					t.Fatalf("len %d: EncodedSize is not the encoding's size", n)
				}

				var out bytes.Buffer
				if ok, err := bao.Decode(
					&out,
					bytes.NewReader(combined),
					nil,
					group,
					root,
				); !ok || err != nil ||
					!bytes.Equal(out.Bytes(), in) {
					t.Fatalf("len %d: Decode of the combined encoding = %v, %v, %d bytes", n, ok, err, out.Len())
				}

				out.Reset()

				if ok, err := bao.Decode(
					&out,
					bytes.NewReader(in),
					bytes.NewReader(outboard),
					group,
					root,
				); !ok || err != nil ||
					!bytes.Equal(out.Bytes(), in) {
					t.Fatalf("len %d: Decode of the outboard encoding = %v, %v, %d bytes", n, ok, err, out.Len())
				}

				if !bao.VerifyBuf(combined, nil, group, root) || !bao.VerifyBuf(in, outboard, group, root) {
					t.Fatalf("len %d: VerifyBuf refused a valid encoding", n)
				}

				wrong := root
				wrong[31] ^= 1

				if bao.VerifyBuf(combined, nil, group, wrong) || bao.VerifyBuf(in, outboard, group, wrong) {
					t.Fatalf("len %d: VerifyBuf accepted a wrong root", n)
				}
			}
		})
	}
}

// slicePoints are the offsets walked over data of n bytes: each chunk
// boundary one step either side, and the end.
func slicePoints(n int) []uint64 {
	var points []uint64

	for c := 0; c*1024 <= n; c++ {
		for _, d := range []int{-1, 0, 1} {
			if p := c*1024 + d; p >= 0 && p <= n {
				points = append(points, uint64(p))
			}
		}
	}

	return append(points, uint64(n))
}

// TestBoundedSlices: over empty, one-byte, one-chunk-and-a-byte and
// five-chunk data, at groups 0 and 2, every offset around every chunk
// boundary with lengths of zero, one, a chunk and the rest; the slice from
// the combined and from the outboard encoding is the reference's, decodes to
// the bytes asked for, is refused against a wrong root and with a byte
// appended.
func TestBoundedSlices(t *testing.T) {
	t.Parallel()

	all := cData(t, 5*1024+300)

	for _, n := range []int{0, 1, 1025, len(all)} {
		for _, group := range []int{0, 2} {
			t.Run(strconv.Itoa(n)+"/"+strconv.Itoa(group), func(t *testing.T) {
				t.Parallel()

				boundedSlices(t, all[:n], group)
			})
		}
	}
}

// boundedSlices is the walk over one data and group.
func boundedSlices(t *testing.T, data []byte, group int) {
	t.Helper()

	combined, root := bao.EncodeBuf(data, group, false)
	outboard, _ := bao.EncodeBuf(data, group, true)
	wrong := root
	wrong[0] ^= 1

	for _, off := range slicePoints(len(data)) {
		for _, length := range []uint64{0, 1, 1024, uint64(len(data)) - off} {
			if off+length > uint64(len(data)) {
				continue
			}

			want := refSlice(data, groupSize(group), off, length)

			var got bytes.Buffer
			if err := bao.ExtractSlice(&got, bytes.NewReader(combined), nil, group, off, length); err != nil {
				t.Fatalf("(%d, %d): %v", off, length, err)
			}

			if !bytes.Equal(got.Bytes(), want) {
				t.Fatalf(
					"(%d, %d): the slice from the combined encoding is not the specification's",
					off,
					length,
				)
			}

			// outboard: the data reader holds the groups the slice touches
			start, end := groupRange(off, length, len(data), groupSize(group))

			got.Reset()

			if err := bao.ExtractSlice(
				&got,
				bytes.NewReader(data[start:end]),
				bytes.NewReader(outboard),
				group,
				off,
				length,
			); err != nil {
				t.Fatalf("(%d, %d): outboard: %v", off, length, err)
			}

			if !bytes.Equal(got.Bytes(), want) {
				t.Fatalf(
					"(%d, %d): the slice from the outboard encoding is not the specification's",
					off,
					length,
				)
			}

			out, ok := bao.VerifySlice(want, group, off, length, root)
			if !ok || !bytes.Equal(out, data[off:off+length]) {
				t.Fatalf("(%d, %d): VerifySlice = %d bytes, %v", off, length, len(out), ok)
			}

			if _, ok := bao.VerifySlice(want, group, off, length, wrong); ok {
				t.Fatalf("(%d, %d): VerifySlice accepted a wrong root", off, length)
			}

			var decoded bytes.Buffer
			if ok, err := bao.DecodeSlice(
				&decoded,
				bytes.NewReader(want),
				group,
				off,
				length,
				root,
			); !ok || err != nil ||
				!bytes.Equal(decoded.Bytes(), data[off:off+length]) {
				t.Fatalf("(%d, %d): DecodeSlice = %v, %v, %d bytes", off, length, ok, err, decoded.Len())
			}

			if _, ok := bao.VerifySlice(append(bytes.Clone(want), 0), group, off, length, root); ok {
				t.Fatalf("(%d, %d): VerifySlice accepted a trailing byte", off, length)
			}
		}
	}
}

// groupRange is the span of data, in whole groups, a slice request touches:
// what an outboard extraction's data reader must hold. A zero length is the
// chunk at offset, the last one at the end.
func groupRange(off, length uint64, n, groupSize int) (start, end int) {
	if length == 0 && n > 0 {
		if off == uint64(n) {
			off--
		}

		length = 1
	}

	start = int(off) / groupSize * groupSize
	end = min(n, (int(off+length)+groupSize-1)/groupSize*groupSize)

	return start, end
}

// TestBoundedVerifyChunk: every group of the data verified alone against the
// outboard encoding, a corrupted one refused, runs of groups verified, and no
// chunks over non-empty content false.
func TestBoundedVerifyChunk(t *testing.T) {
	t.Parallel()

	// A power of two plus one byte: the split of a subtree is the largest
	// power of two strictly below its length, 4096 and not 2048.
	for _, n := range []int{9*1024 + 17, 4*1024 + 1} {
		boundedVerifyChunk(t, cData(t, n))
	}

	// One byte: the smallest content a zero-length request is refused over.
	outboard, root := bao.EncodeBuf([]byte{0}, 0, true)
	if bao.VerifyChunk(nil, outboard, 0, 0, root) {
		t.Fatal("VerifyChunk with no chunks over one byte returned true")
	}
}

func boundedVerifyChunk(t *testing.T, data []byte) {
	t.Helper()

	for _, group := range []int{0, 1, 2} {
		gs := groupSize(group)
		outboard, root := bao.EncodeBuf(data, group, true)

		for off := 0; off < len(data); off += gs {
			chunk := data[off:min(off+gs, len(data))]

			if !bao.VerifyChunk(chunk, outboard, group, uint64(off), root) {
				t.Fatalf("group %d, offset %d: VerifyChunk refused", group, off)
			}

			bad := bytes.Clone(chunk)
			bad[len(bad)-1] ^= 1

			if bao.VerifyChunk(bad, outboard, group, uint64(off), root) {
				t.Fatalf("group %d, offset %d: VerifyChunk accepted a corrupted group", group, off)
			}

			if bao.VerifyChunk(chunk, outboard, group, uint64(off)+1, root) {
				t.Fatalf("group %d, offset %d: VerifyChunk accepted a misaligned offset", group, off)
			}
		}

		for _, run := range []int{2, 3} {
			if run*gs <= len(data) && !bao.VerifyChunk(data[:run*gs], outboard, group, 0, root) {
				t.Fatalf("group %d: VerifyChunk refused %d contiguous groups", group, run)
			}
		}

		for _, off := range []uint64{0, uint64(gs), uint64(len(data))} {
			if bao.VerifyChunk(nil, outboard, group, off, root) {
				t.Fatalf("group %d, offset %d: VerifyChunk with no chunks returned true", group, off)
			}
		}

		if bao.VerifyChunk(data[:gs], outboard[:len(outboard)-1], group, 0, root) {
			t.Fatalf("group %d: VerifyChunk accepted a short outboard encoding", group)
		}

		huge := []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
		if bao.VerifyChunk(data[:gs], huge, group, 0, root) {
			t.Fatalf("group %d: VerifyChunk accepted a header claiming 2^64-1 bytes", group)
		}
	}
}

// TestBoundedStreaming: Decode through readers that hand out one byte at a
// time and a writer that takes one at a time, so no read or write is ever
// whole: the specification's short-read warning.
func TestBoundedStreaming(t *testing.T) {
	t.Parallel()

	data := cData(t, 3*1024+5)
	combined, root := bao.EncodeBuf(data, 0, false)

	var out bytes.Buffer

	ok, err := bao.Decode(&oneByteWriter{&out}, &oneByteReader{bytes.NewReader(combined)}, nil, 0, root)
	if !ok || err != nil || !bytes.Equal(out.Bytes(), data) {
		t.Fatalf("Decode a byte at a time = %v, %v, %d bytes", ok, err, out.Len())
	}

	var slice bytes.Buffer
	if err := bao.ExtractSlice(
		&oneByteWriter{&slice},
		&oneByteReader{bytes.NewReader(combined)},
		nil,
		0,
		1000,
		1000,
	); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(slice.Bytes(), refSlice(data, 1024, 1000, 1000)) {
		t.Fatal("ExtractSlice a byte at a time is not the specification's slice")
	}
}

type oneByteReader struct{ r io.Reader }

func (o *oneByteReader) Read(p []byte) (int, error) {
	if len(p) > 1 {
		p = p[:1]
	}

	return o.r.Read(p)
}

type oneByteWriter struct{ w io.Writer }

func (o *oneByteWriter) Write(p []byte) (int, error) {
	for i := range p {
		if _, err := o.w.Write(p[i : i+1]); err != nil {
			return i, err
		}
	}

	return len(p), nil
}
