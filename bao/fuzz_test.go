package bao_test

import (
	"bytes"
	"io"
	"testing"

	"github.com/forkcloser/blake3/v2"
	"github.com/forkcloser/blake3/v2/bao"
)

// FuzzDecoders drives every function that parses an encoding — a length
// prefix and a tree of chaining values, both attacker-controlled — with
// arbitrary bytes, arbitrary slice bounds and every group size up to 64 KiB.
// The property is termination without a runtime panic: a decoder must reject
// what it cannot verify by returning false or an error, never by crashing,
// and must not recurse or allocate beyond what its arguments bound. The
// group range stops at 6 so a single iteration never allocates more than a
// 64 KiB group buffer. Seeds are valid encodings, combined and outboard, at
// two group sizes, so the engine starts from inputs that reach the leaves and
// mutates from there.
func FuzzDecoders(f *testing.F) {
	seedData := cData(f, 5*1024+17)
	for _, group := range []int{0, 2} {
		combined, _ := (bao.Encoding{Group: group}).EncodeBuf(seedData)
		outboard, _ := (bao.Encoding{Group: group, Outboard: true}).EncodeBuf(seedData)
		f.Add(combined, seedData, uint8(group), uint64(0), uint64(len(seedData)))
		f.Add(outboard, seedData, uint8(group), uint64(1024), uint64(100))
		f.Add(combined[:len(combined)/2], seedData, uint8(group), uint64(0), uint64(1))
	}

	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 3}, []byte{}, uint8(0), uint64(0), uint64(1))
	f.Add([]byte{}, []byte{}, uint8(0), uint64(0), uint64(0))

	f.Fuzz(func(t *testing.T, enc, data []byte, group uint8, offset, length uint64) {
		g := int(group % 7)
		root := blake3.Sum256(data)

		combined, outboard := bao.Encoding{Group: g}, bao.Encoding{Group: g, Outboard: true}

		_ = combined.Decode(io.Discard, bytes.NewReader(enc), nil, root)
		_ = outboard.Decode(io.Discard, bytes.NewReader(data), bytes.NewReader(enc), root)
		_ = combined.VerifyBuf(enc, nil, root)
		_ = outboard.VerifyBuf(data, enc, root)
		_ = combined.DecodeSlice(io.Discard, bytes.NewReader(enc), offset, length, root)
		_, _ = combined.VerifySlice(enc, offset, length, root)
		_ = combined.VerifyChunk(data, enc, offset, root)
		_ = combined.ExtractSlice(io.Discard, bytes.NewReader(enc), nil, offset, length)
		_ = outboard.ExtractSlice(io.Discard, bytes.NewReader(data), bytes.NewReader(enc), offset, length)

		// A combined encoding that verifies must decode to exactly the data
		// whose root it was checked against: the decoder's acceptance is the
		// security property, so a false accept here is the bug that matters.
		var out bytes.Buffer
		if err := combined.Decode(&out, bytes.NewReader(enc), nil, root); err == nil &&
			!bytes.Equal(out.Bytes(), data) {
			t.Fatalf("group %d: Decode accepted an encoding that is not of the data whose root it was given", g)
		}
	})
}

// FuzzSlices: whatever the data, group and request within the content, the
// slice is the specification's and decodes to the bytes asked for.
func FuzzSlices(f *testing.F) {
	f.Add([]byte("seed data for a slice"), uint8(0), uint64(3), uint64(5))
	f.Add(make([]byte, 5*1024+3), uint8(1), uint64(1024), uint64(0))

	f.Fuzz(func(t *testing.T, data []byte, group uint8, offset, length uint64) {
		g := int(group % 5)

		if offset > uint64(len(data)) {
			offset %= uint64(len(data)) + 1
		}

		if length > uint64(len(data))-offset {
			length %= uint64(len(data)) - offset + 1
		}

		combined, root := (bao.Encoding{Group: g}).EncodeBuf(data)
		want := refSlice(data, 1024<<g, offset, length)

		var got bytes.Buffer
		if err := (bao.Encoding{Group: g}).ExtractSlice(
			&got,
			bytes.NewReader(combined),
			nil,
			offset,
			length,
		); err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(got.Bytes(), want) {
			t.Fatalf("group %d (%d, %d): the slice is not the specification's", g, offset, length)
		}

		out, err := (bao.Encoding{Group: g}).VerifySlice(want, offset, length, root)
		if err != nil || !bytes.Equal(out, data[offset:offset+length]) {
			t.Fatalf("group %d (%d, %d): VerifySlice = %d bytes, %v", g, offset, length, len(out), err)
		}
	})
}
