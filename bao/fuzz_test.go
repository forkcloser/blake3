package bao_test

import (
	"bytes"
	"io"
	"testing"

	"github.com/forkcloser/blake3"
	"github.com/forkcloser/blake3/bao"
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
	seedData := make([]byte, 5*1024+17)
	blake3.New(0, nil).XOF().Read(seedData)
	for _, group := range []int{0, 2} {
		combined, _ := bao.EncodeBuf(seedData, group, false)
		outboard, _ := bao.EncodeBuf(seedData, group, true)
		f.Add(combined, seedData, uint8(group), uint64(0), uint64(len(seedData)))
		f.Add(outboard, seedData, uint8(group), uint64(1024), uint64(100))
		f.Add(combined[:len(combined)/2], seedData, uint8(group), uint64(0), uint64(1))
	}
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 3}, []byte{}, uint8(0), uint64(0), uint64(1))
	f.Add([]byte{}, []byte{}, uint8(0), uint64(0), uint64(0))

	f.Fuzz(func(t *testing.T, enc, data []byte, group uint8, offset, length uint64) {
		g := int(group % 7)
		root := blake3.Sum256(data)

		bao.Decode(io.Discard, bytes.NewReader(enc), nil, g, root)
		bao.Decode(io.Discard, bytes.NewReader(data), bytes.NewReader(enc), g, root)
		bao.VerifyBuf(enc, nil, g, root)
		bao.VerifyBuf(data, enc, g, root)
		bao.DecodeSlice(io.Discard, bytes.NewReader(enc), g, offset, length, root)
		bao.VerifySlice(enc, g, offset, length, root)
		bao.VerifyChunk(data, enc, g, offset, root)
		bao.ExtractSlice(io.Discard, bytes.NewReader(enc), nil, g, offset, length)
		bao.ExtractSlice(io.Discard, bytes.NewReader(data), bytes.NewReader(enc), g, offset, length)

		// A combined encoding that verifies must decode to exactly the data
		// whose root it was checked against: the decoder's acceptance is the
		// security property, so a false accept here is the bug that matters.
		var out bytes.Buffer
		if ok, err := bao.Decode(&out, bytes.NewReader(enc), nil, g, root); ok && err == nil && !bytes.Equal(out.Bytes(), data) {
			t.Fatalf("group %d: Decode accepted an encoding that is not of the data whose root it was given", g)
		}
	})
}
