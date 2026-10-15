package bao_test

// The Bao specification (oconnor663/bao, docs/spec.md) as a reference encoder,
// written from the text: the combined encoding is the 8-byte little-endian
// length, then the tree's parent nodes (left child's chaining value, right
// child's) and chunk groups in pre-order; the outboard encoding leaves the
// groups out; a slice keeps the nodes and groups a decoder would meet seeking
// to the start point and reading the count, a zero count standing for the
// chunk at the start point and a start point at the end for the final one.
// The tree is BLAKE3's over chunk groups: the left subtree holds the largest
// power of two of bytes strictly below the length. The chaining values come
// from internal/refimpl, the specification's reference implementation.

import (
	"encoding/binary"
	"math/bits"

	"github.com/forkcloser/blake3/internal/refimpl"
)

const (
	refChunkLen = refimpl.ChunkLen
	refCVLen    = 32
	refHeader   = 8
)

// refTree is the specification's node over data at byte position pos: a
// chunk group is the tree over its chunks (refimpl's chunk states, merged
// left-heavy within the group), a larger span a parent over its halves.
func refTree(data []byte, pos uint64, groupSize int) refimpl.Output {
	if len(data) <= groupSize {
		return refGroup(data, pos/refChunkLen)
	}

	mid := refMid(len(data))

	left := refTree(data[:mid], pos, groupSize)
	right := refTree(data[mid:], pos+uint64(mid), groupSize)

	return refimpl.ParentOutput(left.ChainingValue(), right.ChainingValue(), refimpl.IV, 0)
}

// refGroup is the tree over one chunk group's bytes, its chunks counted from
// counter.
func refGroup(data []byte, counter uint64) refimpl.Output {
	if len(data) <= refChunkLen {
		state := refimpl.ChunkState{Cv: refimpl.IV, Counter: counter}
		state.Update(data)

		return state.Output()
	}

	mid := refMid(len(data))

	left := refGroup(data[:mid], counter)
	right := refGroup(data[mid:], counter+uint64(mid/refChunkLen))

	return refimpl.ParentOutput(left.ChainingValue(), right.ChainingValue(), refimpl.IV, 0)
}

// refMid is the largest power of two strictly below n: the left subtree's
// size in bytes.
func refMid(n int) int {
	return 1 << (bits.Len(uint(n-1)) - 1)
}

// refRoot is the BLAKE3 hash of data: the tree's root output.
func refRoot(data []byte, groupSize int) (root [32]byte) {
	copy(root[:], refTree(data, 0, groupSize).RootBytes(0, refCVLen))

	return root
}

func refCVBytes(cv [8]uint32) []byte {
	b := make([]byte, 0, refCVLen)
	for _, w := range cv {
		b = binary.LittleEndian.AppendUint32(b, w)
	}

	return b
}

// refEncode is the specification's combined and outboard encodings of data
// with groups of groupSize bytes, and the root.
func refEncode(data []byte, groupSize int) (combined, outboard []byte, root [32]byte) {
	header := binary.LittleEndian.AppendUint64(nil, uint64(len(data)))
	combined, outboard = append(combined, header...), append(outboard, header...)

	var walk func(d []byte, pos uint64)

	walk = func(d []byte, pos uint64) {
		if len(d) <= groupSize {
			combined = append(combined, d...)

			return
		}

		mid := refMid(len(d))
		left := refTree(d[:mid], pos, groupSize).ChainingValue()
		right := refTree(d[mid:], pos+uint64(mid), groupSize).ChainingValue()

		parent := append(refCVBytes(left), refCVBytes(right)...)
		combined, outboard = append(combined, parent...), append(outboard, parent...)

		walk(d[:mid], pos)
		walk(d[mid:], pos+uint64(mid))
	}

	walk(data, 0)

	return combined, outboard, refRoot(data, groupSize)
}

// refSlice is the specification's slice of data's encoding for the request:
// the header, then every parent node and chunk group a decoder meets reading
// the range, in pre-order. A zero length stands for one byte at offset, or at
// the last byte when offset is the end; the empty encoding is its own one
// empty chunk.
func refSlice(data []byte, groupSize int, offset, length uint64) []byte {
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(data)))

	if length == 0 && len(data) > 0 {
		if offset == uint64(len(data)) {
			offset--
		}

		length = 1
	}

	inRange := func(pos uint64, n int) bool {
		return pos < offset+length && offset < pos+uint64(n)
	}

	var walk func(d []byte, pos uint64)

	walk = func(d []byte, pos uint64) {
		if len(d) <= groupSize {
			if len(d) == 0 || inRange(pos, len(d)) {
				out = append(out, d...)
			}

			return
		}

		if !inRange(pos, len(d)) {
			return
		}

		mid := refMid(len(d))
		left := refTree(d[:mid], pos, groupSize).ChainingValue()
		right := refTree(d[mid:], pos+uint64(mid), groupSize).ChainingValue()

		out = append(out, refCVBytes(left)...)
		out = append(out, refCVBytes(right)...)

		walk(d[:mid], pos)
		walk(d[mid:], pos+uint64(mid))
	}

	walk(data, 0)

	return out
}
