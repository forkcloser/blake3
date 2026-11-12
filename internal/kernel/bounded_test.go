package kernel_test

// The bounded walk: every chunk length to a whole chunk, every chunk count to
// a SIMD group with and without a partial last chunk, every counter and
// count of Eigentrees to 64, the output blocks at counters around the edges,
// each held to the specification's reference (internal/refimpl).

import (
	"bytes"
	"math"
	"testing"

	"github.com/forkcloser/blake3/v2/internal/kernel"
	"github.com/forkcloser/blake3/v2/internal/refimpl"
)

// TestBoundedCompressChunk: a chunk of every length 0..1024 at a few
// counters and flags is the specification's chunk output; the final node
// carries ChunkEnd, and ChunkStart only when it is the chunk's first block.
func TestBoundedCompressChunk(t *testing.T) {
	t.Parallel()

	data := testData(kernel.ChunkSize)

	for n := range kernel.ChunkSize + 1 {
		for _, counter := range []uint64{0, 1, 1 << 32, math.MaxUint64 >> 1} {
			for _, flags := range []uint32{0, kernel.FlagKeyedHash} {
				node := kernel.CompressChunk(data[:n], &testKey, counter, flags)

				if node.Flags&kernel.FlagChunkEnd == 0 ||
					(node.Flags&kernel.FlagChunkStart != 0) != (n <= kernel.BlockSize) {
					t.Fatalf("len %d: flags %#x", n, node.Flags)
				}

				wantLen := n % kernel.BlockSize
				if wantLen == 0 && n > 0 {
					wantLen = kernel.BlockSize // a full last block
				}

				if node.Counter != counter || int(node.BlockLen) != wantLen {
					t.Fatalf("len %d: counter %d, block length %d, want %d", n, node.Counter, node.BlockLen, wantLen)
				}

				state := refimpl.ChunkState{Cv: testKey, Counter: counter, Flags: flags}
				state.Update(data[:n])

				if kernel.ChainingValue(node) != state.Output().ChainingValue() {
					t.Fatalf(
						"len %d, counter %d, flags %d: the chunk's output is not the specification's",
						n,
						counter,
						flags,
					)
				}
			}
		}
	}
}

// TestBoundedCompressBuffer: every chunk count 0..16, with and without a
// partial last chunk, at several counters and flags, is the specification's
// tree over those bytes. The odd counts, 3 to 15, are what nothing above
// the Hasher produces: the Hasher's trees are powers of two.
func TestBoundedCompressBuffer(t *testing.T) {
	t.Parallel()

	var buf [kernel.MaxSIMD * kernel.ChunkSize]byte
	copy(buf[:], testData(len(buf)))

	for chunks := range kernel.MaxSIMD + 1 {
		for _, extra := range []int{-1, 0, 1, kernel.BlockSize, kernel.ChunkSize - 1} {
			n := chunks*kernel.ChunkSize + extra
			if n < 0 || n > len(buf) {
				continue
			}

			for _, counter := range []uint64{0, 7, 1 << 32} {
				for _, flags := range []uint32{0, kernel.FlagKeyedHash} {
					got := kernel.CompressBuffer(&buf, n, &testKey, counter, flags)
					want := refSubtree(buf[:n], testKey, counter, flags)

					if kernel.ChainingValue(got) != want.ChainingValue() {
						t.Fatalf(
							"%d bytes, counter %d, flags %d: CompressBuffer is not the specification's tree",
							n,
							counter,
							flags,
						)
					}
				}
			}
		}
	}
}

// TestBoundedEigentrees: for every counter and count to 64, the heights
// step the counter to counter+chunks, each tree starts aligned to its size,
// and each is the largest aligned power of two that still fits.
func TestBoundedEigentrees(t *testing.T) {
	t.Parallel()

	for counter := range uint64(65) {
		for chunks := range uint64(65) {
			trees := kernel.Eigentrees(counter, chunks)
			at := counter

			for i, height := range trees {
				size := uint64(1) << height
				if at%size != 0 {
					t.Fatalf(
						"Eigentrees(%d, %d)[%d]: height %d at counter %d is not aligned",
						counter,
						chunks,
						i,
						height,
						at,
					)
				}

				if at+size > counter+chunks {
					t.Fatalf(
						"Eigentrees(%d, %d)[%d]: height %d at counter %d runs past the end",
						counter,
						chunks,
						i,
						height,
						at,
					)
				}

				if bigger := size << 1; at%bigger == 0 && at+bigger <= counter+chunks {
					t.Fatalf(
						"Eigentrees(%d, %d)[%d]: height %d at counter %d is not the largest that fits",
						counter,
						chunks,
						i,
						height,
						at,
					)
				}

				at += size
			}

			if at != counter+chunks {
				t.Fatalf("Eigentrees(%d, %d) = %v steps to %d", counter, chunks, trees, at)
			}
		}
	}
}

// TestBoundedCompressBlocks: the MaxSIMD output blocks from a node's
// counter are the specification's stream; CompressBlocksN gives at least
// what was asked, at most MaxSIMD, each block the stream's.
func TestBoundedCompressBlocks(t *testing.T) {
	t.Parallel()

	node := kernel.CompressChunk(testData(100), &kernel.IV, 0, 0)
	node.Flags |= kernel.FlagRoot

	ref := refimpl.ChunkState{Cv: refimpl.IV}
	ref.Update(testData(100))

	stream := ref.Output()

	for _, counter := range []uint64{0, 1, 15, 16, 1 << 32, math.MaxUint64/kernel.BlockSize - kernel.MaxSIMD} {
		node.Counter = counter

		var out [kernel.MaxSIMD * kernel.BlockSize]byte
		kernel.CompressBlocks(&out, node)

		want := stream.RootBytes(counter*kernel.BlockSize, len(out))
		if !bytes.Equal(out[:], want) {
			t.Fatalf("counter %d: CompressBlocks is not the stream", counter)
		}

		for numBlocks := 1; numBlocks <= kernel.MaxSIMD; numBlocks++ {
			var partial [kernel.MaxSIMD * kernel.BlockSize]byte

			got := kernel.CompressBlocksN(&partial, node, numBlocks)
			if got < numBlocks || got > kernel.MaxSIMD {
				t.Fatalf("counter %d: CompressBlocksN(%d) = %d", counter, numBlocks, got)
			}

			if !bytes.Equal(partial[:got*kernel.BlockSize], want[:got*kernel.BlockSize]) {
				t.Fatalf("counter %d: CompressBlocksN(%d)'s %d blocks are not the stream", counter, numBlocks, got)
			}
		}
	}
}
