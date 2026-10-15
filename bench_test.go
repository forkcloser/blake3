package blake3_test

import (
	"io"
	"strconv"
	"testing"

	"github.com/forkcloser/blake3"
)

// The guards: the package exists for speed on large inputs and outputs, and
// a change that keeps every test green and halves the throughput shows here.
// The benchmarks keep upstream's loop shape (a plain b.N loop, not b.Loop)
// so that their numbers compare one to one with lukechampine/blake3 v1.4.1 on
// the same host: b.Loop costs ~2 ns per iteration, which read as a 4% "loss"
// on the 73 ns Sum256/64 case until the harnesses were made identical.

type nopReader struct{}

func (nopReader) Read(p []byte) (int, error) { return len(p), nil }

func BenchmarkWrite(b *testing.B) {
	b.ReportAllocs()
	b.SetBytes(1024)

	if _, err := io.CopyN(blake3.New(0, nil), nopReader{}, int64(b.N*1024)); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkWriteSizes drives Hasher.Write with fixed-size slices, the way
// io.Copy (32 KiB) and bufio callers feed it: the sizes that decide
// minParallelWriteBytes and the goroutine dealing in writeTreesParallel.
func BenchmarkWriteSizes(b *testing.B) {
	for _, size := range []int{4 << 10, 8 << 10, 16 << 10, 24 << 10, 32 << 10, 48 << 10, 64 << 10, 128 << 10, 1 << 20} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size))
			buf := make([]byte, size)

			h := blake3.New(32, nil)
			for range b.N {
				h.Write(buf)
			}
		})
	}
}

func BenchmarkXOF(b *testing.B) {
	for _, size := range []int64{64, 1024, 65536, 1048576} {
		b.Run(strconv.FormatInt(size, 10), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(size)
			buf := make([]byte, size)

			xof := blake3.New(0, nil).XOF()
			for range b.N {
				_, _ = xof.Seek(0, 0)
				_, _ = xof.Read(buf)
			}
		})
	}
}

func BenchmarkSum256(b *testing.B) {
	for _, size := range []int64{64, 1024, 65536, 1048576} {
		b.Run(strconv.FormatInt(size, 10), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(size)

			buf := make([]byte, size)
			for range b.N {
				blake3.Sum256(buf)
			}
		})
	}
}
