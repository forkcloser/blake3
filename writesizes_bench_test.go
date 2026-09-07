package blake3_test

import (
	"strconv"
	"testing"

	"github.com/forkcloser/blake3"
)

// BenchmarkWriteSizes drives Hasher.Write with fixed-size slices, the way
// io.Copy (32 KiB) and bufio callers feed it — the sizes that decide
// minParallelWriteBytes and the goroutine dealing in writeTreesParallel.
func BenchmarkWriteSizes(b *testing.B) {
	for _, size := range []int{4 << 10, 8 << 10, 16 << 10, 24 << 10, 32 << 10, 48 << 10, 64 << 10, 128 << 10, 1 << 20} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size))
			buf := make([]byte, size)
			h := blake3.New(32, nil)
			for i := 0; i < b.N; i++ {
				h.Write(buf)
			}
		})
	}
}
