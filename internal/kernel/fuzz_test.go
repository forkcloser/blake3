package kernel_test

import (
	"testing"

	"github.com/forkcloser/blake3/internal/kernel"
)

// FuzzCompressBuffer: whatever the bytes and their count up to a SIMD group,
// at whatever counter and flags, CompressBuffer is the specification's tree
// over them. The deterministic walk pins every chunk count; this hands the
// engine the content and the odd lengths between.
func FuzzCompressBuffer(f *testing.F) {
	f.Add([]byte("seed"), uint64(0), uint32(0))
	f.Add(make([]byte, 5*kernel.ChunkSize+3), uint64(1<<32), uint32(kernel.FlagKeyedHash))

	f.Fuzz(func(t *testing.T, data []byte, counter uint64, flags uint32) {
		if len(data) > kernel.MaxSIMD*kernel.ChunkSize {
			data = data[:kernel.MaxSIMD*kernel.ChunkSize]
		}

		var buf [kernel.MaxSIMD * kernel.ChunkSize]byte
		copy(buf[:], data)

		flags &= kernel.FlagKeyedHash | kernel.FlagDeriveKeyContext | kernel.FlagDeriveKeyMaterial

		got := kernel.CompressBuffer(&buf, len(data), &testKey, counter, flags)
		want := refSubtree(data, testKey, counter, flags)

		if kernel.ChainingValue(got) != want.ChainingValue() {
			t.Fatalf(
				"%d bytes, counter %d, flags %d: CompressBuffer is not the specification's tree",
				len(data),
				counter,
				flags,
			)
		}
	})
}
