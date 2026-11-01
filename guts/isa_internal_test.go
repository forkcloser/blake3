//go:build amd64

package guts

import (
	"os"
	"testing"
)

// TestRequiredISA fails when BLAKE3_REQUIRE_ISA names kernels this host does
// not dispatch. The emulated AVX-512 lane sets it to avx512: the other tests
// then prove those kernels, not a fallback the host happened to select,
// produced the specification's results. Unset, the test is skipped.
func TestRequiredISA(t *testing.T) {
	t.Parallel()

	switch isa := os.Getenv("BLAKE3_REQUIRE_ISA"); isa {
	case "":
		t.Skip("BLAKE3_REQUIRE_ISA is not set")
	case "avx512":
		if !haveAVX512 {
			t.Fatal("BLAKE3_REQUIRE_ISA=avx512: this host does not dispatch the AVX-512 kernels")
		}
	case "avx2":
		if !haveAVX2 {
			t.Fatal("BLAKE3_REQUIRE_ISA=avx2: this host does not dispatch the AVX2 kernels")
		}
	default:
		t.Fatalf("BLAKE3_REQUIRE_ISA=%q: the kernels are avx2 and avx512", isa)
	}
}
