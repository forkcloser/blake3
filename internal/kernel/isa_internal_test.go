//go:build amd64

package kernel

import (
	"os"
	"testing"
)

// TestRequiredISA fails when BLAKE3_REQUIRE_ISA names a dispatch arm this
// process does not take: avx512, avx2, or generic (every kernel off, the
// amd64 fallbacks). The ISA lanes set it: the other tests then prove that
// arm, not whichever the host happened to select, produced the
// specification's results. Unset, the test is skipped.
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
		// The dispatch prefers AVX-512 wherever both are on, so AVX2 alone is
		// what this arm means.
		if !haveAVX2 || haveAVX512 {
			t.Fatal("BLAKE3_REQUIRE_ISA=avx2: the AVX2 kernels are not dispatched alone; GODEBUG turns avx512f off")
		}
	case "generic":
		if haveAVX2 || haveAVX512 {
			t.Fatal("BLAKE3_REQUIRE_ISA=generic: this host dispatches SIMD kernels; GODEBUG turns them off")
		}
	default:
		t.Fatalf("BLAKE3_REQUIRE_ISA=%q: the arms are avx512, avx2 and generic", isa)
	}
}
