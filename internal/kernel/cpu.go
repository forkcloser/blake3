// The haveAVX* flags only exist on amd64: they gate the assembly kernels,
// and declaring them on other architectures would drag the detection
// dependency into builds that cannot use it.

//go:build amd64

package kernel

import "golang.org/x/sys/cpu"

// Both flags already fold in OS register support (XGETBV, and the sysctl
// probe on darwin), so no per-OS fallback is needed here.
//
// A flag asks for every extension its kernels' instructions need, which the
// generated file names in its "Requires:" lines: the AVX-512 kernels set
// their masks with KXNORD, an AVX512BW instruction, and a CPU with AVX512F
// alone takes the AVX2 kernels.
//
//nolint:gochecknoglobals // the CPU's features, detected once at init and only read after
var (
	haveAVX2   = cpu.X86.HasAVX2
	haveAVX512 = cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW
)
