// The haveAVX* flags only exist on amd64: they gate the assembly kernels,
// and declaring them on other architectures would drag the detection
// dependency into builds that cannot use it.

//go:build amd64

package guts

import "golang.org/x/sys/cpu"

// Both flags already fold in OS register support (XGETBV, and the sysctl
// probe on darwin), so no per-OS fallback is needed here.
var (
	haveAVX2   = cpu.X86.HasAVX2
	haveAVX512 = cpu.X86.HasAVX512F
)
