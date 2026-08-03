// The haveAVX* flags only exist on amd64: they gate the assembly kernels,
// and declaring them on other architectures would drag the cpuid dependency
// into builds that cannot use it.

//go:build amd64 && !darwin

package guts

import "github.com/klauspost/cpuid/v2"

var (
	haveAVX2   = cpuid.CPU.Supports(cpuid.AVX2)
	haveAVX512 = cpuid.CPU.Supports(cpuid.AVX512F)
)
