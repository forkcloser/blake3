# Changelog

All notable changes to this fork are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). Changes are described against the
fork point, upstream [`lukechampine/blake3`](https://github.com/lukechampine/blake3)
v1.4.1 (commit `dd9ffb9`), which is also upstream's current `master`. For a
user of `lukechampine.com/blake3`, switching is a change of import path plus
the one removal listed below.

## [1.0.0-rc.1] - 2026-09-07

### Fixed

- `OutputReader.Seek` to an offset not aligned to the XOF's internal buffer
  corrupted the following `Read` (bytes from up to 960 positions ahead until
  the stale buffer drained). The buffer is now an absolute window of the
  stream, filled lazily; `Seek` rejects offsets that would wrap past the end
  of the stream instead of wrapping silently.
- `New` panics with a message on a negative size or a key that is not 32
  bytes; over-length keys were silently truncated.
- `bao`: `offset+length` overflow is rejected in `ExtractSlice`, `DecodeSlice`
  and `VerifyChunk` (`DecodeSlice` reported success on overflowing bounds
  while verifying nothing); the empty encoding is verified rather than
  rejected in `VerifyChunk` and its root is verified in `DecodeSlice`;
  `VerifyChunk` counts leaves exactly; the parent-merge stack in
  `compressGroup` is sized for 2⁶⁴ bytes (50 levels; upstream's bound was
  misread as 38).
- `bao`: every function validates its `group` — a value outside
  `[0, MaxGroup]` panics with a message where it previously failed with a
  negative shift, a division by zero, a `makeslice` fault or a slice-bounds
  fault depending on the value — and `Encode` rejects a negative length
  before writing anything.
- amd64: a short eigentree is copied into a stack buffer before it reaches
  the SIMD kernels, which read a full 16 KiB; bytes past the input's length
  are no longer touched.
- amd64: the generated assembly passes `go vet` (`asmdecl`). Upstream's
  kernels broadcast scalar arguments straight from the stack frame, which
  vet rejects; the generator now loads them into a register first. Kernel
  loops are byte-identical.

### Changed

- `Hasher.Write` schedules a write's eigentrees by total size: serial below
  24 KiB of trees, above that one goroutine per tree of 16 chunks or more
  and one per run of smaller trees. `CompressEigentree` deals 16 KiB groups
  out in NumCPU-capped runs, so a 1 MiB write no longer spawns 64
  goroutines. Measured against upstream v1.4.1 on the same machine (see
  README): 4 KiB writes 50% faster, 32 KiB (io.Copy's buffer) 11–13%
  faster, 48 KiB 23% faster, 64 KiB to 1 MiB 4–10% faster, 24 KiB at
  parity; allocations 22–84% lower wherever either side allocates.
- XOF reads compress only the blocks they need (`guts.CompressBlocksN`) and
  run across CPUs only when each goroutine gets at least 16 KiB of work. A
  `Seek(0)` followed by a 64-byte `Read` is ~300× faster; a 64 KiB read
  allocates 9 times instead of 58.
- `bao.Encode` reuses one parent buffer (about 2050 allocations per MiB
  became 3); buffered readers are recommended for decoding and documented.
- Benchmarks use upstream's plain `b.N` loop so the numbers compare one to
  one; `BenchmarkWriteSizes` covers 4 KiB to 1 MiB writes.

### Added

- `guts.CompressBlocksN` and `bao.MaxGroup`.
- Package documentation stating the contracts: concurrency, copyability of
  `Hasher`, `Write`/`Sum` semantics, XOF end-of-stream and `Seek` bounds,
  and a stability statement for `guts`.
- Tests: the eigentree fast path against the chunk-at-a-time reference
  (every shape to 64 chunks, many starting counters) plus a fuzz target on
  the split pattern; XOF seeks at buffer-unaligned offsets and block
  counters past 2³²; the `bao` decoders fuzzed on hostile encodings and
  slice bounds. `just lint-generated` regenerates the amd64 assembly from
  `avo/gen.go` (its own module, so `avo` never enters the library's
  dependency graph) and fails if the committed file differs.
- The limen baseline: aqua-pinned toolchain, golangci configuration, CI on
  darwin/arm64, linux/amd64, linux/arm64, windows/amd64 and windows/arm64,
  a fuzz job, and the same-host benchmark comparison in the README.

### Removed

- The five deprecated root-package `Bao*` wrappers upstream kept after
  moving Bao to its own package. Use package `bao`.
