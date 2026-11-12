# blake3

[BLAKE3](https://github.com/BLAKE3-team/BLAKE3) for Go: hashing, keyed
hashing, key derivation, extendable output, and
[Bao](https://github.com/oconnor663/bao) verified streaming, with AVX-512 and
AVX2 kernels on amd64 and pure Go everywhere else.

This module began as a fork of
[lukechampine/blake3](https://github.com/lukechampine/blake3)
(`lukechampine.com/blake3`). What was taken from upstream, what was changed,
and where the two have since parted is recorded in
[UPSTREAM.md](./UPSTREAM.md); from v2 the API is this module's own.

## Install

```
go get github.com/forkcloser/blake3/v2
```

Go 1.26 or later. The one dependency is `golang.org/x/sys`, for CPU feature
detection on amd64.

## Packages

| Package | What it is |
| --- | --- |
| `blake3` | A `hash.Hash` in each of BLAKE3's three modes, the one-shots, a seekable random-access extendable output, and a hash state that marshals and restores on any host. |
| `bao` | Bao verified streaming: encodings that carry the hash tree beside the data, or apart from it, so any prefix or slice verifies against the root alone. |
| `guts` | The tree-hashing primitives the two packages above are built from, for callers that hash their own trees. |

## Hashing

A `Hasher` is a [`hash.Hash`](https://pkg.go.dev/hash#Hash) with a 32-byte
digest. `New` is the plain hash, `NewKeyed` the keyed mode under a 32-byte
key (a MAC), `NewDeriveKey` the key-derivation mode under a context string.

```go
h := blake3.New()
if _, err := io.Copy(h, r); err != nil {
	return err
}
digest := h.Sum(nil) // 32 bytes

sum := blake3.Sum256(data)           // the same, in one call
wide := blake3.Sum512(data)          // the first 64 bytes of the output
mac := blake3.Sum256Keyed(key, data) // key is a [32]byte
```

Key derivation takes a context string that should be hardcoded, globally
unique and application-specific, so that no two applications derive the same
key from the same material; a good form is `[application] [commit timestamp]
[purpose]`. The derived key may be any length.

```go
const ctx = "example.com 2026-11-12 session tokens v1"

sub := blake3.DeriveKey(ctx, master, 64) // 64 bytes

d := blake3.NewDeriveKey(ctx) // the same, streaming the material
d.Write(master)
sub = d.Sum(nil) // 32 bytes; more through d.XOF()
```

A `Hasher` holds no pointers, so copying one forks its state: the copy and
the original continue independently. `Write` never fails, and `Sum` leaves the
state alone, so it may be called at any point and as often as wanted. A
`Hasher` is not safe for concurrent use.

## Extendable output

`Hasher.XOF` returns an `OutputReader` over the 2^64 - 1 bytes of output of
everything written so far: an `io.ReadSeeker`, and an `io.ReaderAt` whose
`ReadAt` leaves the position alone and may run from several goroutines at
once. The first 32 bytes are the digest.

```go
xof := h.XOF()

buf := make([]byte, 1<<20)
if _, err := xof.ReadAt(buf, 1<<30); err != nil { // one MiB, a GiB in
	return err
}
```

## Keeping a hash in progress

A `Hasher` implements `encoding.BinaryMarshaler`, `BinaryUnmarshaler` and
`BinaryAppender`. The state is at most 18 KiB and a few dozen bytes, a few
dozen alone for a hash of little, and names nothing about the host that
wrote it: a state marshaled
on one machine restores on any other, whatever kernels each has, and
continues from where it was. The state carries the key, so it is as secret as
the key: not to be stored or sent where the key would not be.

```go
state, err := h.MarshalBinary()
if err != nil {
	return err
}

var again blake3.Hasher
if err := again.UnmarshalBinary(state); err != nil { // not a state MarshalBinary made
	return err
}
again.Write(rest)
```

## Verified streaming

Package `bao` implements the [Bao format](https://github.com/oconnor663/bao):
the BLAKE3 tree written beside the data (combined) or apart from it
(outboard), so that a reader verifies the content against the 32-byte root
as it streams, and a slice of the content comes with the nodes that prove it.
An `Encoding` value names the format, its zero value standard Bao; every
operation is a method on it. Content that does not verify is
`bao.ErrVerification`, and a decoder writes nothing it has not verified.

```go
enc := bao.Encoding{} // {Outboard: true} keeps the tree apart from the data

// Encode: dst is an io.WriterAt with room for enc.EncodedSize(size) bytes;
// root is the BLAKE3 hash of the data.
root, err := enc.Encode(dst, src, size)
if err != nil {
	return err
}

// Decode streams the verified content to w; the third argument is the
// outboard tree, nil for a combined encoding.
if err := enc.Decode(w, encoded, nil, root); err != nil {
	return err // bao.ErrVerification, or the reader's or writer's own
}

// A slice: the content from offset for length bytes, and the nodes that
// prove it, which anyone verifies with the root alone.
if err := enc.ExtractSlice(sliceW, encoded, nil, offset, length); err != nil {
	return err
}
if err := enc.DecodeSlice(w, slice, offset, length, root); err != nil {
	return err
}

// In memory.
combined, root := enc.EncodeBuf(data)
if err := enc.VerifyBuf(combined, nil, root); err != nil {
	return err
}
content, err := enc.VerifySlice(slice, offset, length, root)
if err != nil {
	return err
}
```

`Encoding.Group` sets the chunk-group size as a power of two, `1024 << Group`
bytes a leaf; `0` is standard Bao, and larger groups trade finer slices for a
smaller tree and the SIMD kernels' full width.

## Building your own trees

Package `guts` is the compression function, chunk and parent nodes, and the
eigentree helpers, for a format that lays BLAKE3's tree out its own way.
Everything it computes is fixed by the specification.

```go
left := guts.ChainingValue(guts.CompressChunk(chunk0, &guts.IV, 0, 0))
right := guts.ChainingValue(guts.CompressChunk(chunk1, &guts.IV, 1, 0))
root := guts.ParentNode(left, right, &guts.IV, guts.FlagRoot)
out := guts.CompressNode(root) // the first eight words are the digest
```

## Performance

On amd64 the AVX-512 and AVX2 kernels are selected at runtime from the CPU's
features. `Write` holds a group of sixteen chunks back and hands the kernels
whole groups, so a write costs one kernel call per 16 KiB and nothing else up
to 128 KiB; from 192 KiB a write's trees go to goroutines, as do large
output reads. Medians of six runs of `just bench-amd64` on 2026-11-12
([run 43134512263](https://github.com/forkcloser/blake3/actions/runs/43134512263)),
a GitHub `ubuntu-24.04` runner with an AMD EPYC 9V74 (Zen 4, 4 vCPUs), Go
1.27.4; the AVX-512 kernels as dispatched, then the same binary with
`GODEBUG=cpu.avx512f=off` (AVX2) and with `cpu.avx2=off` as well (pure Go).
Half the range over the median is under 7.1% everywhere.

| Benchmark | AVX-512 | AVX2 | pure Go |
| --- | ---: | ---: | ---: |
| `Write`, io.Copy's 32 KiB, per KiB | 280 ns | 350 ns | 1.82 µs |
| `WriteSizes/4096` | 1.12 µs | 1.41 µs | 7.25 µs |
| `WriteSizes/32768` | 8.93 µs | 11.33 µs | 57.06 µs |
| `WriteSizes/131072` | 34.99 µs | 44.90 µs | 127.7 µs |
| `WriteSizes/1048576` | 159.8 µs | 237.6 µs | 796.1 µs |
| `XOF/65536` | 22.34 µs | 22.54 µs | 62.61 µs |
| `XOF/1048576` | 130.2 µs | 135.0 µs | 670.2 µs |
| `Sum256/64` | 110 ns | 110 ns | 110 ns |
| `Sum256/1024` | 1.65 µs | 1.65 µs | 1.65 µs |
| `Sum256/1048576` | 163.7 µs | 243.9 µs | 796.1 µs |

In throughput, hashing 1 MiB runs at 6.6 GB/s under AVX-512, 4.4 under AVX2
and 1.3 in pure Go, and io.Copy's 32 KiB writes at 3.7, 2.9 and 0.56; XOF
output at 1 MiB runs at 8.1, 7.8 and 1.6, the two SIMD paths equal there.
Inputs of a chunk or less never reach the kernels and cost the same on every
path. Writes under 48 KiB allocate nothing. On arm64 the pure-Go code is the
whole story: an Apple M5 Pro hashes 1 MiB at 4.5 GB/s and reads 1 MiB of XOF
at 5.1, with many cores to deal the tree out to.

`just bench` runs the benchmarks on the host; `just bench-amd64` runs the
three paths on a linux/amd64 host with AVX-512, and the `bench` workflow does
the same on a runner when one with `avx512f` is drawn. A cloud runner is one
machine on one day; the artifact of a run is its raw output.

## Correctness

Every path is held to two independent statements of the algorithm: the
BLAKE3 team's published test vectors, at the lengths they cover, and a port
of the specification's reference implementation
(`internal/refimpl`), at every length and output offset the bounded walks
visit; Bao is held to its reference encoder, the Bao golden files and abao's
chunk-group vectors. Four fuzz targets drive the kernels, the write scheduler
and the Bao decoders, and the race detector runs the suite.

Every dispatch arm runs on every CI run, and is asserted by name: the AVX-512
kernels under Intel SDE's emulation, on a runner whose own CPU may lack them;
the AVX2 kernels alone, and the pure-Go arms, forced through `GODEBUG` on
amd64; and pure Go natively on arm64. The suite runs on Linux (amd64, arm64),
Windows (amd64, arm64) and macOS (arm64). The amd64 assembly is generated
from `avo/gen.go`, regenerated and compared on every lint, and passes
`go vet`.

## Limitations

- A single block, 64 bytes or less, is compressed in Go on every path; the
  kernels start at two chunks. Inputs of a chunk or less are the same speed
  everywhere.
- The kernels compute a full group of sixteen chunks whatever they are
  handed. `Write` hands them whole groups while hashing, but the group held
  back at `Sum`, and Bao chunk groups below sixteen chunks, pay a whole kernel
  for what they hold.
- Each kernel inlines all seven rounds, so the generated assembly is long;
  one routine for both instruction sets would be smaller, at some cost.
- No 32-bit or WebAssembly target is tested.

## Development

Every task runs through `just`: `just lint` and `just test` are the whole
gate; `just gen` regenerates the assembly from `avo/gen.go` (its own module,
so avo never enters the library's dependency graph) and `just lint-generated`
proves the committed file is its output; `just test-isa avx2` and
`just test-isa generic` require one amd64 arm; `just test-avx512-emulated`
runs the kernel and root tests under Intel SDE. The tooling is the
organization's; see
[CONTRIBUTING](https://github.com/forkcloser/.github/blob/main/.github/CONTRIBUTING.md).

## Security

Report a vulnerability as the organization's
[SECURITY](https://github.com/forkcloser/.github/blob/main/.github/SECURITY.md)
policy says. Two properties worth knowing: a marshaled `Hasher` state carries
its key, and a Bao decoder writes only content it has verified.

## License

[MIT](./LICENSE).
