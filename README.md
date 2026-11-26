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
go get github.com/forkcloser/go-blake3/v2
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
`BinaryAppender`. The state is 50 bytes for a hash of nothing, 32 more for
each subtree in progress, and the input held back: at most 18 KiB in all. It
names nothing about the host that wrote it: a state marshaled on one machine
restores on any other, whatever kernels each has, and continues from where it
was. Hosts hold back different amounts, so two hosts' states of the same
input need not be the same bytes. The state carries the key, so it is as secret
as the key: not to be stored or sent where the key would not be.

The state starts with a tag and a format version. `UnmarshalBinary` restores
the version this release writes and refuses any other: a state from a release
with another version is an error, never a different state.

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

A root that is not a parent, the node `CompressChunk`, `CompressBuffer` or
`CompressEigentree` returns, is made by setting `guts.FlagRoot` in its
`Flags`, not by passing the flag in.

## Migrating from v1

v1 is `github.com/forkcloser/blake3`, through v1.1.0, under the repository's
former name; so are v2's release candidates through v2.0.0-rc.4, as
`github.com/forkcloser/blake3/v2`. Digests, keyed digests, derived keys,
output streams and Bao encodings are the same bytes in v2. The
calls below are the ones that change, and a v1 call fails to compile against
v2 until it is changed. `enc` is a
`bao.Encoding{Group: group, Outboard: outboard}`.

| v1 | v2 |
| --- | --- |
| `import "github.com/forkcloser/blake3"`, `.../bao`, `.../guts` | `import "github.com/forkcloser/go-blake3/v2"`, `.../v2/bao`, `.../v2/guts` |
| `blake3.New(32, nil)` | `blake3.New()` |
| `blake3.New(32, key)`, `key` a 32-byte slice | `blake3.NewKeyed([32]byte(key))`; `blake3.Sum256Keyed` in one call |
| `blake3.New(n, key)`, then `h.Sum(nil)` for `n` bytes | `io.ReadFull(h.XOF(), out)`, `out` of `n` bytes; `Sum` and `Size` are 32 bytes |
| `blake3.DeriveKey(subKey, ctx, srcKey)` fills `subKey` | `subKey := blake3.DeriveKey(ctx, srcKey, n)` |
| `bao.EncodedSize(dataLen, group, outboard)`, an `int` | `enc.EncodedSize(dataLen)`, `int64` in and out |
| `bao.Encode(dst, data, dataLen, group, outboard)` | `enc.Encode(dst, data, dataLen)` |
| `bao.EncodeBuf(data, group, outboard)` | `enc.EncodeBuf(data)` |
| `ok, err := bao.Decode(dst, data, outboard, group, root)` | `err := enc.Decode(dst, data, tree, root)` |
| `ok, err := bao.DecodeSlice(dst, data, group, offset, length, root)` | `err := enc.DecodeSlice(dst, slice, offset, length, root)` |
| `bao.ExtractSlice(dst, data, outboard, group, offset, length)` | `enc.ExtractSlice(dst, data, tree, offset, length)` |
| `ok := bao.VerifyBuf(data, outboard, group, root)` | `err := enc.VerifyBuf(data, tree, root)` |
| `ok := bao.VerifyChunk(chunks, outboard, group, offset, root)` | `err := enc.VerifyChunk(chunks, tree, offset, root)` |
| `content, ok := bao.VerifySlice(data, group, offset, length, root)` | `content, err := enc.VerifySlice(slice, offset, length, root)` |
| `guts.BytesToWords`, `guts.WordsToBytes` | `encoding/binary`, little-endian |
| `guts.CompressBlocks`, `guts.CompressBlocksN` | `n.Counter = i` and `guts.CompressNode(n)` for output block `i` |

v1 took a non-nil `outboard` argument to mean an outboard encoding. v2 reads
that from `enc.Outboard`, to be set wherever v1 was handed an outboard tree:
without it `Decode`, `ExtractSlice` and `VerifyBuf` ignore `tree`. Where v1 returned a `bool`, v2 returns an error:
nil is verified content, and content that does not verify is
`bao.ErrVerification`, matched with `errors.Is`.

## Performance

On amd64 the AVX-512 and AVX2 kernels are selected at runtime from the CPU's
features. `Write` holds a group of sixteen chunks back and hands the kernels
whole groups, so a write costs one kernel call per 16 KiB; from 192 KiB
under AVX-512 and 256 KiB under AVX2 a write's trees go to goroutines, as do
large output reads. Medians of six runs of `just bench-amd64` on 2026-11-22
([run 45000959243](https://github.com/forkcloser/go-blake3/actions/runs/45000959243)),
a GitHub `ubuntu-24.04` runner with an Intel Xeon 6975P-C (4 vCPUs), Go
1.27.4; the AVX-512 kernels as dispatched, then the same binary with
`GODEBUG=cpu.avx512f=off` (AVX2) and with `cpu.avx2=off` as well (pure Go).
Half the range over the median is under 3.6% everywhere.

| Benchmark | AVX-512 | AVX2 | pure Go |
| --- | ---: | ---: | ---: |
| `Write`, io.Copy's 32 KiB, per KiB | 204 ns | 312 ns | 1.28 µs |
| `WriteSizes/4096` | 810 ns | 1.24 µs | 6.42 µs |
| `WriteSizes/32768` | 6.46 µs | 9.88 µs | 41.42 µs |
| `WriteSizes/131072` | 25.41 µs | 39.26 µs | 111.3 µs |
| `WriteSizes/1048576` | 121.8 µs | 175.2 µs | 655.8 µs |
| `XOF/65536` | 25.17 µs | 22.87 µs | 59.65 µs |
| `XOF/1048576` | 144.1 µs | 157.0 µs | 558.8 µs |
| `Sum256/64` | 100 ns | 100 ns | 100 ns |
| `Sum256/1024` | 1.47 µs | 1.47 µs | 1.47 µs |
| `Sum256/1048576` | 124.9 µs | 176.3 µs | 658.7 µs |

In throughput, hashing 1 MiB runs at 8.6 GB/s under AVX-512, 6.0 under AVX2
and 1.6 in pure Go, and io.Copy's 32 KiB writes at 5.1, 3.3 and 0.79; XOF
output at 1 MiB runs at 7.3, 6.7 and 1.9, the two SIMD paths near equal there.
Inputs of a chunk or less never reach the kernels and cost the same on every
path. A `Write` of up to 24 KiB allocates nothing on any path, nor one of up
to 32 KiB, io.Copy's size, under the SIMD kernels; in pure Go io.Copy's
writes are dealt out to goroutines, and allocate. On arm64 the pure-Go code
is the whole story: an Apple M5 Pro hashes 1 MiB at 4.5 GB/s and reads 1 MiB
of XOF at 5.1, with many cores to deal the tree out to.

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
chunk-group vectors. Five fuzz targets drive the kernels, the write scheduler,
the restoring of a marshaled state and the Bao decoders, and the race detector
runs the suite.

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
