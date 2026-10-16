# Upstream

Fork of <https://github.com/lukechampine/blake3> (`master`).

- Forked from: [`dd9ffb9`](https://github.com/lukechampine/blake3/commit/dd9ffb94dc48974796a2c1aa2082d0c8cc284098), committed 2025-05-08
- Reviewed through: [`dd9ffb9`](https://github.com/lukechampine/blake3/commit/dd9ffb94dc48974796a2c1aa2082d0c8cc284098), reviewed 2026-10-05

Every upstream commit after the fork point, through the reviewed one, is in
exactly one of the tables below. A port is a commit of this repository,
adapted to the fork; nothing is merged from upstream.

## Incorporated

| Upstream commit | Here | What |
| --- | --- | --- |

## Not incorporated

| Upstream commit | Why |
| --- | --- |

## Changes upstream does not have

| Change | Here |
| --- | --- |
| `OutputReader.Seek` to an offset not aligned to the XOF's buffer no longer corrupts the next `Read`; `Seek` rejects offsets that would wrap | [`1a8774f`](https://github.com/forkcloser/blake3/commit/1a8774f) |
| `New` panics on a negative size or a key of the wrong length; `bao` rejects `offset+length` overflow in `ExtractSlice`, `DecodeSlice` and `VerifyChunk`, verifies the root of an empty encoding, counts leaves exactly, and sizes the parent-merge stack for 2⁶⁴ bytes | [`fe4fe18`](https://github.com/forkcloser/blake3/commit/fe4fe18) |
| Small and unaligned XOF reads compress only the blocks they need (`guts.CompressBlocksN`, a lazily filled `OutputReader`); `bao.Encode` reuses one parent buffer instead of allocating per node | [`eae253f`](https://github.com/forkcloser/blake3/commit/eae253f) |
| The write path schedules eigentrees by total size and parallelizes from 24 KiB | [`96c0ae0`](https://github.com/forkcloser/blake3/commit/96c0ae0), [`31e363f`](https://github.com/forkcloser/blake3/commit/31e363f) |
| The amd64 kernels broadcast scalar arguments from registers and never read past `len(buf)` (upstream's assembly failed `go vet`'s asmdecl and read up to 16 KiB past the buffer) | https://github.com/forkcloser/blake3/pull/8 |
| `bao.MaxGroup` bounds the chunk group; `group` and `dataLen` are checked at every entry point with a named panic | https://github.com/forkcloser/blake3/pull/8 |
| The exported API is upstream's with `guts.CompressBlocksN` and `bao.MaxGroup` added and the five deprecated `Bao*` root wrappers removed | https://github.com/forkcloser/blake3/pull/8 |
| A zero-length Bao slice carries the chunk at its start point, and `VerifyChunk` with no chunks is false rather than vacuously true; a header that cannot be read is the reader's error; `VerifyChunk` counts nodes in `uint64` | https://github.com/forkcloser/blake3/pull/98 |
| `guts.CompressEigentree` refuses a partial chunk | https://github.com/forkcloser/blake3/pull/99 |
| `bao.ErrSliceLength` is exported; a negative length panics in `Encode` as in `EncodedSize` (breaking) | https://github.com/forkcloser/blake3/pull/103 |
