// Package guts is the module's low-level surface: the BLAKE3 compression
// function, chunk and parent nodes, and the tree-hashing primitives that
// packages blake3 and bao are built from, for callers that hash their own
// trees. To hash data, use package blake3.
//
// What every function computes is fixed by the BLAKE3 specification and
// pinned by the official test vectors. On amd64 the AVX-512 and AVX2 kernels
// are selected at runtime from the CPU's features, with the pure-Go code as
// the fallback everywhere; every path produces identical results.
//
// The functions set the flags that say where a node sits in the tree:
// CompressChunk sets FlagChunkStart and FlagChunkEnd, ParentNode FlagParent,
// CompressBuffer and CompressEigentree each of them on the chunks and
// parents they compress. Their flags argument is the mode's, the same for
// every node of a tree: zero, FlagKeyedHash, FlagDeriveKeyContext or
// FlagDeriveKeyMaterial. FlagRoot is neither: it marks the last compression
// of the root node and no other. ParentNode compresses nothing, so it takes
// FlagRoot in flags; the node CompressChunk, CompressBuffer or
// CompressEigentree returns becomes the root by setting FlagRoot in its
// Flags, because in their flags argument it would mark every compression
// they do, and for more than one block the result would not be BLAKE3's.
//
// Each function says what it takes, and CompressEigentree alone checks it.
// An argument outside that is not refused: it gives a node that is no
// BLAKE3 tree's, or a runtime panic.
//
// Functions that take pointers read through them only for the duration of
// the call. Nothing here is safe for concurrent use on shared mutable
// arguments.
package guts

import "github.com/forkcloser/go-blake3/v2/internal/kernel"

// The domain flags a Node carries, and the tree's fixed sizes in bytes. The
// values are the BLAKE3 specification's, but for MaxSIMD.
const (
	FlagChunkStart        = kernel.FlagChunkStart        // 1 << 0
	FlagChunkEnd          = kernel.FlagChunkEnd          // 1 << 1
	FlagParent            = kernel.FlagParent            // 1 << 2
	FlagRoot              = kernel.FlagRoot              // 1 << 3
	FlagKeyedHash         = kernel.FlagKeyedHash         // 1 << 4
	FlagDeriveKeyContext  = kernel.FlagDeriveKeyContext  // 1 << 5
	FlagDeriveKeyMaterial = kernel.FlagDeriveKeyMaterial // 1 << 6

	// BlockSize is a compression block, ChunkSize a chunk of the tree, and
	// MaxSIMD the number of chunks CompressBuffer takes at once.
	BlockSize = kernel.BlockSize // 64
	ChunkSize = kernel.ChunkSize // 1024
	MaxSIMD   = kernel.MaxSIMD   // 16
)

// IV is the BLAKE3 initialization vector, the key of an unkeyed hash: the
// eight words
//
//	0x6A09E667 0xBB67AE85 0x3C6EF372 0xA54FF53A
//	0x510E527F 0x9B05688C 0x1F83D9AB 0x5BE0CD19
//
//nolint:gochecknoglobals // exported: an array cannot be a constant, and callers building their own trees read it
var IV = kernel.IV

// A Node is a chunk or parent of the BLAKE3 Merkle tree: the input to one
// compression, whose output is the node's chaining value, or, with FlagRoot
// set, the hash's output stream. Its fields are the compression function's
// arguments:
//
//	CV       [8]uint32  // the key, or the chaining value of the chunk's blocks so far
//	Block    [16]uint32 // the block compressed: input, or two children's chaining values
//	Counter  uint64     // a chunk's number in the input; 0 for a parent
//	BlockLen uint32     // the bytes of input in Block, at most BlockSize
//	Flags    uint32     // the Flag constants that apply
//
// On a root node Counter is the output block's number instead: CompressNode
// with FlagRoot set and Counter i returns bytes 64*i to 64*i+63 of the
// output stream, the first block's first 32 bytes being the digest.
type Node = kernel.Node

// CompressNode compresses a node into its 16-word output.
func CompressNode(n Node) [16]uint32 { return kernel.CompressNode(n) }

// ChainingValue compresses a node and returns the first eight words of its
// output, the chaining value a parent takes.
func ChainingValue(n Node) [8]uint32 { return kernel.ChainingValue(n) }

// ParentNode returns the Node that incorporates the chaining values of two
// child nodes.
func ParentNode(left, right [8]uint32, key *[8]uint32, flags uint32) Node {
	return kernel.ParentNode(left, right, key, flags)
}

// CompressChunk compresses a single chunk, at most ChunkSize bytes, and
// returns its final node, uncompressed: the chunk's last block, with the
// blocks before it compressed into CV. A longer chunk is not refused and is
// not split: it is compressed as one chunk of that many blocks, which is no
// BLAKE3 tree's. More than a chunk is CompressBuffer's or
// CompressEigentree's.
func CompressChunk(chunk []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return kernel.CompressChunk(chunk, key, counter, flags)
}

// CompressBuffer compresses the first buflen bytes of buf, up to MaxSIMD
// chunks starting at counter, and returns their root node. buf must be a
// complete array the caller owns: on amd64 the SIMD kernels read all of it
// whatever buflen says, so a shorter slice reinterpreted as the array would
// be read past its length.
func CompressBuffer(buf *[MaxSIMD * ChunkSize]byte, buflen int, key *[8]uint32, counter uint64, flags uint32) Node {
	return kernel.CompressBuffer(buf, buflen, key, counter, flags)
}

// CompressEigentree compresses a buffer of 2^n whole chunks starting at
// counter, in parallel when it is large, and returns their root node. It
// panics on any other length: a tree is a power-of-two number of chunks,
// and a partial chunk belongs to CompressChunk or CompressBuffer.
func CompressEigentree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return kernel.CompressEigentree(buf, key, counter, flags)
}

// Eigentrees returns the heights of the subtrees that take a tree from
// counter to counter+chunks: the sequence climbs while counter is not yet
// aligned, then descends, and each height h is a tree of 2^h chunks.
func Eigentrees(counter, chunks uint64) []int { return kernel.Eigentrees(counter, chunks) }
