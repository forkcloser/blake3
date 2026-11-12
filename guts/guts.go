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
// Functions that take pointers read through them only for the duration of
// the call. Nothing here is safe for concurrent use on shared mutable
// arguments.
package guts

import "github.com/forkcloser/blake3/v2/internal/kernel"

// The domain flags a Node carries, and the tree's fixed sizes in bytes.
const (
	FlagChunkStart        = kernel.FlagChunkStart
	FlagChunkEnd          = kernel.FlagChunkEnd
	FlagParent            = kernel.FlagParent
	FlagRoot              = kernel.FlagRoot
	FlagKeyedHash         = kernel.FlagKeyedHash
	FlagDeriveKeyContext  = kernel.FlagDeriveKeyContext
	FlagDeriveKeyMaterial = kernel.FlagDeriveKeyMaterial

	// BlockSize is a compression block, ChunkSize a chunk of the tree, and
	// MaxSIMD the number of chunks CompressBuffer takes at once.
	BlockSize = kernel.BlockSize
	ChunkSize = kernel.ChunkSize
	MaxSIMD   = kernel.MaxSIMD
)

// IV is the BLAKE3 initialization vector, the key of an unkeyed hash.
//
//nolint:gochecknoglobals // exported: an array cannot be a constant, and callers building their own trees read it
var IV = kernel.IV

// A Node is a chunk or parent of the BLAKE3 Merkle tree: the input to one
// compression, whose output is the node's chaining value, or, with FlagRoot
// set, the hash's output stream.
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
// returns its final node, uncompressed.
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
