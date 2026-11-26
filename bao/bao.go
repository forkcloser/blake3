// Package bao implements BLAKE3 verified streaming: encodings that carry the
// hash tree alongside (or, "outboard", apart from) the data, so that any
// prefix or slice of the data can be verified against the root hash as it is
// read. See https://github.com/oconnor663/bao for the format. Leaves of more
// than one chunk, a Group above 0, are an extension to it, as in the abao
// project; nothing in an encoding says which Group it was written with. It
// is read with the same one: with another it does not verify, unless the
// content is no longer than a leaf of the smaller of the two, when it is
// one leaf and the same bytes under both.
//
// An [Encoding] value names the format's two parameters, the chunk-group
// size and whether the tree is outboard, and every operation is a method on
// it; its zero value is standard Bao.
//
// A verification that fails is [ErrVerification]; a slice request that the
// length an encoding declares cannot answer is [ErrSliceLength]; an input
// that ends before its encoding does, with io.EOF or an error wrapping it,
// is io.ErrUnexpectedEOF. Any other error of a reader is returned as the
// reader reported it, and a writer's is wrapped. A writer that reports no
// error but a count other than the bytes it was given is an error too:
// io.ErrShortWrite, wrapped, when the count is short.
package bao

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"github.com/forkcloser/go-blake3/v2/internal/kernel"
)

// MaxGroup is the largest Encoding.Group any method accepts. It is a 1 GiB
// chunk group, the largest whose size in bytes fits an int on every
// platform.
const MaxGroup = 20

// An Encoding is a Bao format: how many chunks a leaf holds, and where the
// tree lives. The zero value is standard Bao, one-chunk groups with the
// tree interleaved. An Encoding is a value, safe to share and to compare.
type Encoding struct {
	// Group is the chunk-group size as a power of two: a leaf is
	// guts.ChunkSize << Group bytes. Standard Bao is 0; any value in
	// [0, MaxGroup] is accepted, and another panics at the first method.
	// Every method but EncodedSize and VerifyChunk holds one whole group in
	// memory, however short its input: 1 GiB at MaxGroup.
	Group int
	// Outboard puts the tree in its own stream, apart from the data, rather
	// than interleaved with it.
	Outboard bool
}

// The errors this package names. The others it returns are a reader's or a
// writer's own, io.ErrUnexpectedEOF for an input cut short, and for a writer
// that miscounts io.ErrShortWrite or an error with no name.
var (
	// ErrVerification is content that does not verify against the root: a
	// parent node or a chunk group, any of them, that does not hash to the
	// chaining value held for it. It is also what the in-memory verifiers
	// make of an encoding too short to verify or with bytes left over
	// (VerifyBuf, VerifySlice), and of chunks or a tree that do not fit
	// the content (VerifyChunk, whose doc lists the cases).
	ErrVerification = errors.New("bao: verification failed")
	// ErrSliceLength is a slice request that does not fit the encoded
	// data: its offset plus length overflows, or ends past the content
	// length the encoding declares. That length is the header's, which
	// nothing has verified when the request is refused: the error says
	// this encoding cannot answer the request, not that the content is
	// shorter than the request. Nothing was read past the header, and
	// nothing was written.
	ErrSliceLength = errors.New("bao: slice does not fit the encoded data")
)

// The encoding's fixed sizes, in bytes.
const (
	// headerSize is the little-endian data length the encoding starts with.
	headerSize = 8
	// cvSize is one chaining value: eight little-endian words.
	cvSize = 32
	// parentSize is a parent node: its children's two chaining values.
	parentSize = 2 * cvSize
)

func bytesToCV(b []byte) (cv [8]uint32) {
	_ = b[31] // bounds check hint
	for i := range cv {
		cv[i] = binary.LittleEndian.Uint32(b[4*i:])
	}

	return cv
}

func cvToBytes(cv *[8]uint32) *[32]byte {
	var b [32]byte
	for i, w := range cv {
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}

	return &b
}

func compressGroup(p []byte, counter uint64) kernel.Node {
	// The stack holds one chaining value per bit of the count of buffers
	// pushed. A group is at most ChunkSize << MaxGroup bytes, 2^16 buffers
	// of MaxSIMD chunks with the last never pushed, so 16 entries are
	// reachable. The array holds 50, what 2^64 bytes would take
	// (64 - 10 - 4): room to spare, and not what bounds a group, which
	// MaxGroup does.
	var (
		stack [50][8]uint32
		sc    uint64
	)

	pushSubtree := func(cv [8]uint32) {
		i := 0
		for sc&(1<<i) != 0 {
			cv = kernel.ChainingValue(kernel.ParentNode(stack[i], cv, &kernel.IV, 0))
			i++
		}

		stack[i] = cv
		sc++
	}

	var (
		buf    [kernel.MaxSIMD * kernel.ChunkSize]byte
		buflen int
	)
	for len(p) > 0 {
		if buflen == len(buf) {
			pushSubtree(
				kernel.ChainingValue(kernel.CompressBuffer(&buf, buflen, &kernel.IV, counter+(sc*kernel.MaxSIMD), 0)),
			)
			buflen = 0
		}

		n := copy(buf[buflen:], p)
		buflen += n
		p = p[n:]
	}

	n := kernel.CompressBuffer(&buf, buflen, &kernel.IV, counter+(sc*kernel.MaxSIMD), 0)
	for i := bits.TrailingZeros64(sc); i < bits.Len64(sc); i++ {
		if sc&(1<<i) != 0 {
			n = kernel.ParentNode(stack[i], kernel.ChainingValue(n), &kernel.IV, 0)
		}
	}

	return n
}

// EncodedSize returns the size of the encoding of dataLen bytes of data:
// the whole of it when the tree is interleaved, the tree alone when it is
// outboard. It panics if dataLen is negative, or if the size does not fit
// an int64: a combined encoding's does not once data and tree together pass
// math.MaxInt64 bytes, from about 7.5 EiB of data at Group 0.
func (e Encoding) EncodedSize(dataLen int64) int64 {
	groupSize := int64(e.groupSize()) // #nosec G115 -- at most 1 GiB

	if dataLen < 0 {
		panic("bao: negative data length")
	}

	size := int64(headerSize)

	if dataLen > 0 {
		// Counted by division: dataLen rounded up to a whole group first
		// overflows when it is within a group of math.MaxInt64.
		leaves := dataLen / groupSize
		if dataLen%groupSize != 0 {
			leaves++
		}

		size += (2*leaves - 2) * cvSize // a full binary tree: one parent fewer than leaves
	}

	if !e.Outboard {
		// The tree alone is under 2^60 bytes whatever the length: it is the
		// data beside it that an int64 may not hold, and the sum is checked
		// before it is made, not after it has wrapped.
		if dataLen > math.MaxInt64-size {
			panic("bao: data length too large for a combined encoding")
		}

		size += dataLen
	}

	return size
}

// Encode reads dataLen bytes of data and writes its encoding to dst: the
// tree, and the data interleaved with it unless the Encoding is outboard.
// It returns the tree's root, the BLAKE3 hash of the data, and the zero
// root beside an error.
//
// dst is not written sequentially: a parent node is written after the
// subtrees below it, at its offset before them. The writes fill the
// encoding's EncodedSize bytes, each once, and nothing past them; whether
// dst must hold that room beforehand is dst's own rule. A negative dataLen
// is a programming error and panics, as it does in EncodedSize, before
// anything is written.
//
// A source that fails is its own error, as it reports it, and one that
// ends before dataLen bytes is io.ErrUnexpectedEOF. A destination that
// fails is its error wrapped with the offset of the write, and so is
// io.ErrShortWrite for a write it took less of than it was given.
func (e Encoding) Encode(dst io.WriterAt, data io.Reader, dataLen int64) ([32]byte, error) {
	groupSize := e.groupSize()

	if dataLen < 0 {
		panic("bao: negative data length")
	}

	enc := &encoder{
		dst:       dst,
		data:      data,
		buf:       make([]byte, groupSize),
		groupSize: groupSize,
		outboard:  e.Outboard,
	}

	// Each node is written at its pre-order offset as soon as it is known,
	// a parent after both its subtrees: one pass over the data, and the
	// reason dst is an io.WriterAt. Groups are hashed one at a time, never
	// two in one kernel call, so the kernels' sixteen chunks at once are
	// filled only from Group 4 up.
	binary.LittleEndian.PutUint64(enc.buf[:headerSize], uint64(dataLen))
	enc.write(enc.buf[:headerSize], 0)
	_, root := enc.encode(uint64(dataLen), kernel.FlagRoot, headerSize)

	if enc.err != nil {
		// What the walk hashed around its error is the root of no data: a
		// caller that drops the error must not be left holding it.
		return [32]byte{}, enc.err
	}

	return *cvToBytes(&root), nil
}

// encoder is Encode's walk of the tree. Its first error stops the walk: every
// step after it does nothing, and Encode returns it.
type encoder struct {
	dst       io.WriterAt
	data      io.Reader
	buf       []byte
	groupSize uint64
	outboard  bool
	counter   uint64
	err       error
	// parentBuf is reused for all parent nodes; it escapes into dst.WriteAt,
	// so a per-node buffer would mean a heap allocation per node
	parentBuf [parentSize]byte
}

// readFull reads the whole of p. No read in this package is one an input
// may lack, the header included: an input that ends before it, with io.EOF
// or an error wrapping it, is cut short.
func readFull(r io.Reader, p []byte) error {
	_, err := io.ReadFull(r, p)
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}

	return err //nolint:wrapcheck // the reader's own error, as the package promises it
}

// errWriteCount is a destination that reports, with no error, a count it
// cannot have written: negative, or more than it was given.
var errWriteCount = errors.New("bao: the destination reported an impossible write count")

// writeFull writes the whole of p to w. io.Writer forbids a count short of
// len(p) with no error, and a destination that returns one all the same has
// dropped bytes: that is io.ErrShortWrite, as io.Copy has it. A write whose
// count is not read here reports content as delivered that was not.
func writeFull(w io.Writer, p []byte) error {
	n, err := w.Write(p)

	return writeResult(n, len(p), err)
}

// writeFullAt is writeFull for an io.WriterAt, at off.
func writeFullAt(w io.WriterAt, p []byte, off int64) error {
	n, err := w.WriteAt(p, off)

	return writeResult(n, len(p), err)
}

// writeResult is the outcome of a write of want bytes as an error: the
// destination's own when it reported one, whatever its count; otherwise
// io.ErrShortWrite for a count below want, and errWriteCount for one that is
// negative or above it.
func writeResult(n, want int, err error) error {
	switch {
	case err != nil:
		return err
	case n < 0 || n > want:
		return errWriteCount
	case n < want:
		return io.ErrShortWrite
	default:
		return nil
	}
}

func (e *encoder) read(p []byte) []byte {
	if e.err == nil {
		e.err = readFull(e.data, p)
	}

	return p
}

func (e *encoder) write(p []byte, off uint64) {
	if e.err != nil {
		return
	}

	// #nosec G115 -- an offset inside the encoding of an int64-length input:
	// past 2^63 only for exabyte inputs, where WriteAt refuses it.
	if err := writeFullAt(e.dst, p, int64(off)); err != nil {
		e.err = fmt.Errorf("bao: write the encoding at offset %d: %w", off, err)
	}
}

// encode writes the subtree of bufLen bytes whose encoding starts at off. It
// returns how many chaining values the subtree's parent nodes hold, and the
// subtree's own chaining value.
func (e *encoder) encode(bufLen uint64, flags uint32, off uint64) (uint64, [8]uint32) {
	if e.err != nil {
		return 0, [8]uint32{}
	} else if bufLen <= e.groupSize {
		g := e.read(e.buf[:bufLen])
		if !e.outboard {
			e.write(g, off)
		}

		n := compressGroup(g, e.counter)
		e.counter += bufLen / kernel.ChunkSize
		n.Flags |= flags

		return 0, kernel.ChainingValue(n)
	}

	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)
	lchildren, l := e.encode(mid, 0, off+parentSize)

	llen := lchildren * cvSize
	if !e.outboard {
		llen += (mid / e.groupSize) * e.groupSize
	}

	rchildren, r := e.encode(bufLen-mid, 0, off+parentSize+llen)
	for i := range l {
		binary.LittleEndian.PutUint32(e.parentBuf[4*i:], l[i])
		binary.LittleEndian.PutUint32(e.parentBuf[cvSize+4*i:], r[i])
	}

	e.write(e.parentBuf[:], off)

	return 2 + lchildren + rchildren, kernel.ChainingValue(kernel.ParentNode(l, r, &kernel.IV, flags))
}

// Decode reads an encoding, the data from data and the tree from tree when
// the Encoding is outboard (tree is ignored otherwise), and streams the
// verified content to dst. Content that does not verify against root is
// ErrVerification, and nothing unverified is written; an encoding cut short
// is io.ErrUnexpectedEOF, after the content that verified before the cut.
// Decode reads to the encoding's end and no further: bytes that follow it,
// in data or in tree, are left unread and are no error. VerifyBuf is the
// one that refuses them.
//
// Decode reads the tree 64 bytes at a time, so if the readers are
// unbuffered (e.g. os.File), wrapping them in a bufio.Reader will
// significantly improve performance.
func (e Encoding) Decode(dst io.Writer, data, tree io.Reader, root [32]byte) error {
	groupSize := e.groupSize()
	outboard := e.tree(data, tree)
	buf := make([]byte, groupSize)

	var err error

	read := func(r io.Reader, p []byte) []byte {
		if err == nil {
			err = readFull(r, p)
		}

		return p
	}
	write := func(w io.Writer, p []byte) {
		if err != nil {
			return
		}

		if err = writeFull(w, p); err != nil {
			err = fmt.Errorf("bao: write verified data: %w", err)
		}
	}
	readParent := func() (l, r [8]uint32) {
		read(outboard, buf[:parentSize])
		return bytesToCV(buf[:cvSize]), bytesToCV(buf[cvSize:])
	}

	var (
		counter uint64
		rec     func(cv [8]uint32, bufLen uint64, flags uint32) bool
	)

	rec = func(cv [8]uint32, bufLen uint64, flags uint32) bool {
		if err != nil {
			return false
		} else if bufLen <= groupSize {
			n := compressGroup(read(data, buf[:bufLen]), counter)
			counter += bufLen / kernel.ChunkSize
			n.Flags |= flags

			valid := cv == kernel.ChainingValue(n)
			if valid {
				write(dst, buf[:bufLen])
			}

			return valid
		}

		l, r := readParent()
		n := kernel.ParentNode(l, r, &kernel.IV, flags)
		mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

		return kernel.ChainingValue(n) == cv && rec(l, mid, 0) && rec(r, bufLen-mid, 0)
	}

	read(outboard, buf[:headerSize])
	dataLen := binary.LittleEndian.Uint64(buf[:headerSize])
	ok := rec(bytesToCV(root[:]), dataLen, kernel.FlagRoot)

	return verdict(ok, err)
}

// verdict is a walk's outcome as an error: the reader's or writer's own
// first, then ErrVerification when the content did not verify.
func verdict(ok bool, err error) error {
	switch {
	case err != nil:
		return err
	case !ok:
		return ErrVerification
	default:
		return nil
	}
}

type bufferAt struct {
	buf []byte
}

func (b *bufferAt) WriteAt(p []byte, off int64) (int, error) {
	if copy(b.buf[off:], p) != len(p) {
		panic("bad buffer size")
	}

	return len(p), nil
}

// EncodeBuf returns the encoding of data in memory, the whole of it or the
// outboard tree alone as the Encoding says, and the tree's root.
func (e Encoding) EncodeBuf(data []byte) ([]byte, [32]byte) {
	buf := bufferAt{buf: make([]byte, e.EncodedSize(int64(len(data))))}
	root, _ := e.Encode(&buf, bytes.NewReader(data), int64(len(data)))

	return buf.buf, root
}

// VerifyBuf verifies an encoding in memory against root: data holds the
// whole encoding, or the content with the tree in tree when the Encoding is
// outboard (tree is ignored otherwise, as in Decode). Content that does not
// verify, an encoding too short to, or bytes left over after it, in data or
// in an outboard tree, is ErrVerification.
func (e Encoding) VerifyBuf(data, tree []byte, root [32]byte) error {
	d, o := bytes.NewBuffer(data), bytes.NewBuffer(tree)

	// The readers are these buffers and the writer discards: what Decode
	// reports is the encoding's fault, whichever error says it.
	if err := e.Decode(io.Discard, d, o, root); err != nil {
		return ErrVerification
	}

	// tree is part of the encoding only when it is outboard: Decode never
	// reads it otherwise, and what it holds then is not bytes left over.
	if d.Len() != 0 || e.Outboard && o.Len() != 0 {
		return ErrVerification
	}

	return nil
}

// ExtractSlice writes to dst the slice encoding for length bytes of content
// from offset, read from an encoding: data and, when the Encoding is
// outboard, the tree from tree. It verifies nothing: an encoding that is
// corrupt makes a slice that is, with no error, and it is DecodeSlice that
// refuses the slice. A range past the length the encoding declares is
// ErrSliceLength, and nothing is written to dst.
//
// ExtractSlice reads the encoding from its start to the end of the slice,
// and no further. Of an outboard encoding that is the tree: data is read
// for the chunk groups in the slice and for no others, so data must start
// at offset rounded down to a whole group; for a zero length, at the group
// of the byte at offset, or of the last byte when offset is the end. Data
// that starts anywhere else, the whole content included when the slice
// starts past its first group, makes a slice that does not verify, and no
// error unless data then ends before the slice's groups are read, which is
// io.ErrUnexpectedEOF.
func (e Encoding) ExtractSlice(dst io.Writer, data, tree io.Reader, offset, length uint64) error {
	groupSize := e.groupSize()
	outboard := e.tree(data, tree)
	ext := &extractor{
		dst:              dst,
		data:             data,
		outboard:         outboard,
		buf:              make([]byte, groupSize),
		groupSize:        groupSize,
		offset:           offset,
		length:           length,
		combinedEncoding: !e.Outboard,
	}
	// The header is read, held to the request, and only then written: a
	// request that is refused leaves dst as it was.
	ext.transfer(outboard, headerSize, false)

	if ext.err != nil {
		return ext.err
	}

	dataLen := binary.LittleEndian.Uint64(ext.buf[:headerSize])
	if end := offset + length; end < offset || dataLen < end {
		return ErrSliceLength
	}

	ext.emit(headerSize)

	ext.offset, ext.length = sliceBounds(offset, length, dataLen)
	ext.extract(0, dataLen)

	return ext.err
}

// sliceBounds is the range of content a slice carries for a request. A slice
// holds at least one chunk (the Bao spec's "Slice Format"): a zero length is
// the chunk at offset, or the last one when offset is the end. The empty
// encoding is its own one empty chunk, reached whatever the range, and a
// request on it is left alone.
func sliceBounds(offset, length, dataLen uint64) (sliceOffset, sliceLength uint64) {
	if length != 0 || dataLen == 0 {
		return offset, length
	}

	if offset == dataLen {
		offset--
	}

	return offset, 1
}

// extractor is ExtractSlice's walk of the tree. Its first error stops the
// walk, and ExtractSlice returns it.
type extractor struct {
	dst              io.Writer
	data, outboard   io.Reader
	buf              []byte
	groupSize        uint64
	offset, length   uint64
	combinedEncoding bool
	err              error
}

// transfer reads n bytes from r, and writes them to dst when emit is set.
func (x *extractor) transfer(r io.Reader, n uint64, emit bool) {
	if x.err != nil {
		return
	}

	x.err = readFull(r, x.buf[:n])

	if emit {
		x.emit(n)
	}
}

// emit writes to dst the n bytes transfer read last.
func (x *extractor) emit(n uint64) {
	if x.err != nil {
		return
	}

	if err := writeFull(x.dst, x.buf[:n]); err != nil {
		x.err = fmt.Errorf("bao: write the slice: %w", err)
	}
}

func (x *extractor) extract(pos, bufLen uint64) {
	inSlice := pos < (x.offset+x.length) && x.offset < (pos+bufLen)
	// A node that starts at or past the slice's end is not in it, and neither
	// is any node after it in pre-order: the slice is whole, and what is
	// left of the encoding stays unread.
	if x.err != nil || pos >= x.offset+x.length {
		return
	} else if bufLen <= x.groupSize {
		if x.combinedEncoding || inSlice {
			x.transfer(x.data, bufLen, inSlice)
		}

		return
	}

	x.transfer(x.outboard, parentSize, inSlice)

	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)
	x.extract(pos, mid)
	x.extract(pos+mid, bufLen-mid)
}

// DecodeSlice reads from slice, which must hold a slice encoding for length
// bytes of content from offset (ExtractSlice), and streams the verified
// content to dst. A slice has one form, the tree interleaved, whatever the
// Encoding's Outboard says. Content that does not verify against root is
// ErrVerification.
//
// A range past the length the slice declares is ErrSliceLength. That length
// is the slice's header, which nothing has verified when the request is
// refused: the error says this slice cannot answer the request, not that
// the content is shorter than the request.
//
// DecodeSlice reads the tree 64 bytes at a time, so if the reader is
// unbuffered (e.g. os.File), wrapping it in a bufio.Reader will
// significantly improve performance.
func (e Encoding) DecodeSlice(dst io.Writer, slice io.Reader, offset, length uint64, root [32]byte) error {
	groupSize := e.groupSize()
	dec := &sliceDecoder{
		dst:       dst,
		data:      slice,
		buf:       make([]byte, groupSize),
		groupSize: groupSize,
		offset:    offset,
		length:    length,
	}

	dataLen := binary.LittleEndian.Uint64(dec.read(headerSize))
	if dec.err != nil {
		return dec.err
	}

	if end := offset + length; end < offset || dataLen < end {
		return ErrSliceLength
	}

	dec.sliceOffset, dec.sliceLength = sliceBounds(offset, length, dataLen)
	ok := dec.decode(bytesToCV(root[:]), 0, dataLen, kernel.FlagRoot)

	return verdict(ok, dec.err)
}

// sliceDecoder is DecodeSlice's walk of the tree. Its first error stops the
// walk, and DecodeSlice returns it.
type sliceDecoder struct {
	dst       io.Writer
	data      io.Reader
	buf       []byte
	groupSize uint64
	// offset and length are what the caller asked for and what is written;
	// sliceOffset and sliceLength are what the slice carries (sliceBounds),
	// which decides the nodes read and verified.
	offset, length           uint64
	sliceOffset, sliceLength uint64
	err                      error
}

func (d *sliceDecoder) read(n uint64) []byte {
	if d.err == nil {
		d.err = readFull(d.data, d.buf[:n])
	}

	return d.buf[:n]
}

func (d *sliceDecoder) readParent() (l, r [8]uint32) {
	d.read(parentSize)
	return bytesToCV(d.buf[:cvSize]), bytesToCV(d.buf[cvSize:])
}

func (d *sliceDecoder) write(p []byte) {
	if d.err != nil {
		return
	}

	if err := writeFull(d.dst, p); err != nil {
		d.err = fmt.Errorf("bao: write verified data: %w", err)
	}
}

func (d *sliceDecoder) decode(cv [8]uint32, pos, bufLen uint64, flags uint32) bool {
	inSlice := pos < (d.sliceOffset+d.sliceLength) && d.sliceOffset < (pos+bufLen)
	if d.err != nil {
		return false
	} else if bufLen <= d.groupSize {
		return d.decodeGroup(cv, pos, bufLen, flags, inSlice)
	}

	if !inSlice {
		return true
	}

	l, r := d.readParent()
	n := kernel.ParentNode(l, r, &kernel.IV, flags)
	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

	return kernel.ChainingValue(n) == cv && d.decode(l, pos, mid, 0) && d.decode(r, pos+mid, bufLen-mid, 0)
}

// decodeGroup verifies the chunk group of bufLen bytes at pos against cv, and
// writes the part of it inside the slice.
func (d *sliceDecoder) decodeGroup(cv [8]uint32, pos, bufLen uint64, flags uint32, inSlice bool) bool {
	if bufLen == 0 {
		// the tree for empty data is a single empty group; there is
		// no data to decode, but we can still verify the root
		n := compressGroup(nil, 0)
		n.Flags |= flags

		return cv == kernel.ChainingValue(n)
	}

	if !inSlice {
		return true
	}

	n := compressGroup(d.read(bufLen), pos/kernel.ChunkSize)
	n.Flags |= flags

	valid := cv == kernel.ChainingValue(n)
	if valid {
		// only write within range
		p := d.buf[:bufLen]
		if pos+bufLen > d.offset+d.length {
			p = p[:d.offset+d.length-pos]
		}

		if pos < d.offset {
			p = p[d.offset-pos:]
		}

		d.write(p)
	}

	return valid
}

// VerifySlice verifies a slice encoding in memory for length bytes of
// content from offset, and returns the verified content, whatever the
// Encoding's Outboard says: a slice has one form. A slice that does not
// verify against root, is too short to, or has bytes left over, is
// ErrVerification. A range past the length the slice declares is
// ErrSliceLength, on a length nothing has verified, as in DecodeSlice: the
// slice cannot answer the request, which is not to say that the content is
// shorter.
func (e Encoding) VerifySlice(slice []byte, offset, length uint64, root [32]byte) ([]byte, error) {
	d := bytes.NewBuffer(slice)

	var buf bytes.Buffer
	if err := e.DecodeSlice(&buf, d, offset, length, root); err != nil {
		if errors.Is(err, ErrSliceLength) {
			return nil, err
		}

		return nil, ErrVerification
	}

	if d.Len() > 0 {
		return nil, ErrVerification
	}

	return buf.Bytes(), nil
}

// VerifyChunk verifies chunks, whole chunk groups of content starting at
// offset, against root with the full outboard tree in tree, whatever the
// Encoding's Outboard says: the tree is the same. A tree of another length
// than the content calls for, chunks for a range past the content, or no
// chunks over content that has some, is ErrVerification; the encoding of
// empty data has one empty chunk, which no chunks verify.
func (e Encoding) VerifyChunk(chunks, tree []byte, offset uint64, root [32]byte) error {
	ver := &chunkVerifier{
		chunks:    bytes.NewBuffer(chunks),
		outboard:  bytes.NewBuffer(tree),
		groupSize: e.groupSize(),
		offset:    offset,
		length:    uint64(len(chunks)),
	}
	if ver.outboard.Len() < headerSize {
		return ErrVerification
	}

	dataLen := binary.LittleEndian.Uint64(ver.outboard.Next(headerSize))
	// #nosec G115 -- the buffer's length, never negative
	if end := offset + ver.length; end < offset || dataLen < end ||
		uint64(ver.outboard.Len()) != parentSize*ver.nodesWithin(dataLen) {
		return ErrVerification
	}

	if ver.length == 0 && dataLen > 0 {
		return ErrVerification
	}

	return verdict(ver.verify(bytesToCV(root[:]), 0, dataLen, kernel.FlagRoot), nil)
}

// chunkVerifier is VerifyChunk's walk of the tree.
type chunkVerifier struct {
	chunks, outboard *bytes.Buffer
	groupSize        uint64
	offset, length   uint64
}

// nodesWithin is the number of parent nodes in a subtree of bufLen bytes.
// bufLen comes from the encoding's header, so the arithmetic stays in
// uint64: at most 2^54 nodes, 2^60 bytes of them.
func (v *chunkVerifier) nodesWithin(bufLen uint64) uint64 {
	if bufLen <= v.groupSize {
		return 0 // leaf
	}

	n := bufLen / v.groupSize
	if bufLen%v.groupSize == 0 {
		n--
	}

	return n
}

func (v *chunkVerifier) verify(cv [8]uint32, pos, bufLen uint64, flags uint32) bool {
	inSlice := pos < (v.offset+v.length) && v.offset < (pos+bufLen)
	if bufLen <= v.groupSize {
		return v.verifyGroup(cv, pos, bufLen, flags, inSlice)
	}

	if !inSlice {
		// A subtree's nodes are among the whole tree's, whose byte count
		// was matched to the outboard's length at entry, so this fits.
		_ = v.outboard.Next(int(parentSize * v.nodesWithin(bufLen))) // #nosec G115 -- bounded by the outboard's length

		return true
	}

	l, r := bytesToCV(v.outboard.Next(cvSize)), bytesToCV(v.outboard.Next(cvSize))
	n := kernel.ParentNode(l, r, &kernel.IV, flags)
	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

	return kernel.ChainingValue(n) == cv && v.verify(l, pos, mid, 0) && v.verify(r, pos+mid, bufLen-mid, 0)
}

// verifyGroup verifies the chunk group of bufLen bytes at pos against cv.
func (v *chunkVerifier) verifyGroup(cv [8]uint32, pos, bufLen uint64, flags uint32, inSlice bool) bool {
	if bufLen == 0 {
		// the tree for empty data is a single empty group; there are
		// no chunks to verify, but we can still verify the root
		n := compressGroup(nil, 0)
		n.Flags |= flags

		return cv == kernel.ChainingValue(n)
	}

	if !inSlice {
		return true
	}

	// #nosec G115 -- groupSize is ChunkSize shifted by a checked group: at most 1 GiB, which fits an int
	n := compressGroup(v.chunks.Next(int(v.groupSize)), pos/kernel.ChunkSize)
	n.Flags |= flags

	return cv == kernel.ChainingValue(n)
}

// groupSize is the Encoding's leaf in bytes. A Group outside [0, MaxGroup]
// is a programming error, not a data error, and panics here, at the first
// use: left unchecked, the shifts and allocations below fail with unrelated
// runtime panics.
func (e Encoding) groupSize() uint64 {
	if e.Group < 0 || e.Group > MaxGroup {
		panic("bao: Encoding.Group must be between 0 and MaxGroup")
	}

	return uint64(kernel.ChunkSize) << e.Group
}

// tree is the reader the parent nodes come from: the outboard one, or the
// data itself when the tree is interleaved. An outboard Encoding given no
// tree is a programming error.
func (e Encoding) tree(data, tree io.Reader) io.Reader {
	if !e.Outboard {
		return data
	}

	if tree == nil {
		panic("bao: an outboard Encoding needs its tree")
	}

	return tree
}
