// Package bao implements BLAKE3 verified streaming: encodings that carry the
// hash tree alongside (or, "outboard", apart from) the data, so that any
// prefix or slice of the data can be verified against the root hash as it is
// read. See https://github.com/oconnor663/bao for the format.
//
// Every function takes a group parameter that sets the chunk-group size as a
// power of two: a group is guts.ChunkSize << group bytes, and standard Bao
// uses 0. The parameter is validated at every entry point; a group outside
// [0, MaxGroup] panics with a message that says so.
package bao

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"

	"github.com/forkcloser/blake3/guts"
)

// MaxGroup is the largest group any function in this package accepts. It
// is a 1 GiB chunk group, the largest whose size in bytes fits an int on
// every platform.
const MaxGroup = 20

var (
	// errNegativeLength is Encode's refusal of a negative dataLen.
	errNegativeLength = errors.New("bao: negative data length")
	// errSliceLength is a slice that does not fit the encoded data.
	errSliceLength = errors.New("invalid slice length")
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

// checkGroup panics if group is outside [0, MaxGroup]. An out-of-range group
// is a programming error, not a data error: left unchecked, the shifts and
// allocations below fail with unrelated runtime panics.
func checkGroup(group int) {
	if group < 0 || group > MaxGroup {
		panic("bao: group must be between 0 and MaxGroup")
	}
}

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

func compressGroup(p []byte, counter uint64) guts.Node {
	// stack size is log2(maximum number of buffers in a group), i.e.
	// log2(2^64 bytes / ChunkSize / MaxSIMD) = 64 - 10 - 4
	var (
		stack [50][8]uint32
		sc    uint64
	)

	pushSubtree := func(cv [8]uint32) {
		i := 0
		for sc&(1<<i) != 0 {
			cv = guts.ChainingValue(guts.ParentNode(stack[i], cv, &guts.IV, 0))
			i++
		}

		stack[i] = cv
		sc++
	}

	var (
		buf    [guts.MaxSIMD * guts.ChunkSize]byte
		buflen int
	)
	for len(p) > 0 {
		if buflen == len(buf) {
			pushSubtree(guts.ChainingValue(guts.CompressBuffer(&buf, buflen, &guts.IV, counter+(sc*guts.MaxSIMD), 0)))
			buflen = 0
		}

		n := copy(buf[buflen:], p)
		buflen += n
		p = p[n:]
	}

	n := guts.CompressBuffer(&buf, buflen, &guts.IV, counter+(sc*guts.MaxSIMD), 0)
	for i := bits.TrailingZeros64(sc); i < bits.Len64(sc); i++ {
		if sc&(1<<i) != 0 {
			n = guts.ParentNode(stack[i], guts.ChainingValue(n), &guts.IV, 0)
		}
	}

	return n
}

// EncodedSize returns the size of a Bao encoding for the provided quantity
// of data. It panics if dataLen is negative.
func EncodedSize(dataLen, group int, outboard bool) int {
	checkGroup(group)

	if dataLen < 0 {
		panic("bao: negative data length")
	}

	groupSize := guts.ChunkSize << group
	size := headerSize

	if dataLen > 0 {
		chunks := (dataLen + groupSize - 1) / groupSize
		cvs := 2*chunks - 2 // no I will not elaborate
		size += cvs * cvSize
	}

	if !outboard {
		size += dataLen
	}

	return size
}

// Encode computes the intermediate BLAKE3 tree hashes of data and writes them
// to dst. If outboard is false, the contents of data are also written to dst,
// interleaved with the tree hashes. It also returns the tree root, i.e. the
// 256-bit BLAKE3 hash. The group parameter controls how many chunks are hashed
// per "group," as a power of 2; for standard Bao, use 0.
//
// Note that dst is not written sequentially, and therefore must be initialized
// with sufficient capacity to hold the encoding; see EncodedSize. A negative
// dataLen is an error, reported before anything is written.
func Encode(dst io.WriterAt, data io.Reader, dataLen int64, group int, outboard bool) ([32]byte, error) {
	checkGroup(group)

	if dataLen < 0 {
		return [32]byte{}, errNegativeLength
	}

	groupSize := uint64(guts.ChunkSize << group)
	enc := &encoder{
		dst:       dst,
		data:      data,
		buf:       make([]byte, groupSize),
		groupSize: groupSize,
		outboard:  outboard,
	}

	// NOTE: unlike the reference implementation, we write directly in
	// pre-order, rather than writing in post-order and then flipping. This cuts
	// the I/O required in half, at the cost of making it a lot trickier to hash
	// multiple groups in SIMD. However, you can still get the SIMD speedup if
	// group > 0, so maybe just do that.
	binary.LittleEndian.PutUint64(enc.buf[:headerSize], uint64(dataLen))
	enc.write(enc.buf[:headerSize], 0)
	_, root := enc.encode(uint64(dataLen), guts.FlagRoot, headerSize)

	return *cvToBytes(&root), enc.err
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

func (e *encoder) read(p []byte) []byte {
	if e.err == nil {
		_, e.err = io.ReadFull(e.data, p)
	}

	return p
}

func (e *encoder) write(p []byte, off uint64) {
	if e.err != nil {
		return
	}

	// #nosec G115 -- an offset inside the encoding of an int64-length input:
	// past 2^63 only for exabyte inputs, where WriteAt refuses it.
	if _, err := e.dst.WriteAt(p, int64(off)); err != nil {
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
		e.counter += bufLen / guts.ChunkSize
		n.Flags |= flags

		return 0, guts.ChainingValue(n)
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

	return 2 + lchildren + rchildren, guts.ChainingValue(guts.ParentNode(l, r, &guts.IV, flags))
}

// Decode reads content and tree data from the provided reader(s), and
// streams the verified content to dst. It returns false if verification fails.
// If the content and tree data are interleaved, outboard should be nil.
//
// Decode reads the tree data 64 bytes at a time, so if the readers are
// unbuffered (e.g. os.File), wrapping them in a bufio.Reader will
// significantly improve performance.
func Decode(dst io.Writer, data, outboard io.Reader, group int, root [32]byte) (bool, error) {
	checkGroup(group)

	if outboard == nil {
		outboard = data
	}

	groupSize := uint64(guts.ChunkSize << group)
	buf := make([]byte, groupSize)

	var err error

	read := func(r io.Reader, p []byte) []byte {
		if err == nil {
			_, err = io.ReadFull(r, p)
		}

		return p
	}
	write := func(w io.Writer, p []byte) {
		if err != nil {
			return
		}

		if _, err = w.Write(p); err != nil {
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
			counter += bufLen / guts.ChunkSize
			n.Flags |= flags

			valid := cv == guts.ChainingValue(n)
			if valid {
				write(dst, buf[:bufLen])
			}

			return valid
		}

		l, r := readParent()
		n := guts.ParentNode(l, r, &guts.IV, flags)
		mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

		return guts.ChainingValue(n) == cv && rec(l, mid, 0) && rec(r, bufLen-mid, 0)
	}

	read(outboard, buf[:headerSize])
	dataLen := binary.LittleEndian.Uint64(buf[:headerSize])
	ok := rec(bytesToCV(root[:]), dataLen, guts.FlagRoot)

	return ok, err
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

// EncodeBuf returns the Bao encoding and root (i.e. BLAKE3 hash) for data.
func EncodeBuf(data []byte, group int, outboard bool) ([]byte, [32]byte) {
	checkGroup(group)
	buf := bufferAt{buf: make([]byte, EncodedSize(len(data), group, outboard))}
	root, _ := Encode(&buf, bytes.NewReader(data), int64(len(data)), group, outboard)

	return buf.buf, root
}

// VerifyBuf verifies the Bao encoding and root (i.e. BLAKE3 hash) for data.
// If the content and tree data are interleaved, outboard should be nil.
func VerifyBuf(data, outboard []byte, group int, root [32]byte) bool {
	checkGroup(group)

	d, o := bytes.NewBuffer(data), bytes.NewBuffer(outboard)

	var or io.Reader = o
	if outboard == nil {
		or = nil
	}

	ok, _ := Decode(io.Discard, d, or, group, root)

	return ok && d.Len() == 0 && o.Len() == 0 // check for trailing data
}

// ExtractSlice returns the slice encoding for the given offset and length. When
// extracting from an outboard encoding, data should contain only the chunk
// groups that will be present in the slice.
func ExtractSlice(dst io.Writer, data, outboard io.Reader, group int, offset, length uint64) error {
	checkGroup(group)

	combinedEncoding := outboard == nil
	if combinedEncoding {
		outboard = data
	}

	groupSize := uint64(guts.ChunkSize << group)
	ext := &extractor{
		dst:              dst,
		data:             data,
		outboard:         outboard,
		buf:              make([]byte, groupSize),
		groupSize:        groupSize,
		offset:           offset,
		length:           length,
		combinedEncoding: combinedEncoding,
	}
	ext.transfer(outboard, headerSize, true)

	dataLen := binary.LittleEndian.Uint64(ext.buf[:headerSize])
	if end := offset + length; end < offset || dataLen < end {
		return errSliceLength
	}

	ext.extract(0, dataLen)

	return ext.err
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

	_, x.err = io.ReadFull(r, x.buf[:n])
	if x.err != nil || !emit {
		return
	}

	if _, err := x.dst.Write(x.buf[:n]); err != nil {
		x.err = fmt.Errorf("bao: write the slice: %w", err)
	}
}

func (x *extractor) extract(pos, bufLen uint64) {
	inSlice := pos < (x.offset+x.length) && x.offset < (pos+bufLen)
	if x.err != nil {
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

// DecodeSlice reads from data, which must contain a slice encoding for the
// given offset and length, and streams verified content to dst. It returns
// false if verification fails.
//
// DecodeSlice reads the tree data 64 bytes at a time, so if the reader is
// unbuffered (e.g. os.File), wrapping it in a bufio.Reader will significantly
// improve performance.
func DecodeSlice(dst io.Writer, data io.Reader, group int, offset, length uint64, root [32]byte) (bool, error) {
	checkGroup(group)
	groupSize := uint64(guts.ChunkSize << group)
	dec := &sliceDecoder{
		dst:       dst,
		data:      data,
		buf:       make([]byte, groupSize),
		groupSize: groupSize,
		offset:    offset,
		length:    length,
	}

	dataLen := binary.LittleEndian.Uint64(dec.read(headerSize))
	if end := offset + length; end < offset || dataLen < end {
		return false, errSliceLength
	}

	ok := dec.decode(bytesToCV(root[:]), 0, dataLen, guts.FlagRoot)

	return ok, dec.err
}

// sliceDecoder is DecodeSlice's walk of the tree. Its first error stops the
// walk, and DecodeSlice returns it.
type sliceDecoder struct {
	dst            io.Writer
	data           io.Reader
	buf            []byte
	groupSize      uint64
	offset, length uint64
	err            error
}

func (d *sliceDecoder) read(n uint64) []byte {
	if d.err == nil {
		_, d.err = io.ReadFull(d.data, d.buf[:n])
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

	if _, err := d.dst.Write(p); err != nil {
		d.err = fmt.Errorf("bao: write verified data: %w", err)
	}
}

func (d *sliceDecoder) decode(cv [8]uint32, pos, bufLen uint64, flags uint32) bool {
	inSlice := pos < (d.offset+d.length) && d.offset < (pos+bufLen)
	if d.err != nil {
		return false
	} else if bufLen <= d.groupSize {
		return d.decodeGroup(cv, pos, bufLen, flags, inSlice)
	}

	if !inSlice {
		return true
	}

	l, r := d.readParent()
	n := guts.ParentNode(l, r, &guts.IV, flags)
	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

	return guts.ChainingValue(n) == cv && d.decode(l, pos, mid, 0) && d.decode(r, pos+mid, bufLen-mid, 0)
}

// decodeGroup verifies the chunk group of bufLen bytes at pos against cv, and
// writes the part of it inside the slice.
func (d *sliceDecoder) decodeGroup(cv [8]uint32, pos, bufLen uint64, flags uint32, inSlice bool) bool {
	if bufLen == 0 {
		// the tree for empty data is a single empty group; there is
		// no data to decode, but we can still verify the root
		n := compressGroup(nil, 0)
		n.Flags |= flags

		return cv == guts.ChainingValue(n)
	}

	if !inSlice {
		return true
	}

	n := compressGroup(d.read(bufLen), pos/guts.ChunkSize)
	n.Flags |= flags

	valid := cv == guts.ChainingValue(n)
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

// VerifySlice verifies the Bao slice encoding in data, returning the
// verified bytes.
func VerifySlice(data []byte, group int, offset, length uint64, root [32]byte) ([]byte, bool) {
	checkGroup(group)

	d := bytes.NewBuffer(data)

	var buf bytes.Buffer
	if ok, _ := DecodeSlice(&buf, d, group, offset, length, root); !ok || d.Len() > 0 {
		return nil, false
	}

	return buf.Bytes(), true
}

// VerifyChunk verifies the provided chunks with a full outboard encoding.
func VerifyChunk(chunks, outboard []byte, group int, offset uint64, root [32]byte) bool {
	checkGroup(group)

	ver := &chunkVerifier{
		chunks:    bytes.NewBuffer(chunks),
		outboard:  bytes.NewBuffer(outboard),
		groupSize: uint64(guts.ChunkSize << group),
		offset:    offset,
		length:    uint64(len(chunks)),
	}
	if ver.outboard.Len() < headerSize {
		return false
	}

	dataLen := binary.LittleEndian.Uint64(ver.outboard.Next(headerSize))
	if end := offset + ver.length; end < offset || dataLen < end ||
		ver.outboard.Len() != parentSize*ver.nodesWithin(dataLen) {
		return false
	}

	return ver.verify(bytesToCV(root[:]), 0, dataLen, guts.FlagRoot)
}

// chunkVerifier is VerifyChunk's walk of the tree.
type chunkVerifier struct {
	chunks, outboard *bytes.Buffer
	groupSize        uint64
	offset, length   uint64
}

// nodesWithin is the number of parent nodes in a subtree of bufLen bytes.
func (v *chunkVerifier) nodesWithin(bufLen uint64) int {
	if bufLen <= v.groupSize {
		return 0 // leaf
	}

	n := int(bufLen / v.groupSize) // #nosec G115 -- bufLen is an in-memory slice's length, so the quotient fits an int
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
		_ = v.outboard.Next(parentSize * v.nodesWithin(bufLen)) // skip
		return true
	}

	l, r := bytesToCV(v.outboard.Next(cvSize)), bytesToCV(v.outboard.Next(cvSize))
	n := guts.ParentNode(l, r, &guts.IV, flags)
	mid := uint64(1) << (bits.Len64(bufLen-1) - 1)

	return guts.ChainingValue(n) == cv && v.verify(l, pos, mid, 0) && v.verify(r, pos+mid, bufLen-mid, 0)
}

// verifyGroup verifies the chunk group of bufLen bytes at pos against cv.
func (v *chunkVerifier) verifyGroup(cv [8]uint32, pos, bufLen uint64, flags uint32, inSlice bool) bool {
	if bufLen == 0 {
		// the tree for empty data is a single empty group; there are
		// no chunks to verify, but we can still verify the root
		n := compressGroup(nil, 0)
		n.Flags |= flags

		return cv == guts.ChainingValue(n)
	}

	if !inSlice {
		return true
	}

	// #nosec G115 -- groupSize is ChunkSize shifted by a checked group, a few megabytes at most
	n := compressGroup(v.chunks.Next(int(v.groupSize)), pos/guts.ChunkSize)
	n.Flags |= flags

	return cv == guts.ChainingValue(n)
}
