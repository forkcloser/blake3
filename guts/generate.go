package guts

// The directive lives here, in a file with no build constraints, rather than
// in compress_amd64.go: `go generate` only scans files that match the host's
// build context, so a directive in an amd64-only file would silently not run
// on other machines.
//go:generate go run -C ../avo . -out ../guts/compress_amd64.s
