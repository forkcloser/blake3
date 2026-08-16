# This file is the project's own.
# Add recipes leveraging provided `do` ready-made recipes, or create your own.
# The import must be kept: it mounts every shared limen task under `just do ...`.
import '.limen/just/main.just'

# The FIRST recipe defined here becomes `just`'s default.
lint: do::lint::go::default do::lint::default lint-generated
fix: do::fix::go::default do::fix::default
test: simd-info do::test::go::unit do::test::go::race
bench: do::test::go::bench

# The amd64 assembly is generated — never edited — and this proves it: the
# committed file must be the exact output of the pinned generator. avo is a
# code-generation dependency of the generator alone, so it lives in its own
# module (avo/go.mod, GOSUMDB-verified like any other) and never appears in
# the main module's graph.
[doc('Verify guts/compress_amd64.s is the exact output of avo/gen.go')]
lint-generated:
    #!/usr/bin/env bash
    set -euo pipefail
    # avo embeds its -out argument in the generated header, so a byte-exact
    # comparison must regenerate with the exact command `just gen` runs.
    # Snapshot and restore: the lint observes, never mutates (same doctrine as
    # `do lint aqua`).
    tmp=$(mktemp "${TMPDIR:-/tmp}/blake3-avo.XXXXXX")
    # shellcheck disable=SC2329 # invoked via trap, not dead code
    restore() {
        cp "$tmp" guts/compress_amd64.s
        rm -f "$tmp"
    }
    trap restore EXIT
    cp guts/compress_amd64.s "$tmp"
    go generate ./guts
    if ! cmp -s "$tmp" guts/compress_amd64.s; then
        echo "guts/compress_amd64.s does not match the output of avo/gen.go — run 'just gen'" >&2
        exit 1
    fi

[doc('Regenerate guts/compress_amd64.s from avo/gen.go')]
gen:
    go generate ./guts

# The SIMD kernels are selected by runtime CPU detection, so which
# implementation the suite just exercised is a property of the host. Say so in
# the log: a green run on a runner without AVX-512 is not evidence about the
# AVX-512 code. Diagnostic only — never fails.
[doc('Report which BLAKE3 SIMD paths this host can exercise')]
simd-info:
    #!/usr/bin/env bash
    set -euo pipefail
    arch="$(go env GOHOSTARCH)"
    echo "host: $(go env GOHOSTOS)/$arch"
    if [ "$arch" != "amd64" ]; then
        echo "simd: none (the assembly is amd64-only; this host runs the generic implementation)"
        exit 0
    fi
    if [ -r /proc/cpuinfo ]; then
        for feature in avx2 avx512f; do
            if grep -qw "$feature" /proc/cpuinfo; then
                echo "simd: $feature available"
            else
                echo "simd: $feature NOT available"
            fi
        done
    elif command -v sysctl > /dev/null 2>&1; then
        for feature in hw.optional.avx2_0 hw.optional.avx512f; do
            if [ "$(sysctl -n "$feature" 2> /dev/null)" = "1" ]; then
                echo "simd: $feature available"
            else
                echo "simd: $feature NOT available"
            fi
        done
    else
        echo "simd: unknown (no /proc/cpuinfo or sysctl on this host)"
    fi
