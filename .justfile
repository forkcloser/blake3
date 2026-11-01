# This file is the project's own — add recipes below. Keep the import: it
# mounts every shared limen task under `just do ...`.
import '.limen/just/main.just'

# The FIRST recipe defined here becomes `just`'s default.
lint: do::lint::go::default do::lint::go::deadcode do::lint::default lint-generated
fix: do::fix::go::default do::fix::default
test: simd-info do::test::go::unit do::test::go::race
# The security workflow runs `just security`.
security: do::security::default
bench: do::perf::go::bench

# avo lives in its own module (avo/go.mod) so it never enters the main
# module's graph — do not fold it into the root go.mod.
[doc('Verify guts/compress_amd64.s is the exact output of avo/gen.go')]
lint-generated:
    #!/usr/bin/env bash
    set -euo pipefail
    # avo embeds its -out argument in the generated header, so a byte-exact
    # comparison must regenerate with the exact command `just gen` runs.
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

# A green run on a host without AVX-512 says nothing about the AVX-512
# kernels; this names which paths ran. Diagnostic only — never fails.
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

# The AVX-512 kernels run only on an amd64 host with avx512f, which GitHub's
# ubuntu runners have on some runs and not others. Intel SDE executes a binary
# under an emulated CPU, so the guts tests run here under Sapphire Rapids on
# any x86-64 Linux host. BLAKE3_REQUIRE_ISA=avx512 makes TestRequiredISA fail
# unless the dispatcher selected the AVX-512 kernels, so a pass means those
# kernels, not a fallback, produced the specification's results.
# -force_emulate skx: SDE runs natively whatever the host supports, and on a
# host with AVX-512 the kernels then compute wrong trees at random, their
# vector state lost under Pin across Go's preemption signals. Forced
# emulation of the Skylake-X set, which holds every instruction the kernels
# use (AVX-512F), keeps that state in the emulator on every host (the book's
# upstream page). Pin, which SDE is built on, needs ptrace of its own child:
# the workflow sets kernel.yama.ptrace_scope=0 before this runs.
[doc('Run the guts tests under Intel SDE emulating an AVX-512 CPU (linux/amd64 only)')]
test-avx512-emulated:
    #!/usr/bin/env bash
    set -euo pipefail
    host="$(go env GOHOSTOS)/$(go env GOHOSTARCH)"
    if [ "$host" != "linux/amd64" ]; then
        echo "test-avx512-emulated: Intel SDE runs on linux/amd64 only; this host is $host" >&2
        exit 1
    fi
    if [ "$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)" != "0" ]; then
        echo "test-avx512-emulated: kernel.yama.ptrace_scope must be 0 for SDE to run its child" >&2
        exit 1
    fi
    url=$(limen pins get intel-sde url)
    sha256=$(limen pins get intel-sde sha256)
    mkdir -p build/sde
    kit=build/sde/sde.tar.xz
    if [ ! -f "$kit" ] || ! echo "$sha256  $kit" | sha256sum -c --status; then
        curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-delay 3 -o "$kit" "$url"
        echo "$sha256  $kit" | sha256sum -c --status
    fi
    rm -rf build/sde/kit
    mkdir -p build/sde/kit
    tar -xJf "$kit" -C build/sde/kit --strip-components=1
    go test -c -o build/sde/guts.test ./guts
    BLAKE3_REQUIRE_ISA=avx512 build/sde/kit/sde64 -spr -force_emulate skx -- build/sde/guts.test -test.v | tee build/sde/test.log
    grep -q '^--- PASS: TestRequiredISA' build/sde/test.log || {
        echo "test-avx512-emulated: the tests passed but TestRequiredISA did not run" >&2
        exit 1
    }
    echo "test-avx512-emulated: guts tests passed under SDE on the AVX-512 kernels"
