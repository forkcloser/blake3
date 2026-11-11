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

# The fork's own amd64 figures (README, "Benchmarks"): the three paths on one
# host that has AVX-512, which GitHub's ubuntu runners do on some runs and
# not others. The bench workflow runs this on dispatch and stops, naming the
# host, when the runner lacks avx512f: dispatch again. Emulation says nothing
# about speed, so this is the one lane that needs the real hardware. The
# paths are selected through GODEBUG, which golang.org/x/sys/cpu honours:
# cpu.avx512f=off leaves AVX2, cpu.avx2=off as well leaves the pure Go.
[doc("Benchmark the root package on the AVX-512, AVX2 and pure-Go paths (linux/amd64 with avx512f only)")]
bench-amd64 count="6":
    #!/usr/bin/env bash
    set -euo pipefail
    host="$(go env GOHOSTOS)/$(go env GOHOSTARCH)"
    if [ "$host" != "linux/amd64" ]; then
        echo "bench-amd64: linux/amd64 only; this host is $host" >&2
        exit 1
    fi
    model=$(grep -m1 'model name' /proc/cpuinfo | sed 's/^[^:]*: *//')
    if ! grep -qw avx512f /proc/cpuinfo; then
        echo "bench-amd64: this host ($model) has no avx512f; dispatch again for another runner" >&2
        exit 1
    fi
    out="${LIMEN_BUILD:-.build}/bench"
    mkdir -p "$out"
    {
        echo "host: $model"
        echo "go: $(go version)"
        echo "commit: $(git rev-parse HEAD)"
        echo "date: $(date -u +%Y-%m-%d)"
        echo "count: {{ count }}"
    } | tee "$out/host.txt"
    for path in avx512 avx2 generic; do
        case $path in
            avx512) debug="" ;;
            avx2) debug="cpu.avx512f=off" ;;
            generic) debug="cpu.avx512f=off,cpu.avx2=off" ;;
        esac
        echo "▶ bench-amd64: $path (GODEBUG=$debug)"
        GODEBUG="$debug" go test -run '^$' -bench . -benchmem -count "{{ count }}" . | tee "$out/$path.txt"
    done

# One amd64 dispatch arm, required. The verify legs run whatever kernels their
# host has and assert nothing about which arm ran; here GODEBUG picks it
# (golang.org/x/sys/cpu honours cpu.<feature>=off) and BLAKE3_REQUIRE_ISA makes
# TestRequiredISA fail unless that arm was taken. avx2 turns avx512f off, so an
# AVX-512 host still runs the AVX2 kernels; generic turns both off, so the
# amd64 fallback arms run. avx512 is test-avx512-emulated's: no hosted runner
# guarantees it. The output is captured verbosely because a passing test
# leaves no trace otherwise, and the pass is accepted only if TestRequiredISA
# is among them.
[doc('Run the suite with one amd64 dispatch arm required: avx2 or generic')]
test-isa isa:
    #!/usr/bin/env bash
    set -euo pipefail
    arch="$(go env GOHOSTARCH)"
    if [ "$arch" != "amd64" ]; then
        echo "test-isa: the dispatch arms exist on amd64 only; this host is $arch" >&2
        exit 1
    fi
    case "{{ isa }}" in
        avx2) debug="cpu.avx512f=off" ;;
        generic) debug="cpu.avx512f=off,cpu.avx2=off" ;;
        *)
            echo "test-isa: the arm is avx2 or generic, not {{ isa }}" >&2
            exit 1
            ;;
    esac
    out="${LIMEN_BUILD:-.build}/isa"
    mkdir -p "$out"
    GODEBUG="$debug" BLAKE3_REQUIRE_ISA="{{ isa }}" GOTESTSUM_FORMAT=standard-verbose just do test go unit | tee "$out/{{ isa }}.log"
    grep -q '^--- PASS: TestRequiredISA' "$out/{{ isa }}.log" || {
        echo "test-isa: the tests passed but TestRequiredISA did not run" >&2
        exit 1
    }
    echo "test-isa: the suite passed with the {{ isa }} arm required"

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
