#!/usr/bin/env bash

# bench/ab.sh reports delta=100*(on-off)/off: speedups are NEGATIVE.
# Therefore 1.05x requires delta<=-4.76%, and 1.12x requires delta<=-10.71%.

coli_v4_assert_binary() {
    local root binary expected actual
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || return 2
    binary="$root/c/deepseek_v4"

    if [[ $# -ne 1 ]]; then
        printf 'usage: %s EXPECTED_MD5\n' "${BASH_SOURCE[0]}" >&2
        return 2
    fi
    expected=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    if [[ ! $expected =~ ^[0-9a-f]{32}$ ]]; then
        printf 'invalid expected md5: %s\n' "$1" >&2
        return 2
    fi
    if [[ ! -x $binary ]]; then
        printf 'binary not executable: %s\n' "$binary" >&2
        return 2
    fi

    if command -v md5 >/dev/null 2>&1; then
        actual=$(md5 -q "$binary") || return 2
    else
        actual=$(md5sum "$binary" | cut -d ' ' -f 1) || return 2
    fi

    if [[ $actual != "$expected" ]]; then
        printf 'FAIL binary expected=%s actual=%s path=%s\n' \
            "$expected" "$actual" "$binary" >&2
        return 1
    fi
    printf 'PASS binary md5=%s path=%s\n' "$actual" "$binary"
}

coli_v4_assert_binary "$@"
