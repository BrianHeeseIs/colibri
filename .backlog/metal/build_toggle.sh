#!/usr/bin/env bash
# Build BOTH engine variants from a genuinely clean object state.
#
# WHY THE rm: `make clean` does NOT remove c/*.o, and make does NOT recompile the
# 26 C translation units when only a -D flag (METAL=1) changes. Switching METAL
# therefore relinks the Metal object while every #ifdef COLI_V4_METAL_SEAM call
# site stays compiled OUT -- you get Metal symbols in the binary and zero Metal
# calls at runtime. That trap produced a wrong conclusion twice (see E42/E43).
#
# Produces two named binaries so benchmarks never rebuild mid-run:
#   c/deepseek_v4.cpu    METAL=0  (default/production)
#   c/deepseek_v4.metal  METAL=1  (GPU seam compiled in)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$here"

build() { # $1=metalflag  $2=outname
  rm -f c/*.o c/deepseek_v4
  rm -rf c/build/metal-v4
  ( cd c && make -f Makefile.deepseek-v4 ${1:+METAL=1} >/dev/null 2>&1 )
  mv c/deepseek_v4 "c/$2"
  printf "  %-22s metal_syms=%-3s objects=%s\n" "$2" \
    "$(nm "c/$2" 2>/dev/null | grep -ci metal || echo 0)" \
    "$(ls c/*.o 2>/dev/null | wc -l | tr -d ' ')"
}
echo "building both variants from clean object state..."
build ""  deepseek_v4.cpu
build "1" deepseek_v4.metal
# leave a working default binary in place
cp c/deepseek_v4.cpu c/deepseek_v4
echo "  restored c/deepseek_v4 <- deepseek_v4.cpu (default)"
