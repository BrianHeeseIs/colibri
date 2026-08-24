# AGENTS.md — working rules for this repository

## NEVER USE /tmp FOR ANYTHING YOU NEED

**`/tmp` is cleared on reboot, standby, and at the OS's discretion. Treat it as write-only
scratch that can vanish between two commands.**

This is not hypothetical. Both of these were lost mid-session on this machine:

| lost | consequence |
|---|---|
| `/tmp/coli_usage.snapshot` | the sacred benchmark seed. A gate run started with a broken seed protocol — every `cp` failed silently while the benchmark carried on, producing numbers not comparable to anything previously recorded |
| `/tmp/REF_MD5` | the reference binary hash for the `a20c7aa` binary-identity proof |

### The rule
- Anything you will need again — seeds, reference hashes, logs, measurements, notes, plans,
  recovered files — goes in the **repository**, under `.backlog/` (or another tracked path).
- `/tmp` is acceptable ONLY for a value consumed within the same command, that you would not
  mind losing instantly, and whose loss cannot silently corrupt a result.
- If a script reads a file it needs, it must **verify** the file exists AND hashes as expected,
  and **abort loudly** if not. A silent `cp` failure that lets a benchmark continue is worse
  than a crash, because it produces confident wrong numbers.
- Scratch BUILD trees (e.g. instrumented variants) may live outside the repo since they are
  rebuildable — but never the artefact you intend to compare against.

### Canonical durable locations in this repo
- benchmark seed: `.backlog/lab/coli_usage.snapshot` (md5 `599f3d12e9347ef30541bd6f9ba18bde`)
- reference binary hash: `.backlog/lab/REF_MD5`
- benchmark logs / runners: `.backlog/lab/`
- durable notes and resumable state: `.backlog/*.md`

## Benchmark integrity (earned the hard way)
- Golden output md5 `5d04890413ff539e802985ce8c727814` is SACRED. Never edit the expected
  value to make something pass; fix the code.
- `bench/ab.sh` prints `delta = 100*(on-off)/off` — **FASTER IS NEGATIVE**.
- `bench/ab.sh` runs `--max-tokens 1` and parses only `time_to_first_token`; the metric is
  TTFT (load + prefill) and **decode is excluded by construction**. `p064`/`p256` are prompt
  FILES, not token counts.
- `bench/ab.sh` is `set -euo pipefail`, sends engine output to a per-run log inside a
  `mktemp` dir, and deletes that dir via `trap cleanup EXIT`. A non-zero engine exit
  therefore kills it **silently and destroys the error**. Pre-flight the engine yourself.
- Build with `make -C c -f Makefile.deepseek-v4 METAL=1 deepseek-v4 -j8`. `METAL ?= 0`, so a
  plain `make` silently compiles the Metal seam out with zero errors and zero warnings, and
  every Metal env flag becomes a no-op. Verify:
  `nm c/deepseek_v4 | grep -c coli_v4_metal_expert_forward_batch` must be > 0.
- One engine at a time: `pgrep -f '[d]eepseek_v4'` must be empty before starting one.
- Before dividing any counter by any wall, open the harness and read the exact invocation and
  the field it parses. Never infer a metric's definition from a sibling script.
