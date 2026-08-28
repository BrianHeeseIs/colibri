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

## `bench/golden.sh` is a NECESSARY gate, not a SUFFICIENT one
Its prompt is ~20 tokens and fits a **single** 64-token prefill chunk. Any feature conditioned on
batch / group / chunk size may therefore **never execute during it**, and golden passes trivially.

This already produced one false result here: a rows16 change PASSED golden and was written up as
"bit-exact", then **diverged on p256** (184 tokens, 3 chunks). The entry had to be retracted.

- Before writing "bit-exact", run a differential on a **multi-chunk** prompt (p256 or larger),
  60 tokens, comparing the extracted `generated_text` md5 with the feature ON vs OFF.
- Always run the **determinism control** first: the same arm twice. If an arm does not reproduce
  its own md5, an ON-vs-OFF difference proves nothing.
- A golden **FAIL is informative**: a changed md5 proves the path executed. A golden PASS with a
  flag ON proves nothing about execution — capture a counter that is zero without the feature.

## Two performance axes. Never conflate them
- **TTFT** (load + prefill): `bench/ab.sh`. Decode excluded by construction.
- **tok/s** (decode): `.backlog/lab/tokps.sh`. `tok/s = (max_tokens-1) / after_first`, where
  `after_first` is `gen_stats.decode_sec` (`c/deepseek_v4.c:11708`).

They do not move together. `COLI_V4_METAL_ATTN=1` is -15.9 % TTFT but trends *negative* on decode.
`COLI_V4_KERNELS=all` grows on TTFT with prompt length while its tok/s gain is non-monotonic.

**Noise floors:** TTFT spread 0.6-0.8 % (n=3 is fine). **Decode spread 5-13 % — use n>=5.** A decode
delta under ~10 % cannot be resolved at n=3, and several apparent effects here reversed when a
third or fifth point was added.

## Seed: the harnesses check existence, not correctness
`bench/ab.sh` and `bench/golden.sh` only test that `/tmp/coli_usage.snapshot` EXISTS. A
present-but-wrong seed corrupts results silently. Sync from the durable copy before every run:
`cp .backlog/lab/coli_usage.snapshot /tmp/coli_usage.snapshot`.
`.backlog/lab/tokps.sh` and `.backlog/lab/taskcheck.sh` verify the md5 and abort loudly.

## Fastest known setting
**`COLI_V4_KERNELS=all`** — measured -10.4 / -18.0 / -20.2 % TTFT and +16.7 / +12.1 / +20.1 % tok/s
at p064 / p256 / p512. Keep it **OFF** for golden, bit-exactness differentials, and regression
triage: its output is nondeterministic at short prompts (two variants over eight runs) though
bit-exact at p512.

## Non-bit-exact changes are gated on TASK-LEVEL correctness
Operator-approved bar: capability must be identical (verifiable arithmetic / factual / code / logic
answers stay correct); token identity is not required. Harness `.backlog/lab/taskcheck.sh`.
Reproducibility is a SEPARATE property this bar does not cover — state it explicitly when it fails.

### NEVER conclude an approach is wrong from a CHANGED MD5. READ THE TEXT FIRST.
A golden FAIL means "the output is not token-identical". It does **not** mean the output is wrong,
and it is **not** grounds to reject, revert, or write off an approach. **Slight variations with the
meaning retained are fully acceptable for this project's goal.**

Before calling any approach broken, you MUST:
1. Extract the generated text from BOTH arms and **diff it**. Report what actually changed.
2. Run `.backlog/lab/taskcheck.sh` — the capability gate.
3. Run the arm against itself once to separate a *difference* from *nondeterminism*.

Only after those three may you use the words "fails" or "breaks".

This was learned the expensive way in E97. Widening the Metal expert path to rows16 was reported as
"BREAKS GOLDEN" and reverted on the strength of an md5 alone. The text was then compared and the
ENTIRE divergence was one token in sixty — `FFN layers.` versus `FFN layer.` — deterministic, with
`taskcheck` scoring 5/5 on both arms and byte-identical output on its prompt. The approach was
sound; a 45%-of-expert-calls speedup had been discarded because nobody looked at the words.
`bench/golden.sh` deletes its run directory on exit (`trap cleanup EXIT`), so capture the text
yourself — run the engine directly, or use `.backlog/lab/tokps.sh`/`taskcheck.sh`, which keep
transcripts.

Corollary for the ledger: "fails golden" is never a sufficient experiment write-up. Record WHICH
tokens changed and whether capability survived, or the entry will mislead the next session.

## Re-measure every upstream claim before adopting it
Three upstream performance claims in a row failed to reproduce on this host: #1097 loader lanes
("1.41x decode"), `V4_NGRAM` ("+18.5 %"), and `e36a1c7` hot-pack ("~6.9 s TTFT"). Upstream numbers
come from different hardware and workloads. Measure first; pay the adoption cost only if it
survives. Details in `experiments_results.md` E87-E94 and `mem:deepseek_v4/dead_levers`.
