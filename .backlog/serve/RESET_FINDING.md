# V4 Persistent-Serve KV Reset Finding

This investigation created or modified **only this file**. No engine was launched.

Repository-relative citations below are exact source locations. The general serve
protocol is not authoritative for V4 where it conflicts with engine code: it says
code wins at `docs/serve_protocol.md:14-16`.

## 1. V4 does not support `KV_SLOTS > 1`

**Rule out slot rotation definitively.**

- `coli serve` rejects every non-GLM architecture when `--kv-slots != 1` at
  `c/coli:1144-1149`.
- Gateway validation independently rejects DeepSeek V4 when `kv_slots != 1` at
  `c/openai_server.py:2783-2786`, although its child environment would otherwise
  pass `SERVE=1`, `SERVE_BATCH=1`, and `KV_SLOTS` at
  `c/openai_server.py:1454-1458`.
- V4's parser reads the mux `slot` field, then rejects every value except zero at
  `c/deepseek_v4.c:9044-9055`.
- V4 creates one `ColiV4Session` at startup at `c/deepseek_v4.c:9248-9259` and
  passes that same session to every request at `c/deepseek_v4.c:9266-9274`.
- V4 ignores `SERVE_BATCH`; any `SERVE=1` invocation enters its one-session serve
  main at `c/deepseek_v4.c:9283-9286`. It is not GLM's multi-slot mux engine.
- V4 documentation confirms one active slot and lists more slots as future work:
  `docs/deepseek-v4.md:74-77`, `docs/deepseek-v4.md:102-105`.

Therefore `SUBMIT ... 1 ...` is not a neutralizer. It returns `ERROR <id> bad
submit header`; rotation `0,1,2,...` cannot run on V4.

## 2. Mux reset / clear reachability

**There is no explicit V4 mux reset frame.**

- V4 request parsing recognizes only `CANCEL`, `STOP`, and `SUBMIT` at
  `c/deepseek_v4.c:9035-9042`. Any `RESET` or `\x02RESET` line is ignored; it
  cannot reach a state-clearing function.
- V4 does contain an internal attention reset routine, which zeros window KV and
  resets compressed/indexer state at `c/deepseek_v4.c:1501-1507`, but no mux
  control path calls it directly.
- `STOP` and `CANCEL` have the same V4 engine behavior. While decoding, either
  matching active ID sets `stream.cancelled` and returns the normal callback-stop
  signal at `c/deepseek_v4.c:9100-9107`. There is no KV/history clear there.
  Generation then completes normally at `c/deepseek_v4.c:8838-8841`; serve emits
  `DONE`, merely suppressing `length_limited` when cancelled, at
  `c/deepseek_v4.c:9204-9217`.
- This differs from generic protocol documentation claiming `CANCELLED`; source
  code is the V4 behavior. Neither `STOP` nor `CANCEL` is a reset. Do not send
  either between benchmark turns.

`\x02RESET\n` is real only for GLM's legacy `run_serve` path at
`c/colibri.c:7472-7474`. It is not usable for V4: V4 enters `v4_serve_main` for
every `SERVE=1` launch (`c/deepseek_v4.c:9283-9286`), and V4 `coli chat` starts
the OpenAI server rather than legacy direct chat (`c/coli:972-1005`).

## 3. Actual V4 prefix behavior

V4 is **not** generic longest-common-prefix `truncate-and-extend` behavior.
It is all-or-nothing: exact prior fed sequence plus at least one new token, or a
full reset and prefill.

- `session->fed` describes all token IDs already represented by attention,
  including prompt and generated tokens (`c/deepseek_v4_internal.h:687-692`).
- `kv_prefix_reuse()` returns the entire prior length only if that whole record
  equals the beginning of a *longer* incoming prompt; tainted, shorter, equal,
  or any mismatch returns zero (`c/kv_prefix.h:116-127`). It never returns a
  partial common-prefix length.
- V4 documents why: recurrent compressed/window attention cannot be rewound to
  an arbitrary point, so only an exact conversation extension can continue
  (`c/deepseek_v4.c:8460-8475`).
- On `reuse == 0`, V4 resets every attention layer and DSpark history
  (`c/deepseek_v4.c:8477-8482`). It then prefills `fresh == prompt_count` tokens
  from absolute position zero (`c/deepseek_v4.c:8488-8506`) and overwrites the
  logical fed record from zero (`c/deepseek_v4.c:8541-8545`).

Thus four standalone/distinct benchmark prompts do **not** accumulate attention
context. They already get the required KV reset while the engine and expert
store remain alive. The stated KV-growth hypothesis is false for those requests.

The observed inverse `hit_pct`/`tok_s` trend has a concrete accounting confound:
the per-request hit rate surrounds the whole `coli_v4_session_generate()` call
(`c/deepseek_v4.c:9174-9203`), hence includes full prompt prefill. `tok_s`,
however, divides completion by `stats.decode_sec` (`c/deepseek_v4.c:9204-9217`),
which starts only after prompt prefill and first output setup
(`c/deepseek_v4.c:8504-8553`, `c/deepseek_v4.c:8818-8825`). Rising all-request
hit percentage therefore does not measure decode-cache warmth. Different prompt
and decode routes are also different workloads. The engine mutates its expert
policy on every lookup and can repin at its configured interval
(`c/deepseek_v4.c:6248-6254`), so four aggregate samples cannot establish a
physical monotonic decode slowdown. It does establish neither KV growth nor a
valid comparison between those two aggregate metrics.

## 4. Candidate mechanisms, ranked

| Rank | Mechanism | Neutralizes V4 context? | Preserves in-memory hot expert cache? | Per-turn cost / verdict |
| --- | --- | --- | --- | --- |
| 1 | Send independent, non-extension `SUBMIT`s on slot `0` | Yes | Yes | No extra control action; full prefill of current prompt is required. **Use this now.** |
| 2 | New V4 mux reset frame | Would | Would | Best future explicit API, but **does not exist today**. No bytes can invoke it. |
| 3 | Slot rotation | Would on a true multi-slot engine | Would | **Unavailable:** gateway and C parser reject V4 slots other than `0`. |
| 4 | Legacy `\x02RESET` | Yes for GLM legacy only | Yes for that GLM process | **Unavailable for V4.** Sending it to V4 is ignored. |
| 5 | `STOP` or `CANCEL` between turns | No | Yes | Wrong tool: leaves V4 state intact and changes output length. |
| 6 | EOF / engine restart per turn | Yes | No | Last resort only. EOF exits the loop, destroys session and engine at `c/deepseek_v4.c:9266-9277`; resident expert cache is lost. Per-turn `.coli_usage` persistence exists (`c/deepseek_v4.c:6005-6026`) but is not a retained hot resident cache. |

### Exact recommended mux bytes

Use ordinary one-slot submissions. Do **not** send a reset frame. These literal
payloads are five UTF-8 bytes each and cannot be exact extensions of each other's
prior prompt-plus-generated history:

```text
SUBMIT 1 0 5 80 0 1\nalpha\n
SUBMIT 2 0 5 80 0 1\nbravo\n
SUBMIT 3 0 5 80 0 1\ncharl\n
SUBMIT 4 0 5 80 0 1\ndelta\n
```

Here `\n` denotes one LF byte. For real rendered payload `P`, replace `5` with
the exact UTF-8 byte count of `P`, retain slot `0`, and send one LF after its
payload. V4 reads exactly the declared payload bytes and consumes that trailing
LF at `c/deepseek_v4.c:9063-9067`.

For a benchmark driver, avoid resending prior assistant output inside a growing
chat transcript. A repeated fixed standalone payload is even stronger for a
context-reset probe: equal-length prior history is deliberately non-reusable
(`c/kv_prefix.h:119-127`).

## 5. Falsifiable probe

Run one persistent V4 serve process. Keep engine settings, `max_tokens`, and
payload fixed. Send the exact same standalone `SUBMIT` at least 10 times, all
with slot `0`; capture raw mux stdout and engine stderr.

Expected reset evidence on **every** turn:

1. Raw V4 `DONE` has the implementation-only trailing field
   `prefix_reused`: `DONE <id> STAT <emitted> <tok_s> <hit_pct> <rss_gb>
   <prompt_tokens> <length_limited> <prefix_reused>`
   (`c/deepseek_v4.c:9209-9217`). It must be `0`.
2. `prompt_tokens` must stay exactly constant for fixed payload.
3. Set `COLI_V4_PROFILE=1`. V4 does **not** emit generic mux `PERF t_attn
   t_kvb`; its actual profiling writes stderr phase totals, including
   `attention` and `attn_kv_assembly`, at `c/deepseek_v4.c:7237-7268`.
   Those decode-only phase totals should remain flat within run noise for the
   repeated fixed workload. They must not rise turn-by-turn with prior turns.
4. `tok_s` can improve, worsen, or jitter as the expert store/pins and host
   change. It is not reset proof. Interpret it with same-turn expert data, not
   all-request `hit_pct` alone.

Falsification: a nonzero trailing `prefix_reused`, or rising decode attention /
KV-assembly totals for an otherwise identical fixed prompt, means this result
does not have the expected full-reset path. For intentionally continuing
transcripts, nonzero `prefix_reused` is expected and those turns must not enter
the independent-prompt benchmark set.
