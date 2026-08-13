# Ultrawork Notepad — Metal backend for DeepSeek-V4 on macOS arm64 (ft-deepmetal)
Started: 2026-08-13 02:00 local

## Mandate (user, 2026-08-13 ~02:00)
- Build Metal capability so REAL numbers can replace today's ESTIMATES tomorrow.
- User OVERRIDES the DEFER verdict in .backlog/deepseek-v4-metal-design.md (8.1% warm
  offloadable < 10% kill threshold). Explicitly wants it built anyway.
- NO LLM load testing until user greets "good morning"/"good afternoon".
- HARD: do not tax host - fans must not audibly spin (user asleep).
- Do NOT re-download the model. May use external SSD; may delete GLM-5.2 there if space needed.
- Save ALL questions for tomorrow.
- Review with Atlas / Oracle / Plan. Use team mode where beneficial.
- Build ALL variants likely relevant for tomorrow's testing.

## Known blockers from today's DEFER (must be addressed, not ignored)
B1. Existing Metal MoE pipeline computes a DIFFERENT function than V4 CPU path
    (BF16 rounding boundaries, swiglu_limit clamping, route-weight ordering).
B2. Bit-exactness may be structurally impossible: rows16 CPU accumulates sequentially
    per row; moe_gemv tree-reduces across 32 lanes via simd_sum.
B3. moe_gemv accepts fmt {1,2,5,6} only - fmt=4 grouped-int4 unsupported for batched MoE.
    V4 routed experts are NATIVE FP4 -> likely a new fmt path entirely.
B4. V4 has ZERO Metal surface: 0 COLI_METAL refs in deepseek_v4.c vs 70 in colibri.c.

## Scenarios (the contract) - TO BE FILLED after plan agent
## Now
Bootstrap: notepad + branch + parallel research agents.
## Todo
## Findings
## Learnings

## Findings (2026-08-13 02:09)
F1. HOST: Apple M3 Max, 40 GPU cores, Metal 4, 137.4 GB unified, macOS 26.6.1 (25G76).
    40 GPU cores is substantial - the DEFER's 8.1% offloadable was a CPU-side profile,
    not a statement about GPU capability.
F2. BLOCKER: offline Metal toolchain NOT installed.
      xcrun metal -> "missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain"
      xcrun metallib -> not found
    => cannot precompile .metal -> .air -> .metallib today without a component download.
F3. WORKAROUND (no download): MTLDevice newLibraryWithSource:options:error: compiles MSL at
    RUNTIME using the Metal.framework, which IS present. llama.cpp ships shader source and
    compiles at runtime for portability. Design for BOTH: embedded source (always works)
    + optional precompiled metallib (faster start) when the toolchain exists.
F4. DISK: / has 64Gi avail. External: "Extreme SSD" 109Gi avail (holds GLM-5.2, user
    authorised deleting it if needed). Metal work needs ~nothing; no deletion required.
F5. /tmp is VOLATILE on this host (wipes on reboot; host has a history of night crashes).
    Notepad and all probe/kernel sources live in the repo from now on:
      .backlog/ft-deepmetal-notepad.md   this file
      c/metal/                           kernel sources (.metal / .m)
      validation/metal/                  probes + parity harnesses
F6. CAPABILITY PROVEN (validation/metal/probe_fp4_runtime.m, 02:11):
      runtime MSL compile via newLibraryWithSource: OK
      threadExecutionWidth=32  maxTotalThreadsPerThreadgroup=1024
      FP4(E2M1) nibble + UE8M0 scale decode: BIT-EXACT vs CPU reference (8/8 values)
    => Blocker B2's premise ("bit-exactness may be structurally impossible") is FALSE at the
       DECODE level. Whether it holds at the REDUCTION level (simd_sum tree vs sequential
       row accumulation) is a separate, still-open question.
F7. Toolchain install requested by user at 02:12 -> xcodebuild -downloadComponent MetalToolchain

## BLOCKER DISSECTION — all three refuted as fatal (agent bg_45f6d2b6, 02:15)

B1 "Metal computes a DIFFERENT function"  -> COSTLY, not fatal.
   V4 CPU HAS a clamp; GLM/Metal has plain SiLU with NO clamp:
     deepseek_v4.c:1383-1395  coli_v4_swiglu(): gate=fmin(gate,limit);
                              up=fmax(-limit,fmin(up,limit))
     colibri.c:1166-1167      siluf(x)=x/(1+exp(-x))          <- unclamped
     backend_metal.mm:246-247 moe_silu kernel                  <- unclamped
   V4 BF16-rounds at exact boundaries (deepseek_v4.c:6130-6140 and rows16 6213-6224):
     round(gate), round(up), activated[i]=round(activated[i]*route_weight), round(output)
     coli_bf16_round at :10129-10138 = round-to-nearest-even on the low 16 bits.
   ROUTE-WEIGHT ORDER DIFFERS:
     V4  applies weight BEFORE down projection      (deepseek_v4.c:6135-6138)
     GLM applies weight AFTER  down, during scatter (backend_metal.mm:1300-1302)
   V4 accumulates experts in ASCENDING EXPERT ID, not rank order (deepseek_v4.c:3155-3172).
   => Cannot reuse GLM's MoE verbatim. Need a V4-specific kernel. Implementation work, NOT
      a semantic impossibility.

B2 "bit-exactness structurally impossible"  -> COSTLY, not fatal. Largely REFUTED.
   CPU MoE parallelises across INDEPENDENT output rows/tiles; each tile owns its accumulator
   (deepseek_v4.c:10805-10822 AVX512, :11034-11072 NEON). Agent found NO OpenMP reduction in
   the MoE math - only cache warm-up uses reduction(+:warmed) at :5988.
   => The CPU IS deterministic here, so demanding GPU exactness is a COHERENT requirement.
   My own probes already settled the rest:
     simd_sum tree      = bit-reproducible over 64 runs, 6.481e-06 rel err vs CPU
     serial-chain kernel = BIT-IDENTICAL to CPU (0x493c59f0)
   => Both a fast-approximate and an exact kernel are achievable. This becomes a RUNTIME
      CHOICE, which is exactly what makes tomorrow's measurement worth doing.

B3 "moe_gemv fmt {1,2,5,6} only; fmt=4 unsupported"  -> allowlist TRUE, relevance STALE.
   backend_metal.mm:1198-1203 moe_submit rejects unless fmt in {1,2,5,6}. Confirmed.
   BUT V4 does not use GLM's QT fmt numbering AT ALL:
     tensor.h:13-21          COLI_TENSOR_FP4_NATIVE_BLOCK
     deepseek_v4.c:5185-5201 view->format=FP4_NATIVE_BLOCK, scale=UE8M0,
                             block_rows=1, block_columns=32
     deepseek_v4.c:10236-10243 matvec REQUIRES exactly that combination
   => "fmt=4 grouped-int4" was a red herring. The real gap is simply: Metal MoE has no
      native-FP4/UE8M0 path yet. New kernel needed - which we were writing anyway.

KEY ALIGNMENT: V4 block_columns=32 == threadExecutionWidth=32 on M3 Max.
   One FP4 block maps exactly onto one SIMD group. The hardware fits the format.

## KERNEL SPEC — routed expert, exact CPU order (agent bg_f92c118e, 02:16)

ENTRY: coli_v4_expert_forward_ref  (deepseek_v4.c:6111)
  input float[D] -> gate/up FP4 [I,D] -> down FP4 [D,I] -> output float[D]

EXACT SEQUENCE (any deviation breaks parity):
 1. FP8 ACTIVATION QDQ, group=128   coli_fp8_activation_qdq_ref(...,columns,128)  :10392
    ** easy to miss: activations are FP8-quantised BEFORE the matmul **
 2. matmul_mxfp4(gate), matmul_mxfp4(up)                                          :10396-10399
 3. bf16_round_array(gate); bf16_round_array(up)                                  :6130-6131
 4. swiglu: g=fmin(g,limit); u=fmax(-limit,fmin(u,limit));
            out = g*sigmoid_stable(g)*u                                           :1383-1395
 5. activated[i] = bf16_round(activated[i] * route_weight)                        :6136-6137
 6. matmul_mxfp4(down)                                                            :6138
 7. bf16_round_array(output)                                                      :6140

matmul_mxfp4 INNER ORDER (quant.h:1366-1412) - must be reproduced:
    for o asc: for s asc: for g asc(blocks of 32):
        ga = 0
        for i in block step 2:  ga += x[i]*lut[byte&0xF]; ga += x[i+1]*lut[byte>>4]
        a += ga * sc          <- scale applied PER BLOCK, after the unscaled block sum
    y[o] = a

FP4 DECODE - use the COLD-PATH forms for bit-exactness:
    mx4_lut[16] = {0,.5,1,1.5,2,3,4,6, -0,-.5,-1,-1.5,-2,-3,-4,-6}   (quant.h:1361)
    mx4_scale(s) = bitcast_float(uint32(s) << 23)                     (quant.h:1363)
    ** NOT ldexpf(1,s-127): they differ at s=0 (bitcast->0.0, ldexp->denormal 5.9e-39)
    ** my probe used ldexp + magnitude LUT; the kernel MUST use bitcast + signed LUT
    low nibble = EVEN column, high nibble = ODD column

ROWS16 HOT LAYOUT (deepseek_v4.c:10648-10655):
    tile = row/16, lane = row%16
    packed[(tile*stride + col)*16 + lane] = data[row*stride + col]      (same for scales)
    => 16 rows interleaved; maps naturally onto SIMD lanes. GPU must support BOTH
       cold (block_rows=1, row-major) and hot (block_rows=16, interleaved).

MoE COMBINE (deepseek_v4.c:3640-3646, 3702-3777):
    experts are iterated in ASCENDING EXPERT ID (not top-k rank order)  :3641
    output[i] = 0
    for each selected expert asc id:  output[i] += expert_output[i]
    output[i] = bf16_round(output[i] + shared_output[i])                 :3776-3777

SHARED EXPERT (deepseek_v4.c:6145-6179): FP8_E4M3 weights, NO route weight,
    activated rounded directly (not multiplied-then-rounded). Separate kernel.

GPU CALL BOUNDARY (narrowest clean seam): replace the loop at deepseek_v4.c:3763-3768.
    Takes: input[D], N resolved expert views, expert_weights[N], swiglu_limit,
           zeroed output[D]; requires ascending-expert-id accumulation.

## FMA / CONTRACTION INVESTIGATION (02:21, plan agent Flag #1)

Plan agent's warning was the sharpest objection raised so far: the real inner loop is
`ga += x[i]*lut[...]`, a multiply-add that can contract to FMA. If CPU and Metal contract
differently, the EXACT kernel cannot be bit-identical regardless of summation order.

MEASURED (validation/metal/probe_fma_contraction.m, probe_fma2.m):
  SDK facts: MTLMathMode {Safe=0, Relaxed=1, Fast=2}; fastMathEnabled DEPRECATED since
             macOS 15.0 -> must use opt.mathMode. __MAC_26_0 = 260000.
  inputs 1+k*2^-12 (benign):
      cpu split == cpu explicit fma          -> YES
      gpu Safe/Relaxed/Fast all 0x4100480d   -> all == CPU
      => no divergence at all; math mode has NO observable effect
  inputs 1+k*2^-23 (float epsilon, pathological):
      cpu 0x41000009  vs  gpu 0x4100000a     -> 1 ULP APART
      cpu contract(fast) == cpu contract(off) -> clang did NOT contract either way
      gpu Fast == gpu Safe                    -> math mode changed nothing

VERDICT (honest): FMA contraction is NOT demonstrably the cause. Both sides agree on
ordinary values across every math mode. A 1-ULP divergence appears only at inputs sitting
exactly at float epsilon, and I have NOT isolated its cause. Candidate explanations not yet
discriminated: GPU internal precision on the mul, differing denormal handling, or the
compiler emitting a different instruction sequence for that specific pattern.

CONSEQUENCE FOR THE PLAN: do NOT claim bit-exactness on the strength of micro-probes either
way. T12 (parity on real matmul_mxfp4 arithmetic with real FP4 blocks) is the test that
decides it. Design for BOTH outcomes:
  - if exact holds  -> ship EXACT + FAST variants, user picks tomorrow on measured cost
  - if it does not  -> EXACT degrades to "bounded tolerance", FAST still ships, and the
                       tolerance is REPORTED not hidden. That is an acceptable outcome,
                       not a failure - but it must be stated plainly in the results.

## QUESTIONS QUEUED FOR USER (do not ask tonight)
Q1. If EXACT turns out to be 1-ULP off rather than bit-identical, is bounded-tolerance
    parity acceptable for a production default, or must the CPU path remain authoritative?
Q2. Tomorrow's benchmark shape: which token counts / prompt lengths do you want swept?
Q3. Should BOTH layouts (cold row-major + hot rows16) ship, or is hot the only production
    path worth keeping?
Q4. Acceptable relative-error bound for the FAST variant? Measured 6.481e-06 on a
    deliberately hostile vector.

## T3 CPU PARITY HARNESS (2026-08-13)

- Added `validation/metal/synth_tensors.h` and `validation/metal/parity_v4.m`.
  Fixture is deterministic and model-free: hidden=64, intermediate=128, experts=4,
  top-k=2, tokens=2, FP4 block=32, FP8 activation group=128.
- Canonical FP4 bytes/scales emit both cold row-major and hot rows16 using exactly
  `packed[(tile*stride+col)*16+lane]`. All 12 expert matrices unpack byte-for-byte.
- `coli_v4_expert_forward_ref` requires contraction columns divisible by 128. Harness
  therefore validates each logical-hidden=64 result against a separate reference view
  padded to 128 columns/rows; padded weights are deterministic and padded activation
  values are zero. Seven visible checkpoints still use exact logical geometry.
- Harness directly calls both `matmul_mxfp4` and `coli_v4_expert_forward_ref`; every
  manually replayed final result matches the wrapper bit-for-bit.
- Build (CPU only, no Metal device/model):
  `clang -std=c11 -O2 -Wall -Wextra -Wno-unused-function validation/metal/parity_v4.m -lm -o validation/metal/parity_v4`
- Determinism: two consecutive summary runs SHA-256
  `f9f77ecf725bf704c2f0afbb71734b6ec0b18f0836293e0fd5d4ee4a9d27f759`;
  two full golden files SHA-256
  `526650bb028fea69ebd263ff8e6d298e70544feab1fe5622a4161d1f92ec2326`.
  O0 and O2 summaries also compare byte-identical.
- Actual aggregate golden hex:
  1. fp8_activation_qdq: count=256, fnv64=`0xf8c32dc399b42d85`, first=`0x3ec00000`, last=`0xbf200000`
  2. gate_up_matmul: count=1024, fnv64=`0xd4d9840027cff60d`, first=`0xc066c000`, last=`0xc1130000`
  3. gate_up_bf16: count=1024, fnv64=`0xd350432bb53648f8`, first=`0xc0670000`, last=`0xc1130000`
  4. swiglu: count=512, fnv64=`0xfb6e5e7dfae59a72`, first=`0xbe921d34`, last=`0xc1092bb1`
  5. route_weight_bf16: count=512, fnv64=`0xcc9ef7ef995fa711`, first=`0xbe370000`, last=`0xc0700000`
  6. down_matmul: count=256, fnv64=`0x283a19db50dbcd5c`, first=`0xc2bf6bb0`, last=`0xc1c2bc7c`
  7. output_bf16: count=256, fnv64=`0x09d4b630742ba5e3`, first=`0xc2bf0000`, last=`0xc1c30000`
- Complete per-value IEEE-754 words: `validation/metal/parity_v4.golden.hex`
  (28 lines: seven steps x four token/rank expert calls).

## T2 BUILD WIRING (2026-08-13)

- `METAL=1` now adds only `backend_metal_v4.o` to V4's 26 amalgamation objects and
  links Metal/Foundation/libc++; parent `make deepseek-v4 METAL=1` forwards the flag.
- Offline shader pipeline is live: every `metal/coli_v4_*.metal` source compiles
  through `xcrun metal` to AIR, then `xcrun metallib` emits
  `c/build/metal-v4/deepseek_v4.metallib`.
- Runtime-source fallback is retained from exactly the same source list. Build emits
  `deepseek_v4_source.h`; `backend_metal_v4.mm` tries `newLibraryWithURL:error:` first,
  then `newLibraryWithSource:options:error:`. `COLI_V4_METALLIB` overrides the default.
- Default build object manifest before/after: 26 lines, identical SHA-256
  `e7b07731b295a241da97197f8bd578a7288c0eccdf31f95dde10673985498ca8`.
  Durable manifests: `.backlog/ft-deepmetal-v4-objects.before.txt` and
  `.backlog/ft-deepmetal-v4-objects.after.txt`; `diff -u` exits 0 with no output.
- Darwin scoped build (`make -f Makefile.deepseek-v4 deepseek-v4 METAL=1 -j2`) emitted:
  `xcrun -sdk macosx metal -c metal/coli_v4_decode.metal -o build/metal-v4/coli_v4_decode.air`
  `xcrun -sdk macosx metallib build/metal-v4/coli_v4_decode.air build/metal-v4/coli_v4_matmul.air build/metal-v4/coli_v4_swiglu.air -o build/metal-v4/deepseek_v4.metallib`
  `clang ... backend_metal_v4.o -o deepseek_v4 ... -framework Metal -framework Foundation -lc++`
- Manual library smoke returned `V4_METAL_SMOKE available=1 library=metallib`; forcing
  a nonexistent metallib returned `V4_METAL_SMOKE available=1 library=source`.
- Simulated Linux and Windows default dry runs reported `LINUX_METAL_INJECTION NONE`
  and `WINDOWS_METAL_INJECTION NONE`; explicit `METAL=1` on either fails at parse time
  with `METAL=1 is supported only on macOS`.
- Follow-up after T1 kernel delivery: `V4_METAL_SOURCES` now discovers every approved
  `metal/coli_v4_*.metal`, currently decode + matmul + swiglu. The obsolete build-only
  stub was removed; backend validation uses `coli_v4_probe_primitives`. Shader files stay
  unguarded because only Metal compiles them and runtime source compilation defines no host
  macro. Runtime concatenation exposed duplicate
  sigmoid definitions between decode/swiglu, fixed with `COLI_V4_SIGMOID_DEFINED`; both
  metallib and forced-source smoke again report available=1.

## T4 BENCHMARK HARNESS READY, NOT RUN (2026-08-13)

- Added `validation/metal/bench_v4.m`: CPU reference plus full shipping matrix
  EXACT|FAST x COLD row-major|HOT rows16.
- Harness accepts `--warmup`, `--iterations`, and optional `--variant`; reports wall
  mean/median/min for every variant and command-buffer GPU timestamp mean/median/min for
  Metal variants.
- Production model/kernel access is an explicit `coli_v4_bench_*` adapter seam. The
  standalone harness compiles now and rejects approved execution until adapter version 1
  is linked.
- Execution is hard-gated before argument parsing, autorelease-pool creation, any model
  hook, and `MTLCreateSystemDefaultDevice()`. Only the exact setting
  `COLI_V4_METAL_BENCH_APPROVED=USER_CONFIRMED_REAL_MODEL_BENCHMARK` can cross it; generic
  truthy values do not.
- Safe proof only (no approved execution attempted): strict `clang ... -Werror` compile
  succeeded. No flag and deliberately wrong value both printed loud refusal, exited 0,
  and reported `model_open=0 metal_device=0 cpu_dispatch=0 gpu_dispatch=0`.
- `--list-variants` exited 0 and printed matrix without model or Metal access.

### T4 contract correction (same night, no benchmark run)

- Renamed benchmark rows from `EXACT|FAST` to `ORDERED|SIMD`. T1 proved Metal sigmoid
  differs from CPU libm by up to 2 ULP, so `EXACT` would be a false semantic claim;
  `FAST` was also too vague to identify the reduction strategy.
- Report now includes per-row wall/GPU standard deviation alongside mean/median/min.
- A failed adapter warmup or measured dispatch prints that row as `UNAVAILABLE` and the
  matrix continues, preventing one unsupported path from suppressing other results.
- Recompiled with `-Werror`; repeated only safe modes. No flag and wrong flag still exited
  0 with all safety counters zero. Listing showed CPU plus ORDERED/SIMD x COLD/HOT.

### T4 adapter refinement after T5a kernel landing (no benchmark run)

- Variant descriptors now carry locked MSL entry names into the adapter ABI. COLD rows map
  to the verified `coli_v4_matmul_mxfp4_ordered` and `coli_v4_matmul_mxfp4_simd` kernels.
- No rows16 kernel entry exists yet, so HOT rows deliberately print `adapter-required`
  instead of inventing speculative function names. Production adapter may encode the hot
  path when it lands; failure reports that row `UNAVAILABLE` and continues the matrix.
- Strict compile and safe-path proof repeated: refusal counters stayed zero; listing showed
  exact COLD kernel bindings and explicit HOT adapter requirement. Approved path not run.

### T4 locked full-MoE entry correction (same night, no benchmark run)

- Corrected descriptor ABI from T5a matmul-probe names to T4's locked full expert names:
  `coli_v4_moe_expert_fp4_ordered_cold`, `coli_v4_moe_expert_fp4_ordered_hot`,
  `coli_v4_moe_expert_fp4_simd_cold`, and `coli_v4_moe_expert_fp4_simd_hot`.
- All four rows now represent complete gate/up/SwiGLU/weight/down expert dispatches; adapter
  may mark a row `UNAVAILABLE` until its production MSL function lands.
- Strict compile and safe refusal/list proofs repeated. No approved execution attempted;
  missing/wrong approval still reported all-zero safety counters and exited 0.

## T1 RESULT (02:28) — PRIMITIVES PORTED; TRANSCENDENTAL IS THE REAL BLOCKER

Files: c/metal/coli_v4_decode.metal, validation/metal/probe_primitives.m
RED->GREEN honoured: probe ran and failed (exit 2, port absent) BEFORE the .metal was written.
Repro: ./validation/metal/probe_primitives   (exit 0 = all exact, 1 = mismatch)

BIT-EXACT, verified (588/592):
  mx4_lut     16/16  exact  -- incl. index 8 == NEGATIVE ZERO (0x80000000), distinct from +0
  mx4_scale  256/256 exact  -- EXHAUSTIVE over all u8; incl. s=0 -> +0 and s=255 -> +inf
  e4m3fn     256/256 exact  -- EXHAUSTIVE over all 256 byte values, incl. NaN encodings
  bf16_round  32/32  exact  -- incl. NaN, +/-Inf, denormals, and round-to-even ties
                               (note: denormals were NOT flushed here)

NOT BIT-EXACT:
  sigmoid      4/32  FAIL, two distinct and unrelated causes:
    (A) DENORMAL FLUSH: exp(-88) -> cpu 6.054601e-39 (denormal), gpu +0. Same at -100.
        GPU flushes denormal RESULTS to zero in this path even though bf16 denormals survived.
    (B) TRANSCENDENTAL: precise::exp != libm expf. At x=-0.1 and x=-1/3, 1 ULP apart.

QUANTIFIED (validation/metal/probe_sigmoid_sweep.m, 8192 pts over [-40,40]):
    differing        : 1546  (18.87%)
    max ULP distance : 2
    max rel error    : 1.611012e-07  at x = -0.532291

WHY THIS MATTERS: sigmoidf_stable is on the SwiGLU HOT PATH --
  deepseek_v4.c:1394  output = gate * sigmoidf_stable(gate) * up
So this is not a corner case; it executes for every expert activation.

CORRECTION TO EARLIER HYPOTHESIS: the plan agent flagged FMA contraction as the #1
bit-exactness risk, and I pursued it. That was WRONG. FMA/contraction showed no divergence
across Safe/Relaxed/Fast on ordinary inputs. The actual blocker is the TRANSCENDENTAL, which
neither the plan agent nor I had ranked. Probing found it; reasoning did not.

CONSEQUENCE — "EXACT" must be renamed and re-scoped. It cannot mean bit-identical:
  EXACT  = bit-identical for FP4 decode, scale, bf16, e4m3, and reduction order;
           sigmoid bounded at <=2 ULP / 1.61e-07 relative. State this in the results, do not bury it.
  FAST   = simd_sum tree, 6.481e-06 relative.
Perspective for the decision, NOT an excuse: FP4 (E2M1, 8 magnitude levels) carries ~6e-2
quantization error. 1.61e-07 is ~5 orders of magnitude below the format's own noise floor.
Numerically irrelevant; bit-exactness claim is still dead. Both facts get reported.

OPEN OPTION (not attempted, costs unknown): hand-port an expf matching Apple libm bit-for-bit.
Only worth it if the user answers Q1 with "must be bit-identical".

## OFFLINE TOOLCHAIN PATH VALIDATED ON REAL SOURCE (02:30)
Both compilation routes now proven against c/metal/coli_v4_decode.metal (not a toy):
  xcrun -sdk macosx metal    -c c/metal/coli_v4_decode.metal -o decode.air      -> exit 0, 4272 B
  xcrun -sdk macosx metallib decode.air -o decode.metallib                      -> exit 0, 4462 B
  newLibraryWithURL: -> loads OK, functionNames = [coli_v4_probe_primitives]
  pipeline builds: maxTotalThreadsPerThreadgroup=1024, threadExecutionWidth=32
=> file-scope `constant float[]` arrays and the `precise::` namespace both survive offline
   compilation. No source changes needed for the metallib rule.
=> simdWidth=32 reconfirmed on a real pipeline: V4 block_columns=32 == one SIMD group.

## T5a RESULT (02:33) — matmul_mxfp4 ORDERED IS BIT-EXACT ON GPU

Files: c/metal/coli_v4_matmul.metal, validation/metal/probe_matmul.m
RED->GREEN honoured (probe exit 2 with kernel absent, then run).
Repro: ./validation/metal/probe_matmul     (exit 0 = ordered bit-exact)

REFERENCE CLARIFICATION (important, was not in the earlier spec):
c/quant.h has TWO matmul_mxfp4 bodies. The AVX2 one is #ifdef __AVX2__ and NEVER compiles on
Apple silicon. The arm64 ground truth is the SCALAR path, c/quant.h:1401-1412.
Its accumulation is TWO-LEVEL and the order is load-bearing:
    per 32-col group:  ga accumulates serially across columns
    then:              a += ga * sc     <- scale applied ONCE per group, AFTER the group closes
A flat reduction over I is NOT equivalent and will not reproduce CPU bits.
Nibble order: low nibble = EVEN column, high nibble = ODD column, odd guarded by
`if (i+1 < base+glen)`. Tail groups clamp glen = I - base.

MEASURED:
  ordered  S=2 I=64 O=8 ng=2   BIT-EXACT   0/16 diff, maxULP 0
  ordered  S=2 I=48 O=8 ng=2   BIT-EXACT   0/16 diff, maxULP 0   <- tail16, exercises glen clamp
  ordered  S=1 I=32 O=4 ng=1   BIT-EXACT   0/4  diff, maxULP 0
  simd     S=2 I=64 O=8 ng=2   diff 12/16, maxRel 7.439e-06
  simd     S=2 I=48 O=8 ng=2   diff 14/16, maxRel 4.585e-06
simd agrees with the independently measured simd_sum figure (6.481e-06). Consistent.

METHODOLOGICAL CORRECTION (applies to the parity harness):
maxULP for the simd variant read 76 and 40, which LOOKS catastrophic but is not — the values
involved are ~3.99e3 and ~2.70e5. Integer ULP distance is a poor metric at large magnitude and
overstates the error. Gate on RELATIVE error; report ULP only as secondary colour. I had told
parity-harness to gate on ULP; that instruction was wrong and has been corrected.

RUNNING BIT-EXACTNESS LEDGER (what "ordered" actually buys):
  mx4_lut            BIT-EXACT (16/16, incl. negative zero)
  mx4_scale          BIT-EXACT (256/256 exhaustive)
  e4m3fn decode      BIT-EXACT (256/256 exhaustive)
  bf16_round         BIT-EXACT (32/32, incl. NaN/Inf/denormal/ties)
  matmul_mxfp4       BIT-EXACT (ordered variant, incl. tail)
  sigmoid            NOT exact -- 18.87% differ, <=2 ULP, 1.611e-07 relative
=> "ordered" is bit-identical to CPU in EVERYTHING except the sigmoid transcendental.
   That is a materially stronger position than the DEFER verdict assumed, and it vindicates
   the ordered/simd rename: "ordered" is precisely accurate, "exact" would have been a lie.

## T5b RESULT (02:35) — SwiGLU: THE SIGMOID ERROR MEASURED WHERE IT ACTUALLY LANDS

Files: c/metal/coli_v4_swiglu.metal, validation/metal/probe_swiglu.m
RED->GREEN honoured. Repro: ./validation/metal/probe_swiglu
Reference: c/deepseek_v4.c:1383-1396, ported verbatim.

PRESERVED ASYMMETRY (do not "tidy" this in review — it is deliberate in the CPU source):
    gate clamped ONLY FROM ABOVE   fmin(gate, limit)
    up   clamped on BOTH sides     fmax(-limit, fmin(up, limit))
    product associates left-to-right: (gate * sigmoid(gate)) * up

MEASURED over 4096 gate values swept across [-100, +100]:
    limit=0 (no clamp)  diff 805/4096 (19.65%)  denormal-flush 260  maxRel(non-flush) 2.641e-07
    limit=7 (clamped)   diff 798/4096 (19.48%)  denormal-flush 260  maxRel(non-flush) 2.346e-07
Clamping barely changes the rate, which is the expected consequence of the asymmetry: the
divergences cluster at NEGATIVE gate, and gate is not clamped from below.

TWO ERROR CLASSES, reported separately on purpose — do not merge them into one number:
  (1) DENORMAL FLUSH, 260 cases (6.3%). Deeply negative gate (roughly < -87) makes
      sigmoid(gate) underflow to a denormal on CPU and to exactly +0 on GPU. Relative error
      there is 100%, which is why it is excluded from maxRel — quoting a 100% relative error
      would be true and completely misleading. The ABSOLUTE error is what matters:
      gate=-88, sigmoid~6e-39, up~10  ->  output ~ -5.3e-36 on CPU vs 0 on GPU.
      Absolute error ~5e-36 against activations of order 1. Harmless by ~36 orders of magnitude.
  (2) TRANSCENDENTAL, the rest. maxRel 2.64e-07. This is ~1.6x the raw sigmoid figure
      (1.611e-07), consistent with the error propagating through two multiplies.

END-TO-END POSITION: worst-case SwiGLU relative error 2.64e-07 vs FP4's own ~6e-2
quantization floor. Five orders of magnitude below the noise the format already carries.
Statement to use: "numerically immaterial, but NOT bit-identical." Both halves, every time.

## INCIDENT (02:36-02:38) — COLI_METAL GUARD ON .metal FILES. MY INSTRUCTION WAS WRONG.

SYMPTOM: probe_primitives and probe_matmul went RED minutes after being GREEN. Compilation
SUCCEEDED but entry points were "missing" -- which is the diagnostic tell: an empty translation
unit compiles cleanly. probe_swiglu still passed because I wrote it after the edit.

ROOT CAUSE: I told build-wiring "new Metal code must sit behind #if defined(COLI_METAL) so the
26 non-Metal units never see it." They applied it faithfully -- including to the .metal shader
sources, wrapping each whole file. Under runtime newLibraryWithSource: COLI_METAL is undefined,
so the entire file preprocessed away.

PROOF (not inference):
    metal -c coli_v4_decode.metal              -> 2384 B .air   (empty TU)
    metal -DCOLI_METAL -c coli_v4_decode.metal -> 4272 B .air
    4272 B is exactly the size measured at 02:30 before any guard existed => content intact.

WHY THE GUARD IS WRONG **ON .metal FILES SPECIFICALLY**:
  - .metal files are compiled ONLY by the Metal toolchain. The C amalgamation never includes
    them, so the guard defends against a scenario that cannot occur.
  - It breaks runtime newLibraryWithSource:, our proven fallback path and what every probe uses.
  - Failure mode is SILENT: an empty TU compiles clean and only fails later at entry lookup.
The guard is CORRECT and stays on host .mm/.h code, which the C build genuinely does see.

FIX: removed the outer #if defined(COLI_METAL)/#endif from coli_v4_decode.metal and
coli_v4_matmul.metal only. Inner COLI_V4_MX4_DEFINED / COLI_V4_SIGMOID_DEFINED guards kept.
Verified: .air back to 4272 B with no -D. Suite 4/4 green.

## PROBE EXIT SEMANTICS FIXED (02:38) — SECOND SELF-CORRECTION
probe_primitives was gating on total bit-equality, so it was PERMANENTLY RED because sigmoid
legitimately diverges. A permanently-red test trains people to ignore it and would have masked
a real regression. Now it asserts:
    the 4 bit-exact groups stay BIT-EXACT              (any mismatch => REGRESSION)
    sigmoid transcendental  <= 2 ULP and <= 5e-07 rel
    denormal-flush count    <= 2  (exp(-88), exp(-100) in the fixed 32-sample vector)
First attempt at this bound was ALSO wrong: I applied ULP distance to the denormal-flush class
and got maxULP 4320708. That is the same error I had just corrected parity-harness for. Flush
is now counted as its own class and excluded from ULP/rel statistics.
Current: PASS -- "4 groups bit-exact; sigmoid 4/32 = 2 transcendental + 2 denormal-flush,
maxULP 1, maxRel 7.139e-08, within bound."

## SUITE STATE (02:38): 4/4 GREEN
  probe_primitives  PASS   probe_matmul  PASS   probe_swiglu  PASS   probe_fp4_runtime  PASS
  parity_v4 (T3)    PASS, run1==run2 deterministic, 7 golden steps hashed
  bench_v4  (T4)    refuses without approval token, safety counters all zero, exit 0

## T2 ACCEPTANCE GATE — VERIFIED BY ME, PLUS A RETRACTION (02:40)

RETRACTION FIRST. My initial check compared `make -n` on Makefile.deepseek-v4.orig vs live and
reported ">>> OBJECT LIST BYTE-IDENTICAL <<<". That conclusion was VACUOUS and is withdrawn.
The .orig aborts on Darwin ("supports only x86-64/aarch64 Linux and Windows/MSYS2", rc=2, 0
lines) and the live run said "up to date" (1 line). Both object lists were EMPTY. I compared
two empty sets and declared a pass — the exact false-green failure mode I had just warned the
team about. Caught on re-read of the raw numbers, not by any tool.

Also established: c/Makefile.deepseek-v4.orig is a STALE SNAPSHOT, 44 lines adrift from HEAD.
It was never a valid baseline. HEAD itself still refuses Darwin, so the Darwin support in this
makefile is an UNCOMMITTED working-tree change, not something in 2767d8e.

REAL VERIFICATION (forced dry run, -n -B, live makefile, zero compilation, zero fan noise):
    METAL=0 : 27 commands, 26 objects, 0 Metal artifacts
    METAL=1 : 39 commands, 27 objects, 5 Metal artifacts
    delta   : exactly ONE object -- backend_metal_v4.o

WHY 26 -> 27 IS THE DECISIVE NUMBER: the amalgamation compiles c/deepseek_v4.c once per unit
in Makefile.deepseek-v4.units (26 units, each -DCOLI_V4_UNIT_*). I required that
backend_metal_v4.mm compile EXACTLY ONCE and never be pulled into that sweep, because 26
duplicate ObjC++ objects would produce duplicate-symbol link failures. 26 -> 27 proves it
compiles once. Constraint satisfied.
And METAL=0 carrying ZERO metallib/.air/.mm commands proves the default build is unperturbed.
=> T2 acceptance gate PASSES, verified independently rather than accepted on report.

Makefile deltas vs HEAD: c/Makefile 2 lines, c/Makefile.deepseek-v4 57 lines (54 ins / 5 del).
Shader discovery uses $(wildcard metal/coli_v4_*.metal), so it now picks up decode, matmul and
swiglu with no -D needed (post guard removal).

## T4 CONTRACT RECONCILED (bench-scaffold, confirmed 02:40)
Uses ORDERED/SIMD (stale EXACT/FAST purged), the four locked full-MoE names
coli_v4_moe_expert_fp4_{ordered|simd}_{cold|hot}, GPUStartTime/GPUEndTime timing, per-row SD,
UNAVAILABLE continuation, exact approval-phrase gate. Correctly does NOT use the T5a matmul
probe entry names as benchmark rows -- those are probes, not MoE variants.

## ===== HANDOFF STATE (02:41) — READ THIS FIRST IF RESUMING =====

BRANCH ft-deepmetal. NOTHING COMMITTED (user forbids unrequested commits). All work is in the
working tree. /tmp is volatile on this host — treat only repo paths as durable.

WHAT EXISTS AND PASSES:
  c/metal/coli_v4_decode.metal   mx4_lut, mx4_scale, bf16_round, e4m3fn, sigmoid   [4 BIT-EXACT]
  c/metal/coli_v4_matmul.metal   matmul_mxfp4 ordered (BIT-EXACT) + simd (7.4e-06)
  c/metal/coli_v4_swiglu.metal   swiglu, asymmetric clamp preserved (2.64e-07 rel)
  validation/metal/probe_{primitives,matmul,swiglu,fp4_runtime,reduction_parity}.m
  validation/metal/parity_v4.m + synth_tensors.h   (T3: 7 golden steps, run1==run2 deterministic)
  validation/metal/bench_v4.m                      (T4: refuses without approval token)
  c/Makefile{,.deepseek-v4}  METAL=1 wiring, verified 26->27 objects, default build unperturbed

WHAT DOES NOT EXIST YET — this is the remaining critical path:
  1. The four MoE kernels themselves:
       coli_v4_moe_expert_fp4_{ordered|simd}_{cold|hot}
     Building blocks are all done and proven; what is missing is the 7-step assembly
     (spec is in this notepad under the kernel-spec section) plus the rows16 hot layout.
  2. Real host glue in c/backend_metal_v4.mm (currently a stub that links).
  3. Wiring the GPU seam at c/deepseek_v4.c:3763-3768.
  4. A real correctness hazard flagged in .backlog/deepseek-v4-metal-design.md:160-165 —
     slot->slab is overwritten IN PLACE for a different expert (deepseek_v4.c:5247-5262). If a
     slot is refilled while the GPU still references it, the GPU reads ANOTHER EXPERT'S WEIGHTS
     and is silently wrong. Design proposes synchronous v1. MUST be resolved before any
     benchmark number is believed.

BOTTOM LINE ON THE ORIGINAL QUESTION (is a bit-exact Metal V4 backend feasible?):
  YES for the arithmetic core. matmul_mxfp4 is bit-exact on GPU, including the tail path.
  NO for end-to-end bit-identity, because sigmoid's transcendental cannot match libm.
  Worst case measured end-to-end at the SwiGLU level: 2.64e-07 relative, versus FP4's own
  ~6e-2 quantization floor. Numerically immaterial; bit-identity claim is dead. Report both.
  This is materially better than the original DEFER verdict assumed.

NOTHING HAS BEEN MEASURED FOR PERFORMANCE. No model was loaded. Every number above is
correctness, not speed. The benchmark is built and deliberately gated OFF.

## T5c RESULT (02:45) — FP8 ACTIVATION QDQ BIT-EXACT (the hardest primitive)

Files: c/metal/coli_v4_fp8qdq.metal, validation/metal/probe_fp8qdq.m
RED->GREEN honoured. Repro: ./validation/metal/probe_fp8qdq
Reference: coli_fp8_activation_qdq_ref, c/deepseek_v4.c:10160. Runs at steps 1 and 5 of the
MoE expert chain (activation quantize before gate/up matmul, and before down matmul).

MEASURED: LEN=768 (6 blocks of 128), scales BIT-EXACT (0 bad), outputs BIT-EXACT 0/768, maxULP 0.
Input vector deliberately spanned every hard sub-path:
  wide values, tiny values (subnormal E4M3 encode region < 0.015625), near-448 clamp,
  values near the 1e-4 max floor, and ordinary mid-range.

FOUR sub-primitives ported and all bit-exact as a consequence:
  ceil_log2_positive  -- frexp-based, incl. the fraction==0.5 power-of-two boundary
  coli_e8m0_decode    -- ldexp(1, s-127). DISTINCT from mx4_scale bit-trick; diverges at s=0
                         (2^-127 vs +0). Used the correct ldexp form; that is why scales matched.
  coli_e4m3fn_encode  -- binary32 constant-time branch (FLT_RADIX==2 path), BOTH arms:
                         subnormal (magnitude<0.015625) and normal (round-to-nearest-even on
                         the 24-bit significand, incl. the rounded==16 carry into exponent).
  coli_e4m3fn_decode  -- already proven exhaustive in T1.

## COMPLETE MoE ARITHMETIC BUILDING-BLOCK LEDGER (02:45)
  fp8_activation_qdq   BIT-EXACT   (T5c, incl. encode round-to-even + e8m0 scale)
  matmul_mxfp4         BIT-EXACT   (T5a ordered, incl. tail)
  bf16_round           BIT-EXACT   (T1, exhaustive-ish)
  mx4_lut / mx4_scale  BIT-EXACT   (T1, mx4_scale exhaustive 256/256)
  e4m3fn enc/dec       BIT-EXACT   (T1 dec exhaustive; T5c enc across all sub-paths)
  swiglu / sigmoid     BOUNDED     (T5b, 2.64e-07 rel; the ONLY non-exact block)
=> Steps 1,2,3,5,6,7 of the 7-step MoE chain are bit-exact primitives. Step 4 (swiglu) is
   bounded at 2.64e-07. There is now NO remaining unknown primitive in the expert path.
   The full-MoE assembly kernel (T6) can be built entirely from proven parts.

## T6 IS FULLY SPECIFIED AND UNBLOCKED — turnkey plan for whoever resumes (02:47)

The reference expert-forward is coli_v4_expert_forward_ref (c/deepseek_v4.c:6111) and the
parity_v4 harness reproduces it step-by-step. Exact chain, per (token, rank):
  1. fp8_qdq(input, block=hidden)                    -> qdq            [BIT-EXACT kernel exists]
  2. matmul_mxfp4(gate, qdq, W_gate)                                   [BIT-EXACT kernel exists]
     matmul_mxfp4(up,   qdq, W_up)
  3. bf16_round_array(gate); bf16_round_array(up)                      [primitive exists, needs
                                                                        an array entry point]
  4. swiglu(activated, gate, up, limit)              [BOUNDED 2.64e-07, the only non-exact link]
  5. weighted[i] = bf16_round(activated[i] * route_weight)
     fp8_qdq(weighted, block)                        -> down_input     [BIT-EXACT kernel exists]
  6. matmul_mxfp4(down, down_input, W_down)                            [BIT-EXACT kernel exists]
  7. bf16_round_array(down)                          -> output         [BIT-EXACT]

=> Every kernel T6 needs EXISTS and is proven, except one trivial addition: a
   coli_v4_bf16_round_array entry wrapping the already-exact coli_v4_bf16_round.

HOW TO VALIDATE T6 WITHOUT MODEL LOAD (the important part):
Do NOT run the full chain end-to-end and hash the final output — swiglu's 2.64e-07 propagates
through steps 5-7 and every downstream hash will differ for a NON-bug reason. Instead ISOLATE:
feed each step's CPU golden INPUT into the corresponding GPU kernel and compare to that step's
CPU golden OUTPUT. Then:
    steps 1,2,3,5(qdq),6,7  MUST be bit-exact (fnv64 hash match)
    step 4 (swiglu)         MUST be within the documented bound (<=2 ULP transcendental,
                            plus the denormal-flush class counted separately)
parity_v4.m already dumps per-step golden hex in full mode; that dump is the fixture source.
This gives a rigorous, model-free, fan-quiet correctness proof of the entire expert path.

WHAT T6 does NOT unblock (still needs the user):
  - Live-engine wiring at the GPU seam (deepseek_v4.c:3763-3768) is gated on the slot->slab
    reuse hazard (design doc :160-165). That hazard is about the LIVE serve path, NOT the
    standalone kernel, so T6-against-golden can proceed without touching it.
  - Q1 (is bounded-tolerance acceptable as the production default, or must CPU stay
    authoritative?) decides whether a bit-exact expf hand-port is ever worth attempting.
  - Benchmark execution is gated on the user's morning approval + real model.

## ============================================================
## T6 CAPSTONE (02:53) — FULL EXPERT FORWARD IS BIT-EXACT END-TO-END ON GPU
## ============================================================
Files: validation/metal/probe_expert.m, c/metal/_moe_combined.metal (runtime-combined source),
       new kernel coli_v4_bf16_round_array in coli_v4_decode.metal.
Repro: for s in 1 7 42 100 24601 ...; do ./validation/metal/probe_expert $s; done

METHOD: full single-expert forward (hidden=64, intermediate=128), the exact 7-step parity_v4
sequence, run as an INDEPENDENT chain on CPU and on GPU from the same input, compared per step
and at the output. Fixtures self-generated (proven-bit-exact matmul RNG), NO model, fan-quiet.

HEADLINE RESULT, 12/12 seeds, both clamp modes (lim=0 and 7), route weights 0.257..1.183:
    step1 qdq      BIT-EXACT
    step2 gate/up  BIT-EXACT
    step3 bf16      BIT-EXACT
    step4 swiglu    diverges 9/128, maxRel ~1.1e-07   (the known transcendental, in isolation)
    step5 requant + step6 down + step7 out:  BIT-EXACT
    END-TO-END EXPERT OUTPUT: diff = 0/64, maxRel = 0.000e+00, on ALL 12 inputs.

WHY (mechanism, verified not assumed): swiglu's output is immediately re-quantized to FP8 E4M3
(step 5) before the down matmul. The swiglu transcendental error (~1e-7 relative) is far below
the E4M3 quantization step, so both CPU and GPU swiglu outputs map to the SAME FP8 codes. The
requantization ABSORBS the transcendental divergence. Everything downstream is then bit-exact
because every remaining op (fp8 qdq, matmul_mxfp4, bf16) is bit-exact.

HONEST BOUNDARY (do not overclaim): this is EMPIRICAL bit-exactness across 12 varied inputs
with a clear mechanism, NOT a hard mathematical guarantee. A swiglu value landing exactly on an
FP8 round-to-even boundary could in principle flip a single code and produce a 1-FP8-ULP output
difference. Across 12 seeds x 128 intermediate lanes that never occurred. The correct claim is:
"the expert forward is bit-exact end-to-end in practice; the only residual risk is a
1-FP8-code flip at an exact rounding boundary, which is itself far below the model's FP4
weight-quantization error." Real-model runs tomorrow are the final confirmation.

## THIS RESOLVES Q1 (the transcendental tolerance question)
Q1 asked whether bounded-tolerance parity is acceptable, or whether CPU must stay authoritative.
ANSWER FROM MEASUREMENT: the bounded swiglu error DOES NOT REACH the expert output at all — the
FP8 requantization erases it. So there is no end-to-end tolerance to accept in the common case;
the GPU expert is bit-identical to CPU. A hand-ported bit-exact expf is therefore NOT WORTH
BUILDING — it would defend against a divergence that is already absorbed. Recommend: ship the
ordered path as-is, label it "bit-exact end-to-end (empirical, 12/12), 1-FP8-code worst-case
residual at rounding boundaries", and confirm on the real model tomorrow.
Q1 is downgraded from "blocking decision" to "confirm on real model".

## SUITE STATE (02:53): 8/8 GREEN
  probe_primitives, probe_matmul, probe_swiglu, probe_fp8qdq, probe_fp4_runtime,
  probe_reduction_parity, parity_v4, probe_expert(x12 seeds) -- all green.
  Bug fixed en route: swiglu.metal wrapped its KERNEL inside the COLI_V4_SIGMOID_DEFINED guard
  (#endif at EOF), so any source-combine after decode.metal silently preprocessed the whole
  kernel away. Moved #endif to close right after the sigmoid inline. Standalone + combined green.

## TRANSITIVE VALIDATION (02:55) — reconciling parity-harness T3 with my T6
parity-harness's parity_v4 (T3) internally asserts: manual 7-step matmul_mxfp4 replay ==
coli_v4_expert_forward_ref (the REAL engine expert function). That assert PASSES, and its
golden is durable at validation/metal/parity_v4.golden.hex (step FNV64:
f8c32dc399b42d85 d4d9840027cff60d d350432bb53648f8 fb6e5e7dfae59a72 cc9ef7ef995fa711
283a19db50dbcd5c 09d4b630742ba5e3), O0==O2, two runs byte-equal.

My T6 (probe_expert) independently asserts: GPU 7-step chain == my own CPU 7-step chain,
bit-exact end-to-end, 12/12.

These are TWO INDEPENDENTLY-WRITTEN CPU references (parity-harness's and mine), both structured
as the same 7-step sequence, and parity-harness's is proven == the real engine function. So by
transitivity the GPU chain matches coli_v4_expert_forward_ref, not just my private reference.
That is a stronger anchor than either probe alone — the end-to-end bit-exactness is not an
artifact of one person's CPU reimplementation.
Note: probe_expert and parity_v4 use DIFFERENT fixtures (mine self-generated, theirs from
synth_tensors.h), so their golden hashes are not directly comparable; the strength is in the
independent structural agreement + the engine-function anchor, not a hash match.

## TEAM STATE (02:55): all 4 members closure-ready, Wave 1 COMPLETE
  T1 kernel-primitives  DONE  (5 primitives, 4 bit-exact + sigmoid bounded)
  T2 build-wiring       DONE  (METAL=1, 26 objs unperturbed SHA e7b077..98ca8, host-only guard)
  T3 parity-harness     DONE  (CPU golden, 7 steps, engine-anchored, durable .golden.hex)
  T4 bench-scaffold     DONE  (ORDERED/SIMD x COLD/HOT + CPU, GPU-timestamped, gated OFF)
  T5a/b/c + T6 (lead)   DONE  (matmul/swiglu/fp8qdq + full expert BIT-EXACT end-to-end 12/12)
Suite 8/8 green. 0 commits. Wave 2 (rows16 hot, host glue, LIVE seam) is user-gated on the
slot->slab reuse hazard + morning benchmark approval.

## ROWS16 HOT LAYOUT VALIDATED (02:58)
Files: coli_v4_matmul_mxfp4_ordered_hot in c/metal/coli_v4_matmul.metal; validation/metal/probe_hot.m
Repro: ./validation/metal/probe_hot
Layout (from synth_v4_rows16_pack): packed[(tile*stride+col)*16+lane] = data[row*stride+col],
tile=row/16, lane=row%16, applied to BOTH weights (stride=rb) and scales (stride=ng).

RESULT: hot == cold == cpu, BIT-IDENTICAL on all shapes:
  O=32 (2 tiles), O=48 (3 tiles), O=16 (tail16 I), O=17 (RAGGED tile) -- all bit-identical.

CRITICAL STORAGE RULE (learned via a SIGTRAP): rows16 requires output rows PADDED to a
multiple of 16. The packed buffer is ((O+15)/16)*16 * stride, not O*stride. pack16 and the GPU
kernel both index full 16-lane tiles, so a ragged O writes/reads the padding lanes. This is
exactly why synth_tensors.h uses SYNTH_V4_STORAGE_HIDDEN=128 (padded from HIDDEN=64). Any host
glue that allocates hot weight/scale storage MUST pad O to a multiple of 16 and zero the pad.
My first probe allocated O*rb and crashed with Trace/BPT trap 5 (heap OOB in pack). Fixed.

TRANSITIVE COVERAGE: the ONLY layout-sensitive step in the expert chain is the weight matmul
(gate/up/down). fp8_qdq, swiglu, bf16 all operate on ACTIVATIONS and are layout-agnostic. So
proving hot==cold for matmul_mxfp4 covers the full hot MoE path's correctness; a fused hot
kernel would differ from the cold fused kernel only in the weight addressing just validated.

## CORRECTNESS FOUNDATION COMPLETE (02:58) — 9/9 probes green
  probe_primitives  probe_matmul  probe_swiglu  probe_fp8qdq  probe_hot
  probe_fp4_runtime  probe_reduction_parity  parity_v4  probe_expert(x12)
Every V4 MoE arithmetic path proven on GPU: primitives bit-exact (sigmoid bounded, absorbed),
matmul cold+hot bit-identical, full expert forward bit-exact end-to-end. What remains is
PACKAGING (fuse the 7 steps into the 4 named single-dispatch kernels for benchmarking) and the
LIVE seam -- both downstream of the user's morning decisions, neither a correctness unknown.

## ================= AFTERNOON (12:10) — REAL-DIM RESULTS + A RETRACTION =================
Gate lifted by user greeting. Real model confirmed at models/deepseek-v4-flash (48 shards).
REAL DIMS: hidden=4096, moe_intermediate=2048, n_routed_experts=256, top-k=6, layers=43,
swiglu_limit=10.0. Device threadgroup memory = 32 KB, maxThreadsPerTG = 1024.

### PERF, real shapes, best-of-5, CPU = 16-thread OpenMP (validation/metal/bench_matmul.m)
  shape                      CPU ms   ordered   hot      simd     best speedup
  gate/up S=1 4096->2048     0.418    1.618     0.720    0.075    5.55x
  down    S=1 2048->4096     0.383    0.299     0.421    0.044    8.72x
  gate/up S=8 4096->2048     2.941    0.758     1.088    0.097   30.28x
  ordered == CPU BIT-EXACT on ALL THREE real shapes (confirms the night's result at production scale).
  NOTE: ordered at S=1 gate/up is SLOWER than CPU (1.618 vs 0.418). hot is NOT a consistent win
  (better than ordered at S=1 gate/up, worse on the other two).

### RETRACTION — "simd is bit-exact end-to-end (12/12)" WAS A FALSE GREEN
I reported that at 12:09 from probe_expert with hidden=64/intermediate=128. It DOES NOT
generalize. Measured at real dims (/tmp/absorb.m):
  gate/up 4096->2048 : simd raw diff 1634/2048, maxRel 4.034e-04
                       after bf16 -> 1/2048 SURVIVES ; after fp8 -> 0/2048 absorbed
  down    2048->4096 : simd raw diff 3149/4096, maxRel 2.642e-04
                       after bf16 -> 2/4096 SURVIVES ; after fp8 -> 0/4096 absorbed
ROOT CAUSE: simd_sum error grows with REDUCTION LENGTH. Tiny fixture reduced over 64 elements;
real reduces over 4096 -> 64x longer -> error 7.4e-06 becomes ~4e-04, ~54x worse. The tiny
fixture was structurally incapable of exposing this. Classic undersized-fixture false green.
CONSEQUENCE: the absorption argument is only HALF true. FP8 requantize (step 5, ~6% granularity)
absorbs simd error fully. bf16 (steps 3 and 7, ~0.2% granularity) does NOT. Step 7 is the FINAL
OUTPUT bf16 round, so ~2/4096 outputs differ by 1 bf16 ULP.
=> simd is NOT bit-exact end-to-end at production scale. ordered IS (it reproduces CPU
   accumulation order irrespective of reduction length; verified bit-exact at all real shapes).

### THE REAL TRADEOFF (this is the user's decision, restated with numbers)
  ordered : bit-exact end-to-end at real dims. Perf mixed -- 3.9x on S=8 gate/up, 1.3x on down,
            but 0.26x (SLOWER than CPU) on S=1 gate/up.
  simd    : 5.5x - 30.3x faster. ~2/4096 final values differ by 1 bf16 ULP (~2.6e-04 pre-round).
Q1 is REOPENED in modified form: it is no longer "is a 1e-7 transcendental acceptable" (that one
is genuinely absorbed) but "is 1-bf16-ULP on ~0.05% of outputs acceptable for a 5-30x speedup".
For context: FP4 weight quantization already carries ~6e-2 error, ~230x larger than simd's 2.6e-04.

### METHOD LESSON (recorded so it is not repeated)
Numerical fixtures MUST match production reduction lengths. Correctness-at-tiny-dims does not
imply correctness-at-real-dims for anything involving accumulation. Every future parity claim in
this project must state the reduction length it was measured at.

## PREWARM A/B RE-RUN IN PROGRESS (12:34, afternoon, gate lifted)
Launched V4RAM=64 ./validation/dsv4/prewarm_ab.sh (bg, log validation/dsv4/prewarm_ab.run.log).
VALIDITY GATES verified before launch: /tmp snapshot survived (29127 B, host no reboot),
c/coli resolves engine to c/deepseek_v4 (CURRENT, sha 7406b1fd) NOT stale libexec (3e34b4ed) --
stale-libexec hazard real but structurally avoided. Both arms seed from frozen 14190-selection
snapshot, COLI_V4_SAVE_USAGE=0. colibri-lab:0.0 pane was idle. arm A COMPLETE.csv backed up.
MACHINE-STATE NOTE: this re-run's arm 0 p1 = 946s / 0.1353 tok/s vs frozen arm A p1 = 712s /
0.1798. ~25% slower because today free RAM is ~1.6 GB (my session + metal artifacts) vs the
night's headroom. Compressor flat 2.5 GB so GATE IS CLEAN and data is valid. This is PRECISELY
why the task mandated re-running BOTH arms on ONE machine state: the valid comparison is
today-arm0 vs today-arm1, NOT arm1 vs the frozen arm A. The verdict must use today's pair.

## ============ USER DECISIONS (13:25, binding) ============
D1 Metal numeric default : DEFER -- decide AFTER fixing ordered's batch-1 occupancy.
D2 slot->slab hazard     : SYNCHRONOUS V1. GPU completes before any slot refill. Correctness
                           first; optimize overlap later.
D3 libexec refresh       : AFTER the PREWARM verdict is written. (Binary stays stable across
                           the whole A/B, then sync c/deepseek_v4 -> libexec/colibri/.)
D4 rows16 hot layout     : KEEP BOTH, select per-shape at runtime.
D5 next priority         : Build the 4 FUSED MoE kernels, then full-model benchmark.
D6 benchmark sweep       : SHORT -- 4 prompts x 128 tok, matching the PREWARM harness, so
                           results are directly comparable to existing RESULTS.md V4 data.

## D1 ANALYSIS — WHY ordered IS SLOW AT BATCH-1, AND THE FIX (diagnosed from existing data)
Measured: batch-1 gate/up (I=4096,O=2048) CPU 0.418 ms, ordered COLD 1.618 ms, ordered HOT 0.720 ms.
Weight matrix is 2048x2048/2 = ~4 MB. CPU achieves ~10 GB/s effective; ordered cold ~2.5 GB/s.
GPU LOSING on bandwidth => this is NOT compute-bound, it is a memory-access-pattern problem.

CAUSE 1 (coalescing): in the COLD kernel thread o reads w[o*rb + ...], so adjacent threads read
addresses rb=2048 B apart -- worst-case uncoalesced. This is exactly what rows16 fixes, and the
data agrees: hot 0.720 vs cold 1.618 = 2.2x. Confirms D4 (keep hot; it is the batch-1 win).

CAUSE 2 (redundant x traffic) -- NOT YET ADDRESSED, and it is the big one:
every thread reads the ENTIRE input vector x (4096 floats = 16 KB) from device memory.
With O=2048 threads that is 2048 x 16 KB = ~32 MB of redundant reads for a 4 MB weight matrix.
x is 8x larger in traffic than the weights themselves.
FIX: stage x in THREADGROUP memory once per threadgroup (16 KB of the 32 KB budget, verified
maxThreadgroupMemoryLength=32768), then every thread reads x from fast on-chip memory.
CRITICALLY: this preserves BIT-EXACTNESS perfectly -- identical values, identical accumulation
order. It is purely a data-placement change, not a math change.

CONSTRAINT that bounds how far ordered can go (important for D1):
bit-exactness FORBIDS splitting a row's reduction across threads, because CPU accumulates
ga serially within a group and then a += ga*sc serially across groups. Any split changes the
association and therefore the rounding. So ordered's thread count is capped at O (=2048),
and parallelism cannot be raised further without abandoning exactness. The x-staging fix works
WITHIN that cap by cutting memory traffic ~8x rather than adding threads.
=> If ordered+hot+x-staging still loses to CPU at batch-1, that is a REAL ceiling, not a bug,
   and D1 should then resolve in favour of simd for decode-heavy workloads.

## PREWARM A/B RELAUNCH (16:41) — new binary, both arms on one machine state
Engine sha f84813a9 (was 7406b1fd this morning). The Metal work changed the CPU path too:
  - seam guards at 3 sites in deepseek_v4.c
  - expert slab posix_memalign 4096 -> 16384 (affects CPU memcpy/SIMD alignment as well)
=> This morning's ram64 numbers (arm0 mean 0.1595, arm1 p1 0.1544) are NOT a valid comparator.
   Both arms of THIS run use sha f84813a9, so the within-run paired delta is valid.
Metal confirmed OFF by default (metal_dispatches=0) - PREWARM is a CPU cache-policy experiment.
Pre-flight: free 66.0 + inactive 31.8 GB, compressor 2.5 GB, load 4.43, history 29127 B.

## PREWARM ram64 RELAUNCH ABORTED — GATE FAIL, ambient contamination (17:11)
arm0 #1 ok (901s, 0.1421). arm0 #2 GATE FAIL: compressor 2.4 -> 25.0 GB (limit 20), gap_gb=20.1
=> 20 GB of the engine was compressed. Data DISCARDED, preserved as .GATEFAIL.*
CAUSE (measured, not my Metal work - no language servers were running):
  non-engine RSS 43-44 GB (qemu 4.3, 3x opencode 5.0, ~7+ GB across a dozen Chrome procs, claude 0.5)
  --ram 64 => 64 + 44 = 108 GB against the ~100 GB ceiling (RESULTS.md S11). Cannot fit.
  S11 says ram64 is safe at "normal desktop load of 18-31 GB"; load is now 43-44 GB, outside it.
DECISION: relaunch at --ram 48 (48 + 44 = 92 GB, 8 GB margin). --ram 56 lands exactly on 100.0
  with zero margin, which is how this run just failed. Did NOT kill the user's Chrome/qemu.
CAVEAT TO CARRY INTO THE VERDICT: ram48 has fewer expert slots than ram64 (ram96=143,
  ram64=104, ram48~78 target_slots), so PREWARM has fewer slots to warm. A NULL result at
  ram48 may be a FALSE NEGATIVE for the effect at larger budgets. The earlier ram64 arm1
  partial did show +14.1% on the cold prompt, so the effect is real at 64. State this
  limitation explicitly in the S7 justify/kill write-up; do not kill PREWARM on ram48 alone.
  The hit-rate kill-condition remains evaluable regardless of budget.

## PREWARM ram64 LAUNCH #3 (17:24) — headroom restored by user
User closed desktop apps. non-engine RSS 43-44 GB -> 21.9 GB. free 96.0, inactive 23.0, compressor 2.6.
Budget check vs the ~100 GB ceiling:
  ram96 117.9 COMPRESS | ram80 101.9 COMPRESS | ram72 93.9 fits(6) | ram64 85.9 FITS(14) | ram56 77.9 fits(22)
CHOSE --ram 64: the originally specified budget, 14 GB margin, and non-engine 21.9 GB is inside
the 18-31 GB band S11 says keeps ram64 uncompressed. Also above the retention threshold, so
PREWARM gets full slots (target_slots=104) - avoids the ram48 false-negative risk.
Superseded runs preserved: prewarm_off_ram64.GATEFAIL.csv (contaminated), prewarm_*_ram48.PARTIAL.*
