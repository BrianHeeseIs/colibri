#!/usr/bin/env python3
"""PREWARM A/B verdict. Paired per-prompt deltas only (RESULTS.md S8 discipline).

Kill condition from the original task: if the expert-cache HIT RATE does not improve,
the limit is PIN POLICY, not persistence -- and COLI_V4_PREWARM should be killed, not tuned.
"""
import csv, os, statistics as st, sys

ram = sys.argv[1] if len(sys.argv) > 1 else "64"
base = "validation/dsv4"
off_p = f"{base}/prewarm_off_ram{ram}.csv"
on_p  = f"{base}/prewarm_on_ram{ram}.csv"

def rows(p):
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

off, on = rows(off_p), rows(on_p)
print(f"PREWARM A/B verdict @ --ram {ram}")
print(f"  arm0 (PREWARM=0): {len(off)}/4 rows   arm1 (PREWARM=1): {len(on)}/4 rows")
if len(off) < 4 or len(on) < 4:
    print("  INCOMPLETE - refusing to compute a verdict from a partial run.")
    print("  (A partial arm1 vs a complete arm0 is exactly the confound the frozen")
    print("   snapshot exists to prevent. Wait for 4/4 in both arms.)")
    sys.exit(1)

bad = [r for r in off+on if r["gate"] != "ok"]
if bad:
    print(f"  GATE FAILURES: {len(bad)} row(s) exceeded the compressor limit -> DISCARD, rerun.")
    for r in bad: print(f"    arm{r['prewarm']} #{r['n']} comp {r['comp_start_gb']}->{r['comp_end_gb']}")
    sys.exit(2)

print("\n  paired per-prompt (same prompt, same frozen history, same binary):")
print(f"  {'#':>2} {'off tok/s':>10} {'on tok/s':>10} {'delta':>9} {'off wall':>9} {'on wall':>8}")
deltas = []
for a, b in zip(off, on):
    fo, fn = float(a["tok_s"]), float(b["tok_s"])
    d = (fn - fo) / fo * 100.0
    deltas.append(d)
    print(f"  {a['n']:>2} {fo:>10.4f} {fn:>10.4f} {d:>+8.1f}% {a['wall_s']:>9} {b['wall_s']:>8}")

mo, mn = st.mean(float(r['tok_s']) for r in off), st.mean(float(r['tok_s']) for r in on)
print(f"\n  mean tok/s : off {mo:.4f}  on {mn:.4f}  -> {(mn-mo)/mo*100:+.1f}%")
print(f"  paired mean delta {st.mean(deltas):+.1f}%"
      + (f", sd {st.stdev(deltas):.1f}pp" if len(deltas) > 1 else ""))
print(f"  cold prompt #1 only: {deltas[0]:+.1f}%   (PREWARM targets cold-start specifically)")

# KILL CONDITION: hit rate must improve, else the limit is pin policy not persistence
def hr(rs):
    v = [float(r["hit_rate"]) for r in rs if r.get("hit_rate")]
    return st.mean(v) if v else None
ho, hn = hr(off), hr(on)
print("\n  KILL-CONDITION CHECK (hit rate):")
if ho is None or hn is None:
    print("    hit_rate not captured by the harness -> cannot evaluate. Must be added before a")
    print("    justify/kill decision; throughput alone does not settle it.")
else:
    print(f"    off {ho:.3f}%  on {hn:.3f}%  -> {hn-ho:+.3f}pp")
    print("    " + ("IMPROVED: persistence is doing work; PREWARM is justifiable."
                    if hn - ho > 0.5 else
                    "NOT IMPROVED: the limit is PIN POLICY, not persistence -> KILL PREWARM."))

sig = abs(st.mean(deltas)) > 5.0
print(f"\n  VERDICT: {'PREWARM has a real effect' if sig else 'no material effect'}"
      f" ({st.mean(deltas):+.1f}% paired mean)")
print("  NOTE: absolute tok/s is not comparable across sessions (RESULTS.md S11 - the same")
print("        config moved 22% on ambient load alone). Only these paired deltas are valid.")
