#!/usr/bin/env python3
"""STOCK vs RECOMMENDED - ABBA analysis.

Only gate-clean rows count. Any row whose compressor breached the limit at
EITHER end is discarded per RESULTS.md S4b protocol step 7 - it measured
memory pressure, not configuration.
"""
import csv, os, glob, statistics as st

D = os.path.dirname(os.path.abspath(__file__))
ARMS = [("A1 stock", 48, "_stockA1"), ("B1 reco", 96, "_recoB1"),
        ("B2 reco", 96, "_recoB2"), ("A2 stock", 48, "_stockA2")]

def rows(ram, tag, kind):
    f = os.path.join(D, f"coldwarm_{kind}_ram{ram}{tag}.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []

def clean(rs):
    return [r for r in rs if r.get("gate") == "ok" and float(r.get("tok_s", 0)) > 0]

print("\n" + "=" * 76)
print("  STOCK (--ram 48, no speculation)  vs  RECOMMENDED (--ram 96, n-gram)")
print("=" * 76)

data = {}
for label, ram, tag in ARMS:
    for kind in ("cold", "warm"):
        rs = rows(ram, tag, kind)
        ok = clean(rs)
        data[(label, kind)] = ok
        if not rs:
            print(f"  {label:<10} {kind:<5} : no data yet")
            continue
        bad = len(rs) - len(ok)
        m = st.mean(float(r["tok_s"]) for r in ok) if ok else 0.0
        comp = max((float(r["comp_end_gb"]) for r in rs), default=0)
        flag = f"  !! {bad} row(s) DISCARDED (gate)" if bad else ""
        print(f"  {label:<10} {kind:<5} : mean {m:.4f} tok/s  n={len(ok)}/{len(rs)}  peak_comp={comp:.1f}GB{flag}")

def pooled(pref, kind):
    v = [float(r["tok_s"]) for (lab, k), rs in data.items()
         if k == kind and lab.split()[1] == pref for r in rs]
    return (st.mean(v), len(v)) if v else (0.0, 0)

print("\n" + "-" * 76)
print("  POOLED (both repeats of each arm, gate-clean rows only)")
print("-" * 76)
res = {}
for kind in ("cold", "warm"):
    s, ns = pooled("stock", kind)
    r, nr = pooled("reco", kind)
    if s and r:
        gain = (r - s) / s * 100
        res[kind] = gain
        print(f"  {kind:<5}: stock {s:.4f} (n={ns})  ->  reco {r:.4f} (n={nr})   = {gain:+.1f}%")
    else:
        print(f"  {kind:<5}: incomplete (stock n={ns}, reco n={nr})")

print("\n" + "=" * 76)
print("  MEASURED vs MODELLED")
print("=" * 76)
model = {"cold": 58.6, "warm": 80.7}
for kind in ("cold", "warm"):
    if kind in res:
        m, meas = model[kind], res[kind]
        print(f"  {kind:<5}: modelled {m:+.1f}%   measured {meas:+.1f}%   "
              f"delta {meas-m:+.1f}pp  ({'model HELD' if abs(meas-m)<10 else 'model MISSED'})")
    else:
        print(f"  {kind:<5}: modelled {model[kind]:+.1f}%   measured -- (incomplete)")
print()
