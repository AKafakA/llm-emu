"""Compare any experiment dir vs Phase 1 real (random 256/128, 2000p, seed=0).

Usage:
    EMU=validate-exp1-newdefault python3 tools/compare_exp.py
    EMU=validate-exp3-newknn     python3 tools/compare_exp.py
"""
import json, os, sys
from pathlib import Path

REAL = Path("results/clean-5rate-default")
EMU_DIR = Path("results") / os.environ.get("EMU", "validate-exp1-newdefault")
RATES = [2, 4, 8, 16, 32]

METRICS = [
    ("TTFT", "median_ttft_ms", 10.0),
    ("TPOT", "median_tpot_ms", 6.0),
    ("ITL",  "median_itl_ms",  10.0),
    ("E2E",  "median_e2el_ms", 6.0),
]

print(f"=== {EMU_DIR.name} vs Phase 1 real ===")
pass_counts = {m: 0 for m, _, _ in METRICS}
fail_cells = []
for r in RATES:
    real_p = REAL / f"real_r{r}.json"
    emu_p = EMU_DIR / f"hookOn_r{r}.json"
    if not real_p.exists() or not emu_p.exists():
        print(f"r={r}: MISSING")
        continue
    real = json.load(open(real_p))
    emu = json.load(open(emu_p))
    row = f"r={r:>2}  "
    for m, key, thr in METRICS:
        rv = real[key]
        ev = emu[key]
        pct = (ev - rv) / rv * 100 if rv > 0 else float("nan")
        ok = abs(pct) <= thr
        pass_counts[m] += 1 if ok else 0
        mark = "✓" if ok else "✗"
        row += f"{m}:{pct:>+6.2f}%{mark}  "
        if not ok:
            fail_cells.append((r, m, pct, thr))
    print(row)

print()
print("pass counts:")
for m, _, thr in METRICS:
    print(f"  {m} (|Δ|≤{thr:.0f}%): {pass_counts[m]}/5")

if fail_cells:
    print()
    print("failures:")
    for r, m, pct, thr in fail_cells:
        print(f"  r={r} {m}: {pct:+.2f}% (target ≤{thr:.0f}%)")
