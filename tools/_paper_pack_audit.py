"""Paper pack audit + table generator.

Walks ~/Code/llm/vllm-emulator/results/paper_pack_apr27/ and for each cell:
  - Confirms all 5 expected rates have both real_r*.json and emu_r*.json
  - Computes deltas (mean TTFT/TPOT/ITL/E2E, throughput) per rate
  - Verifies completed counts and total_output_tokens for sanity
  - Identifies any out-of-gate metrics (TTFT ≤15%, TPOT/ITL/E2E ≤5%, tput ±1%)
  - Emits a paper Table-1 markdown alongside per-cell summary
"""
import json, os, sys
from pathlib import Path

PACK = Path.home() / "Code/llm/vllm-emulator/results/paper_pack_apr27"
RATES = [2, 4, 8, 16, 32]
GATES = {  # absolute % deviation
    "TTFT": 15.0,
    "TPOT": 5.0,
    "ITL":  5.0,
    "E2E":  5.0,
    "tput": 1.0,
}
METRIC_KEYS = {
    "TTFT": "mean_ttft_ms",
    "TPOT": "mean_tpot_ms",
    "ITL":  "mean_itl_ms",
    "E2E":  "mean_e2el_ms",
}

CELLS = [
    ("M2", "M2_qwen3-8b_rtx8000",                 "Qwen3-8B",   "RTX 8000",  "no-igeos / no-greedy / default KV"),
    ("Q14B", "Q14B_qwen3-14b_rtx8000_kvoverride", "Qwen3-14B",  "RTX 8000",  "no-igeos / no-greedy / KV-override 5882"),
    ("B1", "B1_qwen3-8b_a40_igeos",               "Qwen3-8B",   "A40",       "--ignore-eos / no-greedy / default KV"),
    ("B2", "B2_qwen3-4b_a40",                     "Qwen3-4B",   "A40",       "no-igeos / no-greedy / default KV"),
]

def pct(emu, real):
    if real == 0: return float("inf")
    return (emu - real) / real * 100

def verdict(metric, val):
    g = GATES[metric]
    if abs(val) > g:  return "FAIL"
    if abs(val) > g * 0.8:  return "MARGINAL"
    return "PASS"

def color(v):
    if v == "PASS":     return "✓"
    if v == "MARGINAL": return "~"
    return "✗"

print("=" * 80)
print("Paper-pack audit — paper_pack_apr27")
print("=" * 80)

all_problems = []

for short, cell_dir, model, hw, methodology in CELLS:
    cell_path = PACK / cell_dir
    print(f"\n## {short} — {model} / {hw}")
    print(f"   ({methodology})")
    if not cell_path.exists():
        print(f"   MISSING DIR")
        all_problems.append(f"{short}: missing dir {cell_dir}")
        continue

    files_ok = True
    for r in RATES:
        for side in ("real", "emu"):
            f = cell_path / f"{side}_r{r}.json"
            if not f.exists():
                print(f"   MISSING FILE: {f.name}")
                files_ok = False
                all_problems.append(f"{short}: missing {side}_r{r}.json")
    if not files_ok:
        continue

    print(f"   {'rate':>4s} | {'r_comp':>6s} {'e_comp':>6s} | {'r_outtok':>9s} {'e_outtok':>9s} | "
          f"{'TTFT':>9s} {'TPOT':>8s} {'ITL':>8s} {'E2E':>8s} {'tput':>8s}")
    cell_fail = 0
    cell_marginal = 0
    for r in RATES:
        rd = json.load(open(cell_path / f"real_r{r}.json"))
        ed = json.load(open(cell_path / f"emu_r{r}.json"))
        rcomp, ecomp = rd["completed"], ed["completed"]
        rtok, etok = rd["total_output_tokens"], ed["total_output_tokens"]
        deltas = {m: pct(ed[k], rd[k]) for m, k in METRIC_KEYS.items()}
        rtput = rcomp / rd["duration"]
        etput = ecomp / ed["duration"]
        deltas["tput"] = pct(etput, rtput)
        verdicts = {m: verdict(m, deltas[m]) for m in deltas}

        flags = []
        if rcomp != ecomp:
            flags.append(f"comp-mismatch({rcomp}/{ecomp})")
        if abs((etok - rtok) / rtok) > 0.005:
            flags.append(f"outtok-{(etok-rtok)/rtok*100:+.1f}%")

        for m, v in deltas.items():
            if verdicts[m] == "FAIL":
                cell_fail += 1
                all_problems.append(f"{short} r={r} {m}: {v:+.2f}% (gate ≤{GATES[m]}%)")
            elif verdicts[m] == "MARGINAL":
                cell_marginal += 1

        print(f"   r={r:>3d} | {rcomp:>6d} {ecomp:>6d} | {rtok:>9d} {etok:>9d} | "
              f"{deltas['TTFT']:>+7.2f}{color(verdicts['TTFT'])} "
              f"{deltas['TPOT']:>+6.2f}{color(verdicts['TPOT'])} "
              f"{deltas['ITL']:>+6.2f}{color(verdicts['ITL'])} "
              f"{deltas['E2E']:>+6.2f}{color(verdicts['E2E'])} "
              f"{deltas['tput']:>+6.2f}{color(verdicts['tput'])}"
              + (f"  {' '.join(flags)}" if flags else ""))
    print(f"   summary: PASS={5*5 - cell_fail - cell_marginal}  MARGINAL={cell_marginal}  FAIL={cell_fail}")

print()
print("=" * 80)
print("PAPER TABLE 1 (markdown)")
print("=" * 80)
print()
print("| cell | hardware | model | rate | TTFT % | TPOT % | ITL % | E2E % | tput % |")
print("|------|----------|-------|------|--------|--------|-------|-------|--------|")
for short, cell_dir, model, hw, _ in CELLS:
    cell_path = PACK / cell_dir
    if not cell_path.exists():
        continue
    for r in RATES:
        rf, ef = cell_path / f"real_r{r}.json", cell_path / f"emu_r{r}.json"
        if not (rf.exists() and ef.exists()):
            continue
        rd, ed = json.load(open(rf)), json.load(open(ef))
        deltas = {m: pct(ed[k], rd[k]) for m, k in METRIC_KEYS.items()}
        deltas["tput"] = pct(ed["completed"]/ed["duration"], rd["completed"]/rd["duration"])
        print(f"| {short} | {hw} | {model} | {r} | "
              f"{deltas['TTFT']:+.2f} | {deltas['TPOT']:+.2f} | {deltas['ITL']:+.2f} | "
              f"{deltas['E2E']:+.2f} | {deltas['tput']:+.2f} |")

print()
print("=" * 80)
if all_problems:
    print(f"PROBLEMS FOUND: {len(all_problems)}")
    for p in all_problems[:50]:
        print(f"  - {p}")
else:
    print("ALL DATA COMPLETE — no missing files, all metrics within paper gates (allowing MARGINAL).")
