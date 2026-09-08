"""Render a results.json (from `main.py`) into readable tables and a paired
significance analysis of each model's gain over the first (baseline) model.

    python scripts/analyze_results.py <run_dir>/results.json
"""
import json
import sys

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HIGHER_BETTER = ("NSE", "KGE")
METRICS = ("NSE", "KGE", "MAE", "RMSE", "MAE_extreme")


def load(path):
    d = json.load(open(path))
    studies = d["studies"]
    names = list(studies.keys())
    ratios = sorted(float(k) for k in studies[names[0]]["sweep"].keys())
    return studies, names, ratios, d.get("persistence", {})


def vals(studies, name, mr, metric):
    md = studies[name]["sweep"][str(mr)]
    return np.array(md[metric][2], float) if metric in md else np.array([np.nan])


def print_tables(studies, names, ratios):
    for metric in METRICS:
        arrow = "↑" if metric in HIGHER_BETTER else "↓"
        print(f"\n{metric} {arrow}")
        print("  mask% | " + " | ".join(f"{n:>18}" for n in names))
        for mr in ratios:
            cells = []
            for n in names:
                v = vals(studies, n, mr, metric)
                fmt = f"{v.mean():.3f}" if metric in HIGHER_BETTER else f"{v.mean():.1f}"
                cells.append(f"{fmt:>18}")
            print(f"  {int(mr*100):4d}% | " + " | ".join(cells))


def print_gains(studies, names, ratios):
    base = names[0]
    for other in names[1:]:
        print(f"\n=== Gain: {other}  vs  {base}  (paired across seeds) ===")
        for metric in METRICS:
            arrow = "↑" if metric in HIGHER_BETTER else "↓"
            print(f"  {metric} {arrow}")
            for mr in ratios:
                b = vals(studies, base, mr, metric)
                o = vals(studies, other, mr, metric)
                higher = metric in HIGHER_BETTER
                improve = (o.mean() - b.mean()) if higher else (b.mean() - o.mean())
                rel = 100 * improve / abs(b.mean()) if b.mean() != 0 else float("nan")
                p = float("nan")
                if HAVE_SCIPY and len(b) >= 2 and not np.allclose(b - o, (b - o)[0]):
                    p = stats.ttest_rel(b, o).pvalue
                star = "*" if p < 0.05 else " "
                print(f"    {int(mr*100):4d}%: {improve:+9.3f} ({rel:+6.1f}%)  p={p:.3f}{star}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    studies, names, ratios, persist = load(path)
    print(f"Studies: {names}\nMask grid: {[int(r*100) for r in ratios]}%")
    if persist:
        print(f"Persistence (0% missing): NSE {persist.get('NSE', float('nan')):.4f}, "
              f"MAE {persist.get('MAE', float('nan')):.1f} cfs")
    print("\n" + "=" * 72 + "\nROBUSTNESS TABLES (mean over seeds)\n" + "=" * 72)
    print_tables(studies, names, ratios)
    print("\n" + "=" * 72 + "\nPAIRED GAINS  (* = p < 0.05)\n" + "=" * 72)
    print_gains(studies, names, ratios)


if __name__ == "__main__":
    main()
