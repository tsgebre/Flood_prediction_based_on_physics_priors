"""Antecedent-moisture (AMC) experiment.

Isolates the antecedent-moisture mechanism (API feature + wetness-gated
response + dry-antecedent suppression) on its two target basins, 01013500
(mixed rain-snow) and 07067000 (humid rain). Trains three arms x N seeds
under 100% discharge-sensor failure and reports NSE + flood-MAE with paired
t-tests:

* control        -- no physics (meteorology + discharge + missingness only)
* physics        -- router physics, antecedent-moisture OFF
* physics+amc    -- router physics + antecedent-moisture ON

    python -m src.exp_amc --seeds 0 1 2 --loss nse --hidden 32 --epochs 20
"""

import argparse
import copy
import csv
import json
import os
from dataclasses import replace

import numpy as np

from src import attributes as A
from src import multibasin as MB
from src import router

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
TARGETS = ["01013500", "07067000"]


def _spec_no_amc(spec):
    """Same physics prescription with the antecedent-moisture additions removed."""
    w = {k: v for k, v in spec.weights.items() if k != "dry_suppress"}
    return replace(spec, api_feature=False, weights=w)


def _at(rows, mr, metric):
    for r in rows:
        if abs(r["mask_ratio"] - mr) < 1e-9:
            return r.get(metric, float("nan"))
    return float("nan")


def _vals(seed_rows, mr, metric):
    return np.array([_at(rows, mr, metric) for rows in seed_rows], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basins", nargs="+", default=TARGETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mask-grid", type=float, nargs="+", default=[0.0, 0.8, 1.0])
    ap.add_argument("--headline-mask", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-mask-hi", type=float, default=1.0)
    ap.add_argument("--loss", choices=["mse", "nse"], default="nse")
    ap.add_argument("--outdir", default="results/amc")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    hp = {"hidden": args.hidden, "layers": 1, "lr": 0.0018, "epochs": args.epochs, "patience": 8}
    outdir = os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))

    print(f"AMC validation | basins={args.basins} | seeds={args.seeds} | loss={args.loss} "
          f"| hidden={hp['hidden']} epochs<={hp['epochs']}", flush=True)

    results = []
    for g in args.basins:
        attrs = A.compute_attributes(g, DATA)
        spec = router.route_physics(attrs, enable_amc=True)   # antecedent-moisture ON
        spec_noamc = _spec_no_amc(spec)                        # same prescription, AMC removed
        print(f"\n[{g}] {spec.regime} | fs={attrs['frac_snow']:.2f} "
              f"| api_feature={spec.api_feature} dry_w={spec.weights.get('dry_suppress')}", flush=True)

        arms = {}
        for name, use_phys, sp in [("control", False, spec_noamc),
                                   ("physics", True, spec_noamc),
                                   ("physics_amc", True, spec)]:
            seeds_out = []
            for s in args.seeds:
                seeds_out.append(MB.train_arm(g, use_phys, sp, hp, args.seq_len, device, s,
                                              args.train_mask_hi, args.mask_grid, args.loss))
                print(f"    {name:<12} seed {s} done", flush=True)
            arms[name] = seeds_out
        rec = {"gauge": g, "regime": spec.regime, "frac_snow": attrs["frac_snow"], "arms": arms}
        results.append(rec)
        json.dump(results, open(os.path.join(outdir, "amc_results.json"), "w"), indent=2, default=float)

    # ---- report ----
    print("\n" + "=" * 92)
    print(f"ANTECEDENT-MOISTURE PHYSICS @ {int(hm*100)}% missing discharge "
          f"({len(args.seeds)} seeds, mean±std)")
    print("-" * 92)
    print(f"{'gauge':<10} {'arm':<13} {'NSE':>16} {'flood-MAE':>16}   vs physics")
    with open(os.path.join(outdir, "amc_gain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge", "arm", "mask", "NSE_mean", "NSE_std", "floodMAE_mean", "floodMAE_std"])
        for rec in results:
            nse = {a: _vals(rec["arms"][a], hm, "NSE") for a in rec["arms"]}
            fme = {a: _vals(rec["arms"][a], hm, "MAE_extreme") for a in rec["arms"]}
            for a in ["control", "physics", "physics_amc"]:
                # relative flood-MAE reduction of this arm vs physics baseline
                base = fme["physics"]
                rel = 100 * np.nanmean((base - fme[a]) / np.abs(base)) if a != "physics" else 0.0
                tag = "" if a == "physics" else f"{rel:+.1f}% flood"
                print(f"{rec['gauge']:<10} {a:<13} "
                      f"{np.nanmean(nse[a]):>7.3f} ± {np.nanstd(nse[a]):<6.3f} "
                      f"{np.nanmean(fme[a]):>7.3f} ± {np.nanstd(fme[a]):<6.3f}   {tag}")
                for mr in args.mask_grid:
                    w.writerow([rec["gauge"], a, mr,
                                float(np.nanmean(_vals(rec["arms"][a], mr, "NSE"))),
                                float(np.nanstd(_vals(rec["arms"][a], mr, "NSE"))),
                                float(np.nanmean(_vals(rec["arms"][a], mr, "MAE_extreme"))),
                                float(np.nanstd(_vals(rec["arms"][a], mr, "MAE_extreme")))])
            # paired significance: physics_amc vs physics
            try:
                from scipy import stats
                dnse = stats.ttest_rel(nse["physics_amc"], nse["physics"]).pvalue
                dfme = stats.ttest_rel(fme["physics_amc"], fme["physics"]).pvalue
                print(f"           paired p (amc vs physics): dNSE p={dnse:.3f}  flood-MAE p={dfme:.3f}")
            except Exception as e:
                print("           (paired test skipped:", e, ")")
            print("-" * 92)
    print(f"Artifacts -> {outdir}/ (amc_results.json, amc_gain.csv)")


if __name__ == "__main__":
    main()
