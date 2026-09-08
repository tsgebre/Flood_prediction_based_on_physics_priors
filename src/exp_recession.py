"""Anchored-recession experiment.

Tests whether pinning dry-day discharge decay to the train-fitted master
recession constant helps the storage/baseflow-dominated target basins,
01466500 (groundwater/baseflow) and 11143000 (Mediterranean). Three arms x
N seeds under 100% discharge-sensor failure:

* control              -- no physics
* physics              -- router physics, recession band OFF
* physics_recession    -- router physics + anchored recession-rate band

    python -m src.exp_recession --seeds 0 1 2 --loss nse --hidden 32 --epochs 20
"""

import argparse
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
TARGETS = ["01466500", "11143000"]


def _spec_no_recession(spec):
    w = {k: v for k, v in spec.weights.items() if k != "recession_rate"}
    return replace(spec, force_decay=False, weights=w)


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
    ap.add_argument("--outdir", default="results/recession")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    hp = {"hidden": args.hidden, "layers": 1, "lr": 0.0018, "epochs": args.epochs, "patience": 8}
    outdir = os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))

    print(f"Recession-band validation | basins={args.basins} | seeds={args.seeds} | loss={args.loss}",
          flush=True)

    results = []
    for g in args.basins:
        attrs = A.compute_attributes(g, DATA)
        spec = router.route_physics(attrs, enable_recession=True)
        spec_norec = _spec_no_recession(spec)
        print(f"\n[{g}] {spec.regime} | fs={attrs['frac_snow']:.2f} bfi={attrs['baseflow_index']:.2f} "
              f"| rec_w={spec.weights.get('recession_rate')} force_decay={spec.force_decay}", flush=True)

        arms = {}
        for name, use_phys, sp in [("control", False, spec_norec),
                                   ("physics", True, spec_norec),
                                   ("physics_recession", True, spec)]:
            seeds_out = []
            for s in args.seeds:
                seeds_out.append(MB.train_arm(g, use_phys, sp, hp, args.seq_len, device, s,
                                              args.train_mask_hi, args.mask_grid, args.loss))
                print(f"    {name:<18} seed {s} done", flush=True)
            arms[name] = seeds_out
        results.append({"gauge": g, "regime": spec.regime, "frac_snow": attrs["frac_snow"],
                        "baseflow_index": attrs["baseflow_index"], "arms": arms})
        json.dump(results, open(os.path.join(outdir, "recession_results.json"), "w"), indent=2, default=float)

    # ---- report ----
    print("\n" + "=" * 96)
    print(f"ANCHORED RECESSION-RATE BAND @ {int(hm*100)}% missing discharge ({len(args.seeds)} seeds, mean±std)")
    print("-" * 96)
    print(f"{'gauge':<10} {'arm':<20} {'NSE':>16} {'flood-MAE':>16}")
    with open(os.path.join(outdir, "recession_gain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge", "arm", "mask", "NSE_mean", "NSE_std", "floodMAE_mean", "floodMAE_std"])
        for rec in results:
            nse = {a: _vals(rec["arms"][a], hm, "NSE") for a in rec["arms"]}
            fme = {a: _vals(rec["arms"][a], hm, "MAE_extreme") for a in rec["arms"]}
            for a in ["control", "physics", "physics_recession"]:
                print(f"{rec['gauge']:<10} {a:<20} "
                      f"{np.nanmean(nse[a]):>7.3f} ± {np.nanstd(nse[a]):<6.3f} "
                      f"{np.nanmean(fme[a]):>7.1f} ± {np.nanstd(fme[a]):<6.1f}")
                for mr in args.mask_grid:
                    w.writerow([rec["gauge"], a, mr,
                                float(np.nanmean(_vals(rec["arms"][a], mr, "NSE"))),
                                float(np.nanstd(_vals(rec["arms"][a], mr, "NSE"))),
                                float(np.nanmean(_vals(rec["arms"][a], mr, "MAE_extreme"))),
                                float(np.nanstd(_vals(rec["arms"][a], mr, "MAE_extreme")))])
            try:
                from scipy import stats
                dnse = stats.ttest_rel(nse["physics_recession"], nse["physics"]).pvalue
                print(f"           dNSE (recession vs physics) = "
                      f"{np.nanmean(nse['physics_recession'] - nse['physics']):+.3f}  p={dnse:.3f}")
            except Exception as e:
                print("           (paired test skipped:", e, ")")
            print("-" * 96)
    print(f"Artifacts -> {outdir}/ (recession_results.json, recession_gain.csv)")


if __name__ == "__main__":
    main()
