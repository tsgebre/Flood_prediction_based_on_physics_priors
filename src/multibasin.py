"""Regime-adaptive physics study across multiple CAMELS basins.

For each basin in ``data/`` it computes static attributes (:mod:`src.attributes`),
routes them to a per-basin physics prescription (:mod:`src.router`), then trains and
stress-tests two models under simulated discharge-sensor failure, across one or more
random seeds:

* **Control**        — meteorology + discharge + missingness channel, no physics.
* **Router-physics** — the same, plus the degree-day snow features and the
  attribute-gated physics loss the router selected for that basin.

With several seeds it reports the per-basin gain as mean ± std with a paired t-test.

    python -m src.multibasin --quick
    python -m src.multibasin --basins 01013500 09210500 09352900 11266500 --seeds 0 1 2 3
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import attributes as A
from src import dataset, models, router, train, utils

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
HIGHER = ("NSE", "KGE")


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_loaders(tr, va, te, meta, seq_len, seed, train_mask_hi):
    g = torch.Generator().manual_seed(seed)
    tds = dataset.CamelsDataset(tr, meta["feature_cols"], seq_len, is_train=True,
                                train_mask_range=(0.0, train_mask_hi), generator=g)
    vds = dataset.CamelsDataset(va, meta["feature_cols"], seq_len)
    eds = dataset.CamelsDataset(te, meta["feature_cols"], seq_len)
    dg = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(tds, batch_size=32, shuffle=True, generator=dg),
        "val": DataLoader(vds, batch_size=32),
        "test": DataLoader(eds, batch_size=32),
        "input_dim": tds.n_features,
    }


def train_arm(gauge, use_physics, spec, hp, seq_len, device, seed, train_mask_hi, mask_grid,
              loss_type="mse"):
    sp = os.path.join(DATA, f"{gauge}_streamflow_qc.txt")
    fp = os.path.join(DATA, f"{gauge}_lump_cida_forcing_leap.txt")
    tr, va, te, meta = dataset.load_and_preprocess_data(
        sp, fp,
        physics_features=(use_physics and spec.physics_features),
        api_feature=(use_physics and spec.api_feature))
    loaders = build_loaders(tr, va, te, meta, seq_len, seed, train_mask_hi)
    set_seed(seed)
    model = models.build_model("LSTM", loaders["input_dim"], hp["hidden"], hp["layers"], 1, meta)
    if use_physics:
        phys = train.PhysicsConfig(use_physics=True, mode=spec.mode,
                                   weights=dict(spec.weights), lag=spec.lag, smooth=spec.smooth,
                                   force_decay=spec.force_decay)
    else:
        phys = train.PhysicsConfig(use_physics=False)
    model, _ = train.train_model(model, loaders["train"], loaders["val"], lr=hp["lr"],
                                 epochs=hp["epochs"], patience=hp["patience"], device=device,
                                 physics=phys, loss_type=loss_type, verbose=False)
    return utils.robustness_sweep(model, loaders["test"], meta, mask_grid, seed=123, device=device)


def run_basin(gauge, hp, seq_len, device, seeds, train_mask_hi, mask_grid, loss_type="mse"):
    attrs = A.compute_attributes(gauge, DATA)
    spec = router.route_physics(attrs)
    ctrl = [train_arm(gauge, False, spec, hp, seq_len, device, s, train_mask_hi, mask_grid, loss_type) for s in seeds]
    phys = [train_arm(gauge, True, spec, hp, seq_len, device, s, train_mask_hi, mask_grid, loss_type) for s in seeds]
    return {"gauge": gauge, "regime": spec.regime, "attrs": attrs, "seeds": list(seeds),
            "spec": {"weights": spec.weights, "lag": spec.lag, "smooth": spec.smooth, "gates": spec.gates},
            "control_seeds": ctrl, "physics_seeds": phys}


def _at(rows, mr, metric):
    for r in rows:
        if abs(r["mask_ratio"] - mr) < 1e-9:
            return r.get(metric, float("nan"))
    return float("nan")


def _seed_vals(seed_rows, mr, metric):
    return np.array([_at(rows, mr, metric) for rows in seed_rows], float)


def _paired_p(a, b):
    """Two-sided paired t-test p-value; None if <2 or degenerate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a - b, (a - b)[0]):
        return None
    try:
        from scipy import stats
        return float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        return None


def basin_gain(r, mr, metric):
    """Return dict describing the physics gain for one basin at (mr, metric)."""
    c = _seed_vals(r["control_seeds"], mr, metric)
    p = _seed_vals(r["physics_seeds"], mr, metric)
    higher = metric in HIGHER
    improve = (p - c) if higher else (c - p)                 # per-seed improvement
    rel = 100 * (p - c) / np.abs(c) if higher else 100 * (c - p) / np.abs(c)
    pv = _paired_p(p, c) if higher else _paired_p(c, p)
    return {"ctrl_mean": float(np.nanmean(c)), "ctrl_std": float(np.nanstd(c)),
            "phys_mean": float(np.nanmean(p)), "phys_std": float(np.nanstd(p)),
            "improve_mean": float(np.nanmean(improve)), "improve_std": float(np.nanstd(improve)),
            "rel_mean": float(np.nanmean(rel)), "rel_std": float(np.nanstd(rel)), "p": pv}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basins", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mask-grid", type=float, nargs="+", default=[0.0, 0.8, 1.0])
    ap.add_argument("--headline-mask", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-mask-hi", type=float, default=1.0)
    ap.add_argument("--loss", choices=["mse", "nse"], default="mse")
    ap.add_argument("--outdir", default="results/regime")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    hp = {"hidden": args.hidden, "layers": 1, "lr": 0.0018, "epochs": args.epochs, "patience": 8}
    if args.quick:
        hp.update(hidden=32, epochs=4, patience=3)
    basins = args.basins or A.basins_in(DATA)
    outdir = os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))

    print(f"Regime-adaptive physics | basins={len(basins)} | seeds={args.seeds} | mask={args.mask_grid} "
          f"| hidden={hp['hidden']} epochs<={hp['epochs']}", flush=True)

    results = []
    for i, g in enumerate(basins):
        print(f"[{i+1}/{len(basins)}] {g} ...", flush=True)
        r = run_basin(g, hp, args.seq_len, device, args.seeds, args.train_mask_hi, args.mask_grid, args.loss)
        results.append(r)
        gm = basin_gain(r, hm, "MAE_extreme")
        gn = basin_gain(r, hm, "NSE")
        pstr = f" p={gm['p']:.3f}" if gm["p"] is not None else ""
        print(f"    {r['regime']:<22} @ {int(hm*100)}% | flood-MAE {gm['rel_mean']:+.1f}±{gm['rel_std']:.1f}%{pstr}"
              f" | NSE {gn['ctrl_mean']:.3f}->{gn['phys_mean']:.3f} ({gn['improve_mean']:+.3f})", flush=True)
        json.dump(results, open(os.path.join(outdir, "regime_results.json"), "w"), indent=2, default=float)

    # ---- summary ----
    ns = len(args.seeds)
    print("\n" + "=" * 104)
    print(f"REGIME-ADAPTIVE PHYSICS — flood-peak error reduction at {int(hm*100)}% missing "
          f"({ns} seed{'s' if ns > 1 else ''}, mean±std)")
    print("-" * 104)
    print(f"{'gauge':<10} {'regime':<22} {'frac_snow':>9} {'flood-MAE reduction':>22} {'p':>7} {'NSE ctrl->phys':>18}")
    with open(os.path.join(outdir, "regime_gain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge", "regime", "frac_snow", "mask", "metric",
                    "ctrl_mean", "ctrl_std", "phys_mean", "phys_std", "rel_mean", "rel_std", "p"])
        for r in results:
            for mr in args.mask_grid:
                for m in ["MAE", "RMSE", "NSE", "KGE", "MAE_extreme"]:
                    gg = basin_gain(r, mr, m)
                    w.writerow([r["gauge"], r["regime"], r["attrs"]["frac_snow"], mr, m,
                                gg["ctrl_mean"], gg["ctrl_std"], gg["phys_mean"], gg["phys_std"],
                                gg["rel_mean"], gg["rel_std"], gg["p"]])
    snow_rel, other_rel = [], []
    for r in sorted(results, key=lambda x: -x["attrs"]["frac_snow"]):
        gm = basin_gain(r, hm, "MAE_extreme"); gn = basin_gain(r, hm, "NSE")
        fs = r["attrs"]["frac_snow"]
        pstr = f"{gm['p']:.3f}" if gm["p"] is not None else "  -"
        print(f"{r['gauge']:<10} {r['regime']:<22} {fs:>9.2f} "
              f"{gm['rel_mean']:>+8.1f} ± {gm['rel_std']:<8.1f}% {pstr:>7} "
              f"{gn['ctrl_mean']:>7.3f} -> {gn['phys_mean']:.3f}")
        (snow_rel if fs > 0.15 else other_rel).append(gm["rel_mean"])
    print("-" * 104)
    if snow_rel:
        print(f"Snow-influenced (frac_snow>0.15, n={len(snow_rel)}): "
              f"mean flood-MAE reduction {np.mean(snow_rel):+.1f}% (across-basin std {np.std(snow_rel):.1f})")
    if other_rel:
        print(f"Other regimes (n={len(other_rel)}): mean {np.mean(other_rel):+.1f}%")
    print("=" * 104)

    # ---- plot: per-basin gain with error bars ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rs = [(r["gauge"], r["regime"], basin_gain(r, hm, "MAE_extreme")) for r in results]
        rs.sort(key=lambda t: -t[2]["rel_mean"])
        labels = [f"{g}\n{reg.split('/')[0][:10]}" for g, reg, _ in rs]
        means = [t[2]["rel_mean"] for t in rs]
        errs = [t[2]["rel_std"] for t in rs]
        cols = ["#1E8449" if m > 0 else "#B03A2E" for m in means]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(range(len(rs)), means, yerr=errs, capsize=4, color=cols)
        ax.set_xticks(range(len(rs))); ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("Flood-peak error reduction (%)")
        ax.set_title(f"Regime-adaptive physics gain at {int(hm*100)}% missing discharge "
                     f"({ns} seed{'s' if ns > 1 else ''}, mean±std)")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "regime_gain.png"), dpi=130); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nArtifacts -> {outdir}/ (regime_results.json, regime_gain.csv, regime_gain.png)")


if __name__ == "__main__":
    main()
