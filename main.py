"""Flood prediction: Physics-Informed vs data-driven recurrent models.

Default run trains a data-driven Control and a Physics-Informed model across
several seeds, then evaluates both on the test set over a grid of
missing-discharge ratios (simulated sensor failure). It writes a metrics table,
JSON/CSV results and plots to ``results/`` and prints a paired comparison with
dispersion and a significance test.

``--ablation`` additionally trains a model with a reduced input
representation ([precip, discharge], no missingness channel) so the gain from
the richer data representation can be separated from the gain from physics.

Examples
--------
    python main.py                      # multi-seed Control vs PI (LSTM)
    python main.py --ablation           # orig-repr -> improved-repr -> +physics
    python main.py --model GRU
    python main.py --quick              # fast smoke run (1 seed, few epochs)
    python main.py --single --no-physics
    python main.py --adaptive-physics   # Lagrangian dual-ascent weighting
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import src.dataset as dataset
import src.models as models
import src.train as train
import src.utils as utils

BASE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.join(BASE, "data", "01013500_streamflow_qc.txt")
FP = os.path.join(BASE, "data", "01013500_lump_cida_forcing_leap.txt")
HIGHER_BETTER = ("NSE", "KGE")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(train_df, val_df, test_df, meta, seq_len, batch_size, seed,
                  train_mask_range, use_mask_channel):
    g = torch.Generator().manual_seed(seed)
    train_ds = dataset.CamelsDataset(
        train_df, meta["feature_cols"], seq_len, is_train=True,
        train_mask_range=train_mask_range, generator=g, use_mask_channel=use_mask_channel,
    )
    val_ds = dataset.CamelsDataset(val_df, meta["feature_cols"], seq_len,
                                   use_mask_channel=use_mask_channel)
    test_ds = dataset.CamelsDataset(test_df, meta["feature_cols"], seq_len,
                                    use_mask_channel=use_mask_channel)
    dl_g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=dl_g),
        "val": DataLoader(val_ds, batch_size=batch_size),
        "test": DataLoader(test_ds, batch_size=batch_size),
        "test_ds": test_ds,
        "input_dim": train_ds.n_features,
    }


def persistence_baseline(test_ds, meta):
    """Naive reference: tomorrow's discharge = last observed discharge today.

    Reported only in the unmasked regime -- persistence has no fallback when the
    sensor it copies is the one that fails.
    """
    q = test_ds.discharge.numpy().ravel()
    sl = test_ds.seq_len
    preds = q[sl - 1:len(q) - 1].reshape(-1, 1)
    targets = q[sl:].reshape(-1, 1)
    lt = meta["log_transform"]
    preds = utils._inverse(meta["scaler_discharge"], preds, lt)
    targets = utils._inverse(meta["scaler_discharge"], targets, lt)
    return utils.compute_metrics(targets, preds)


def train_one(model_type, use_physics, adaptive, hp, meta, loaders, device, seed, verbose,
              physics_mode="precip"):
    set_seed(seed)
    model = models.build_model(
        model_type, loaders["input_dim"], hp["hidden"], hp["layers"], 1, meta, dropout=hp["dropout"])
    phys = train.PhysicsConfig(use_physics=use_physics, adaptive=adaptive, mode=physics_mode)
    return train.train_model(
        model, loaders["train"], loaders["val"], lr=hp["lr"], epochs=hp["epochs"],
        patience=hp["patience"], device=device, physics=phys, verbose=verbose)


def aggregate(per_seed_rows):
    """list[seed] of list[mask_ratio] of metric dict -> {mask: {metric: (mean,std,[vals])}}."""
    out = {}
    ratios = [r["mask_ratio"] for r in per_seed_rows[0]]
    for i, mr in enumerate(ratios):
        vals = {}
        for metric in per_seed_rows[0][i]:
            if metric == "mask_ratio":
                continue
            arr = [sr[i].get(metric, np.nan) for sr in per_seed_rows]
            vals[metric] = (float(np.nanmean(arr)), float(np.nanstd(arr)), [float(a) for a in arr])
        out[mr] = vals
    return out


def paired_test(base_vals, new_vals):
    """Paired diff (base-new) across seeds and a p-value if possible."""
    b = np.asarray(base_vals, float)
    n = np.asarray(new_vals, float)
    diff = float(np.mean(b - n))
    pval = None
    if len(b) >= 2 and not np.allclose(b - n, (b - n)[0]):
        try:
            from scipy import stats
            pval = float(stats.ttest_rel(b, n).pvalue)
        except Exception:
            pval = None
    return diff, pval


def run_study(name, features, use_mask_channel, use_physics, adaptive,
              seeds, hp, args, device, physics_features=False, physics_mode="precip"):
    """Train `name` across seeds and evaluate the robustness sweep. Returns dict."""
    train_df, val_df, test_df, meta = dataset.load_and_preprocess_data(
        SP, FP, start_date=args.start, end_date=args.end,
        met_features=tuple(features), log_transform=args.log_transform,
        physics_features=physics_features)

    per_seed_rows = []
    first = {}
    for si, seed in enumerate(seeds):
        loaders = build_loaders(train_df, val_df, test_df, meta, args.seq_len, 32, seed,
                                (0.0, args.train_mask_hi), use_mask_channel)
        print(f"  [{name}] seed {seed} (input_dim={loaders['input_dim']}) ...", flush=True)
        model, hist = train_one(args.model, use_physics, adaptive, hp, meta, loaders,
                                 device, seed, verbose=(si == 0), physics_mode=physics_mode)
        rows = utils.robustness_sweep(model, loaders["test"], meta, args.mask_grid,
                                      seed=123, device=device)
        per_seed_rows.append(rows)
        if si == 0:
            hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))
            pc, tgt = utils.evaluate_stress_test(model, loaders["test"], meta, hm, seed=123, device=device)
            first = {"model": model, "history": hist, "meta": meta,
                     "test_ds": loaders["test_ds"], "preds_headline": pc, "targets_headline": tgt}
    return {"name": name, "agg": aggregate(per_seed_rows), "first": first,
            "features": list(meta["met_features"]), "use_mask_channel": use_mask_channel,
            "use_physics": use_physics}


def print_table(studies, mask_grid, headline_mask, seeds):
    hm = min(mask_grid, key=lambda x: abs(x - headline_mask))
    metrics = ["NSE", "KGE", "MAE", "RMSE", "MAE_extreme"]
    names = [s["name"] for s in studies]
    w = max(18, max(len(n) for n in names) + 2)
    print("\n" + "=" * (16 + (w + 3) * len(names)))
    print(f"Metrics @ {hm*100:.0f}% missing discharge (mean±std over {len(seeds)} seed(s))")
    print("-" * (16 + (w + 3) * len(names)))
    header = f"{'Metric':<14} | " + " | ".join(f"{n:<{w}}" for n in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        if m not in studies[0]["agg"][hm]:
            continue
        cells = []
        for s in studies:
            mean, std, _ = s["agg"][hm][m]
            cells.append(f"{mean:>8.3f} ± {std:<{w-11}.3f}")
        print(f"{m:<14} | " + " | ".join(cells))
    print("=" * (16 + (w + 3) * len(names)))
    # pairwise improvements vs the first study (baseline)
    base = studies[0]
    for s in studies[1:]:
        print(f"\nΔ  {s['name']}  vs  {base['name']}  @ {hm*100:.0f}% missing:")
        for m in metrics:
            if m not in base["agg"][hm]:
                continue
            bmean, _, bvals = base["agg"][hm][m]
            nmean, _, nvals = s["agg"][hm][m]
            higher = m in HIGHER_BETTER
            improve = (nmean - bmean) if higher else (bmean - nmean)
            diff, pval = paired_test(bvals, nvals) if higher else paired_test(nvals, bvals)
            # paired_test returns base-new; align sign to "improvement"
            rel = (100.0 * improve / abs(bmean)) if bmean != 0 else float("nan")
            sig = "" if pval is None else f"  (paired p={pval:.3f})"
            mark = "better" if improve > 0 else "worse"
            print(f"    {m:<12}: {improve:+.3f} ({rel:+.1f}%) {mark}{sig}")


def save_artifacts(studies, persist, args, outdir):
    hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))
    results = {"config": vars(args), "persistence": persist, "studies": {}}
    for s in studies:
        results["studies"][s["name"]] = {
            "features": s["features"], "use_mask_channel": s["use_mask_channel"],
            "use_physics": s["use_physics"],
            "sweep": {str(mr): s["agg"][mr] for mr in s["agg"]},
        }
    json.dump(results, open(os.path.join(outdir, "results.json"), "w"), indent=2, default=float)

    with open(os.path.join(outdir, "robustness.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["study", "mask_ratio", "metric", "mean", "std"])
        for s in studies:
            for mr, md in s["agg"].items():
                for metric, (mean, std, _) in md.items():
                    wtr.writerow([s["name"], mr, metric, mean, std])

    for metric in ["MAE", "NSE", "MAE_extreme"]:
        agg = {}
        for s in studies:
            rows = [{"mask_ratio": mr, "mean": s["agg"][mr][metric][0], "std": s["agg"][mr][metric][1]}
                    for mr in sorted(s["agg"]) if metric in s["agg"][mr]]
            if rows:
                agg[s["name"]] = rows
        if agg:
            utils.plot_robustness_curve(agg, metric, os.path.join(outdir, f"robustness_{metric}.png"))

    # hydrograph + checkpoints from the physics vs its control (last two studies)
    preds = {s["name"]: s["first"]["preds_headline"] for s in studies if "preds_headline" in s["first"]}
    if preds:
        tgt = studies[0]["first"]["targets_headline"]
        utils.plot_hydrograph(tgt, preds, os.path.join(outdir, "hydrograph.png"), mask_ratio=hm)
    histories = {s["name"]: s["first"]["history"] for s in studies if "history" in s["first"]}
    if histories:
        utils.plot_training_history(histories, os.path.join(outdir, "training_history.png"))
    for s in studies:
        if "model" in s["first"]:
            tag = s["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            torch.save(s["first"]["model"].state_dict(), os.path.join(outdir, f"{args.model}_{tag}.pt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["LSTM", "GRU"], default="LSTM")
    ap.add_argument("--single", action="store_true", help="Train one config, no comparison")
    ap.add_argument("--no-physics", action="store_true", help="Disable physics (with --single)")
    ap.add_argument("--ablation", action="store_true",
                    help="Add the reduced-representation baseline")
    ap.add_argument("--snow-study", action="store_true",
                    help="Degree-day snow ablation: Control -> +snow feature -> +snow physics")
    ap.add_argument("--physics-mode", choices=["precip", "snow"], default="precip",
                    help="Physics constraint family for the PI model")
    ap.add_argument("--train-mask-hi", type=float, default=0.5,
                    help="Upper bound of the training-time discharge dropout range")
    ap.add_argument("--adaptive-physics", action="store_true",
                    help="Learn constraint weights via Lagrangian dual ascent")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42, help="Single seed (with --single)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--headline-mask", type=float, default=0.8)
    ap.add_argument("--mask-grid", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--features", type=str, nargs="+", default=list(dataset.DEFAULT_MET_FEATURES))
    ap.add_argument("--start", default="1980-01-01")
    ap.add_argument("--end", default="1999-12-31")
    ap.add_argument("--log-transform", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="Fast smoke run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Using device: {device}")
    if not os.path.exists(SP):
        print(f"Error: data file not found at {SP}")
        return

    hp = {
        "LSTM": dict(hidden=64, lr=0.0018),
        "GRU": dict(hidden=64, lr=0.0011),
    }[args.model]
    hp.update(layers=args.layers, dropout=0.0, epochs=40, patience=8,
              w_mono=0.1, w_thresh=0.1, w_nonneg=0.5)
    if args.quick:  # fast defaults; explicit --epochs/--hidden below still win
        hp.update(epochs=6, hidden=32)
    if args.hidden:
        hp["hidden"] = args.hidden
    if args.epochs:
        hp["epochs"] = args.epochs
    seeds = args.seeds or ([0] if (args.quick or args.single) else [0, 1, 2])

    outdir = os.path.join(BASE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # ---- single-run mode ----
    if args.single:
        study = run_study("single", args.features, True, not args.no_physics,
                          args.adaptive_physics, [args.seed], hp, args, device,
                          physics_features=(args.physics_mode == "snow"),
                          physics_mode=args.physics_mode)
        print("\nRobustness sweep (single):")
        for mr in sorted(study["agg"]):
            md = study["agg"][mr]
            print(f"  {mr*100:4.0f}% | MAE {md['MAE'][0]:8.2f} | NSE {md['NSE'][0]:.4f} | "
                  f"KGE {md['KGE'][0]:.4f} | MAE_extreme {md.get('MAE_extreme', (float('nan'),))[0]:8.2f}")
        json.dump({str(mr): study["agg"][mr] for mr in study["agg"]},
                  open(os.path.join(outdir, "single_run.json"), "w"), indent=2, default=float)
        return

    # ---- comparison study ----
    # config = (name, features, use_mask_channel, use_physics, physics_features, physics_mode)
    full = list(args.features)
    if args.snow_study:
        configs = [
            ("Control", full, True, False, False, "precip"),
            ("+ Snow feature", full, True, False, True, "precip"),
            ("+ Snow physics", full, True, True, True, "snow"),
        ]
    else:
        configs = []
        if args.ablation:
            configs.append(("Control (orig repr)", ["prcp"], False, False, False, "precip"))
        configs.append(("Control (improved)" if args.ablation else "Control", full, True, False, False, "precip"))
        configs.append(("Physics-Informed", full, True, True, False,
                        args.physics_mode))

    print(f"\n=== {args.model}: {' -> '.join(c[0] for c in configs)} "
          f"{'(adaptive)' if args.adaptive_physics else '(fixed weights)'} ===")
    print(f"Seeds: {seeds} | mask grid: {args.mask_grid} | train-mask<= {args.train_mask_hi} "
          f"| epochs<= {hp['epochs']} | hidden {hp['hidden']}")

    studies = []
    for name, feats, mask_ch, phys, pf, pmode in configs:
        studies.append(run_study(name, feats, mask_ch, phys, args.adaptive_physics,
                                 seeds, hp, args, device, physics_features=pf, physics_mode=pmode))

    persist = persistence_baseline(studies[-1]["first"]["test_ds"], studies[-1]["first"]["meta"])
    print(f"\n[reference] persistence (0% missing): MAE {persist['MAE']:.2f}, NSE {persist['NSE']:.4f}")

    print_table(studies, args.mask_grid, args.headline_mask, seeds)
    save_artifacts(studies, persist, args, outdir)
    print(f"\nArtifacts -> {outdir}/ (results.json, robustness.csv, *.png, checkpoints)")


if __name__ == "__main__":
    main()
