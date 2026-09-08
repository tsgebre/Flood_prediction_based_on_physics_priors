"""Regional (multi-basin) LSTM with static attributes and attribute-gated physics.

One network is trained over all basins pooled — the CAMELS-DL convention (Kratzert
et al. 2019). Key differences from the single-basin pipeline (:mod:`src.main`):

* discharge is converted to specific discharge (mm/day) via catchment area, so a
  single global scaler is physically comparable across basins;
* dynamic features, static attributes, and discharge are normalized on the pooled
  training data (not per basin);
* each basin's static attributes (frac_snow, aridity, baseflow_index, …) are
  broadcast over time and appended to the inputs, letting one network specialize;
* the physics is attribute-gated per basin: the snow features (`w_eff`,
  `swe_recon`) plus a snow-physics loss whose thresholds and weights are gathered
  per sample by basin index (the router of :mod:`src.router`).

Windows never straddle a basin boundary. Evaluation is per basin under the same
seeded sensor-failure mask; the headline is the median NSE across basins.

    python -m src.regional --quick
    python -m src.regional --physics --epochs 20 --hidden 64
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import attributes as A
from src import dataset as D
from src import router as Rt
from src import utils

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

MET = ["prcp", "tmax", "tmin", "srad", "vp"]
STATIC = ["frac_snow", "aridity", "baseflow_index", "p_seasonality",
          "elev_mean", "area_km2", "high_prec_freq"]
PHYS_FEATS = ["w_eff", "swe_recon"]


# --------------------------------------------------------------------------- #
# Loading / pooling
# --------------------------------------------------------------------------- #
def _load_raw(gauge, start="1980-01-01", end="2010-12-31"):
    fp = os.path.join(DATA, f"{gauge}_lump_cida_forcing_leap.txt")
    sp = os.path.join(DATA, f"{gauge}_streamflow_qc.txt")
    meta = A.read_forcing_header(fp)
    f = pd.read_csv(fp, skiprows=3, sep=r"\s+").rename(columns={"Mnth": "month", "Year": "year", "Day": "day"})
    f["date"] = pd.to_datetime(f[["year", "month", "day"]])
    ren = {}
    for c in f.columns:
        for m in MET:
            if m in c.lower():
                ren[c] = m
    f = f.rename(columns=ren)
    q = pd.read_csv(sp, sep=r"\s+", header=None,
                    names=["b", "year", "month", "day", "discharge", "qc"])
    q["date"] = pd.to_datetime(q[["year", "month", "day"]])
    q["discharge"] = q["discharge"].replace(-999, np.nan)
    df = pd.merge(f[["date"] + MET], q[["date", "discharge"]], on="date").sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    df["discharge"] = df["discharge"].interpolate(limit_direction="both")
    df = df.dropna().reset_index(drop=True)
    # snow physics features
    tmean = (df["tmax"].values + df["tmin"].values) / 2.0
    snow = D.degree_day_snow(df["prcp"].values, tmean)
    df["w_eff"] = snow["w_eff"]
    df["swe_recon"] = snow["swe"]
    # specific discharge mm/day
    area_m2 = meta["area_km2"] * 1e6
    df["q_mm"] = df["discharge"].values * 0.0283168 * 86400.0 / area_m2 * 1000.0
    return df, meta


def load_pooled(gauges, physics_features=True, split=(0.70, 0.15)):
    """Return (basins, layout, disch_scaler) with global normalization."""
    dyn_cols = MET + (PHYS_FEATS if physics_features else [])
    raws = {}
    train_frames = []
    for g in gauges:
        df, meta = _load_raw(g)
        n = len(df)
        n_tr = int(n * split[0]); n_va = int(n * (split[0] + split[1]))
        raws[g] = {"df": df, "meta": meta, "n_tr": n_tr, "n_va": n_va,
                   "attrs": A.compute_attributes(g, DATA)}
        train_frames.append(df.iloc[:n_tr])
    pooled = pd.concat(train_frames, ignore_index=True)

    dyn_mu = pooled[dyn_cols].mean().values
    dyn_sd = pooled[dyn_cols].std().replace(0, 1).values
    q_mu = float(pooled["q_mm"].mean()); q_sd = float(pooled["q_mm"].std() or 1)
    stat_df = pd.DataFrame([raws[g]["attrs"] for g in gauges])[STATIC]
    stat_mu = stat_df.mean().values; stat_sd = stat_df.std().replace(0, 1).values

    basins = []
    for bi, g in enumerate(gauges):
        r = raws[g]; df = r["df"]
        dyn = (df[dyn_cols].values - dyn_mu) / dyn_sd
        qn = (df["q_mm"].values - q_mu) / q_sd
        static = (np.array([r["attrs"][k] for k in STATIC]) - stat_mu) / stat_sd
        sl = slice(0, r["n_tr"])
        thr = {
            "thresh_flood": float(np.quantile(qn[sl], 0.95)),
            "q_zero_norm": float((0.0 - q_mu) / q_sd),
        }
        spec = Rt.route_physics(r["attrs"])
        if physics_features:
            weff_norm = dyn[:, dyn_cols.index("w_eff")]
            thr["weff_hi"] = float(np.quantile(weff_norm[sl], 0.95))
            thr["weff_low"] = float(np.quantile(weff_norm[sl], 0.40))
        basins.append({
            "gauge": g, "idx": bi, "regime": spec.regime, "attrs": r["attrs"],
            "dyn": dyn.astype(np.float32), "q": qn.astype(np.float32),
            "static": static.astype(np.float32),
            "splits": (r["n_tr"], r["n_va"], len(df)),
            "thr": thr,
            # per-basin discharge std on the train split (normalized space) -> the NSE-loss
            # denominator that balances the pooled objective across basins (Kratzert 2019).
            "q_std": float(np.std(qn[sl]) or 1.0),
            "w": {"recession": spec.weights["recession"], "response": spec.weights["response"],
                  "nonneg": spec.weights["nonneg"]},
        })
    layout = {"dyn_cols": dyn_cols, "n_met": len(MET), "n_static": len(STATIC),
              "physics_features": physics_features,
              "precip_idx": 0, "w_eff_idx": (dyn_cols.index("w_eff") if physics_features else None)}
    return basins, layout, {"mu": q_mu, "sd": q_sd}


# --------------------------------------------------------------------------- #
# Dataset (pooled windows, static broadcast, discharge + missingness channel)
# --------------------------------------------------------------------------- #
class RegionalDataset(Dataset):
    def __init__(self, basins, layout, seq_len=30, part="train", is_train=False,
                 train_mask_hi=1.0, generator=None):
        self.seq_len = seq_len
        self.is_train = is_train
        self.train_mask_hi = train_mask_hi
        self.generator = generator
        self.n_static = layout["n_static"]
        self.index = []          # (basin_i, start)
        self.B = basins
        for bi, b in enumerate(basins):
            n_tr, n_va, n = b["splits"]
            if part == "train":
                lo, hi = 0, n_tr
            elif part == "val":
                lo, hi = n_tr, n_va
            else:
                lo, hi = n_va, n
            for s in range(lo, hi - seq_len):
                self.index.append((bi, s))
        # column layout of x: [dyn..., static..., discharge, obs_mask]
        self.n_dyn = basins[0]["dyn"].shape[1]
        self.q_col = self.n_dyn + self.n_static
        self.mask_col = self.q_col + 1

    @property
    def n_features(self):
        return self.n_dyn + self.n_static + 2

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        bi, s = self.index[k]
        b = self.B[bi]
        dyn = torch.from_numpy(b["dyn"][s:s + self.seq_len])
        q = torch.from_numpy(b["q"][s:s + self.seq_len]).unsqueeze(-1)
        static = torch.from_numpy(b["static"]).unsqueeze(0).expand(self.seq_len, -1)
        y = torch.tensor([b["q"][s + self.seq_len]], dtype=torch.float32)
        obs = torch.ones_like(q)
        if self.is_train:
            if self.generator is not None:
                p = self.train_mask_hi * torch.rand(1, generator=self.generator).item()
                keep = (torch.rand(q.shape, generator=self.generator) >= p).float()
            else:
                p = self.train_mask_hi * torch.rand(1).item()
                keep = (torch.rand(q.shape) >= p).float()
            q = q * keep; obs = keep
        x = torch.cat([dyn, static, q, obs], dim=-1)
        return {"x": x, "y": y, "basin": bi}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RegionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Attribute-gated snow physics (per-sample thresholds/weights by basin index)
# --------------------------------------------------------------------------- #
def _mavg(x, k):
    if k <= 1:
        return x
    xt = x.transpose(1, 2)
    xt = F.pad(xt, (k - 1, 0), mode="replicate")
    w = torch.ones(1, 1, k, device=x.device) / k
    return F.conv1d(xt, w).transpose(1, 2)


def regional_physics_loss(outputs, x, basin_idx, thr_t, w_t, layout, lag=4, smooth=5, k=6.0):
    """thr_t: (n_basins, 4) [thresh_flood, q_zero, weff_hi, weff_low];
       w_t:   (n_basins, 3) [recession, response, nonneg]. Gathered by basin_idx."""
    q = outputs
    widx = layout["w_eff_idx"]
    w = x[:, :, widx:widx + 1]
    w_s = _mavg(w, smooth)
    L = max(1, lag)
    thr = thr_t[basin_idx]                      # (B,4)
    ww = w_t[basin_idx]                          # (B,3)
    tf = thr[:, 0].view(-1, 1, 1); qz = thr[:, 1].view(-1, 1, 1)
    whi = thr[:, 2].view(-1, 1, 1); wlo = thr[:, 3].view(-1, 1, 1)

    nonneg = F.relu(qz - q).mean(dim=(1, 2))
    dry = torch.sigmoid(k * (wlo - w_s))
    rise = F.relu(q[:, 1:, :] - q[:, :-1, :])
    rec = (rise * dry[:, :-1, :]).mean(dim=(1, 2))
    hi = torch.sigmoid(k * (w_s - whi))
    if q.size(1) > L:
        resp = (F.relu(tf - q[:, L:, :]) * hi[:, :-L, :]).mean(dim=(1, 2))
    else:
        resp = (F.relu(tf - q) * hi).mean(dim=(1, 2))
    total = (ww[:, 0] * rec + ww[:, 1] * resp + ww[:, 2] * nonneg).mean()
    return total, {"recession": rec.mean().item(), "response": resp.mean().item(), "nonneg": nonneg.mean().item()}


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _regional_data_loss(pred, y, bidx, qstd_t, loss_type, eps=0.1):
    """Pooled data loss. 'mse' = plain pooled MSE (high-flow basins dominate).
    'nse' = basin-averaged NSE loss: each sample's squared error divided by its basin's
    discharge variance, so every basin contributes equally (Kratzert et al. 2019)."""
    if loss_type == "nse":
        denom = (qstd_t[bidx].view(-1, 1) + eps)  2
        return torch.mean((pred - y)  2 / denom)
    return F.mse_loss(pred, y)


def train_regional(basins, layout, hp, device, seed, use_physics, train_mask_hi,
                   loss_type="mse", verbose=True):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr = RegionalDataset(basins, layout, hp["seq_len"], "train", True, train_mask_hi, g)
    va = RegionalDataset(basins, layout, hp["seq_len"], "val")
    dg = torch.Generator().manual_seed(seed)
    trl = DataLoader(tr, batch_size=256, shuffle=True, generator=dg)
    val = DataLoader(va, batch_size=256)

    model = RegionalLSTM(tr.n_features, hp["hidden"], hp["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    thr_t = torch.tensor([[b["thr"].get("thresh_flood", 2.0), b["thr"].get("q_zero_norm", 0.0),
                           b["thr"].get("weff_hi", 2.0), b["thr"].get("weff_low", 0.0)] for b in basins],
                         dtype=torch.float32, device=device)
    w_t = torch.tensor([[b["w"]["recession"], b["w"]["response"], b["w"]["nonneg"]] for b in basins],
                       dtype=torch.float32, device=device)
    qstd_t = torch.tensor([b.get("q_std", 1.0) for b in basins], dtype=torch.float32, device=device)

    best = float("inf"); best_w = copy.deepcopy(model.state_dict()); pc = 0
    for ep in range(hp["epochs"]):
        model.train()
        for batch in trl:
            x = batch["x"].to(device); y = batch["y"].to(device); bidx = batch["basin"].to(device)
            opt.zero_grad()
            out = model(x)
            loss = _regional_data_loss(out[:, -1, :], y, bidx, qstd_t, loss_type)
            if use_physics:
                pl, _ = regional_physics_loss(out, x, bidx, thr_t, w_t, layout,
                                              lag=hp["lag"], smooth=hp["smooth"])
                loss = loss + pl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # val
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for batch in val:
                x = batch["x"].to(device); y = batch["y"].to(device)
                bidx = batch["basin"].to(device)
                vl += float(_regional_data_loss(model(x)[:, -1, :], y, bidx, qstd_t, loss_type)); nb += 1
        vl /= max(nb, 1)
        if vl < best:
            best = vl; best_w = copy.deepcopy(model.state_dict()); pc = 0
        else:
            pc += 1
            if pc >= hp["patience"]:
                break
        if verbose and (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{hp['epochs']} val {vl:.5f}", flush=True)
    model.load_state_dict(best_w)
    return model


def evaluate_regional(model, basins, layout, disch_scaler, mask_grid, seq_len, device, seed=123):
    """Per-basin metrics under seeded discharge masking. Returns {gauge: {mask: metrics}}."""
    model.eval()
    mu, sd = disch_scaler["mu"], disch_scaler["sd"]
    out = {}
    for b in basins:
        ds = RegionalDataset([b], layout, seq_len, "test")
        dl = DataLoader(ds, batch_size=256)
        per = {}
        for mr in mask_grid:
            gen = torch.Generator().manual_seed(seed)
            preds, targs = [], []
            with torch.no_grad():
                for batch in dl:
                    x = batch["x"].clone()
                    if mr > 0:
                        keep = (torch.rand(x[:, :, ds.q_col].shape, generator=gen) >= mr).float()
                        x[:, :, ds.q_col] *= keep
                        x[:, :, ds.mask_col] *= keep
                    o = model(x.to(device))[:, -1, :].cpu().numpy()
                    preds.append(o); targs.append(batch["y"].numpy())
            p = np.concatenate(preds).ravel() * sd + mu     # -> mm/day
            t = np.concatenate(targs).ravel() * sd + mu
            m = utils.compute_metrics(t, p)
            m["mask_ratio"] = float(mr)
            per[mr] = m
        out[b["gauge"]] = {"regime": b["regime"], "frac_snow": b["attrs"]["frac_snow"], "sweep": per}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basins", nargs="+", default=None)
    ap.add_argument("--physics", action="store_true", help="also train the physics arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mask-grid", type=float, nargs="+", default=[0.0, 0.8, 1.0])
    ap.add_argument("--headline-mask", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-mask-hi", type=float, default=1.0)
    ap.add_argument("--loss", choices=["mse", "nse"], default="mse",
                    help="'nse' = basin-averaged NSE loss (balances the pooled objective)")
    ap.add_argument("--outdir", default="results/regional")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    basins_ids = args.basins or A.basins_in(DATA)
    hp = {"seq_len": args.seq_len, "hidden": args.hidden, "layers": args.layers,
          "lr": 0.001, "epochs": args.epochs, "patience": 5, "lag": 4, "smooth": 5}
    if args.quick:
        hp.update(hidden=32, epochs=3, patience=2)
    outdir = os.path.join(HERE, args.outdir); os.makedirs(outdir, exist_ok=True)
    hm = min(args.mask_grid, key=lambda x: abs(x - args.headline_mask))

    arms = ["control"] + (["physics"] if args.physics else [])
    print(f"Regional model | basins={len(basins_ids)} | arms={arms} | seeds={args.seeds} "
          f"| hidden={hp['hidden']} epochs<={hp['epochs']}", flush=True)

    results = {"config": {"basins": basins_ids, "hp": hp, "mask_grid": args.mask_grid,
                          "seeds": args.seeds, "static": STATIC}, "arms": {}}
    for arm in arms:
        use_phys = (arm == "physics")
        basins, layout, dsc = load_pooled(basins_ids, physics_features=use_phys)
        per_seed = []
        for sd in args.seeds:
            print(f"  [{arm}] seed {sd} (input_dim={RegionalDataset(basins, layout, hp['seq_len']).n_features}) ...", flush=True)
            model = train_regional(basins, layout, hp, device, sd, use_phys, args.train_mask_hi,
                                   loss_type=args.loss, verbose=(sd == args.seeds[0]))
            per_seed.append(evaluate_regional(model, basins, layout, dsc, args.mask_grid, hp["seq_len"], device))
        results["arms"][arm] = per_seed
        # report median NSE across basins at each mask (seed-0)
        nse0 = {mr: np.median([per_seed[0][g]["sweep"][mr]["NSE"] for g in per_seed[0]]) for mr in args.mask_grid}
        print(f"    {arm}: median NSE across basins " +
              " ".join(f"{int(mr*100)}%={nse0[mr]:.3f}" for mr in args.mask_grid), flush=True)
        json.dump(results, open(os.path.join(outdir, "regional_results.json"), "w"), indent=2, default=float)

    # ---- comparison if both arms ----
    if "physics" in results["arms"]:
        c = results["arms"]["control"][0]; p = results["arms"]["physics"][0]
        print("\n" + "=" * 92)
        print(f"REGIONAL: Control vs +physics, per basin @ {int(hm*100)}% missing (seed {args.seeds[0]})")
        print("-" * 92)
        print(f"{'gauge':<10}{'regime':<22}{'fsnow':>7}{'NSE ctrl':>10}{'NSE phys':>10}{'floodMAE Δ%':>13}")
        for g in sorted(c, key=lambda x: -c[x]["frac_snow"]):
            cc = c[g]["sweep"][hm]; pp = p[g]["sweep"][hm]
            cme, pme = cc.get("MAE_extreme", np.nan), pp.get("MAE_extreme", np.nan)
            rel = 100 * (cme - pme) / cme if cme else np.nan
            print(f"{g:<10}{c[g]['regime']:<22}{c[g]['frac_snow']:>7.2f}"
                  f"{cc['NSE']:>10.3f}{pp['NSE']:>10.3f}{rel:>+12.1f}%")
        med_c = np.median([c[g]["sweep"][hm]["NSE"] for g in c])
        med_p = np.median([p[g]["sweep"][hm]["NSE"] for g in p])
        print("-" * 92)
        print(f"median NSE @ {int(hm*100)}% missing: control {med_c:.3f} -> physics {med_p:.3f}")
        print("=" * 92)

    print(f"\nArtifacts -> {outdir}/ (regional_results.json)")


if __name__ == "__main__":
    main()
