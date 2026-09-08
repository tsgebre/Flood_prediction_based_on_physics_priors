"""Evaluation metrics, reproducible stress testing and plotting."""

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def calculate_nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (1 = perfect, 0 = mean baseline)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def calculate_kge(y_true, y_pred):
    """Kling-Gupta Efficiency (Gupta et al., 2009)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.std() == 0 or y_pred.std() == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std() / y_true.std()
    beta = y_pred.mean() / y_true.mean() if y_true.mean() != 0 else np.nan
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def compute_metrics(targets, preds, extreme_pct=95):
    """Return a dict of overall and extreme-event metrics (physical units)."""
    t = np.asarray(targets).ravel()
    p = np.asarray(preds).ravel()
    err = p - t
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "NSE": float(calculate_nse(t, p)),
        "KGE": float(calculate_kge(t, p)),
        "PBIAS": float(100.0 * np.sum(err) / np.sum(t)) if np.sum(t) != 0 else np.nan,
    }
    # Extreme (flood) subset: observations above the test-set percentile.
    thr = np.percentile(t, extreme_pct)
    hi = t >= thr
    if hi.sum() > 0:
        out["MAE_extreme"] = float(np.mean(np.abs(err[hi])))
        out["RMSE_extreme"] = float(np.sqrt(np.mean(err[hi] ** 2)))
        out["n_extreme"] = int(hi.sum())
        out["extreme_threshold"] = float(thr)
    return out


# --------------------------------------------------------------------------- #
# Inference / stress testing
# --------------------------------------------------------------------------- #
def _inverse(scaler, arr, log_transform):
    arr = scaler.inverse_transform(np.asarray(arr).reshape(-1, 1))
    if log_transform:
        arr = np.expm1(arr)
    return arr


def evaluate_stress_test(model, loader, meta, mask_ratio=0.0, seed=0, device="cpu"):
    """Evaluate under simulated discharge-sensor failure.

    Masking is reproducible and, given the same ``seed`` and (unshuffled) loader,
    *identical* across models -- so Control vs PI is a fair, paired comparison.
    A masked entry has its discharge zeroed AND (when the dataset carries an
    observed-flag channel) its flag set to 0, so the model is explicitly told the
    value is absent.
    """
    model.eval()
    scaler = meta["scaler_discharge"]
    ds = loader.dataset
    q_idx, m_idx = ds.q_idx, ds.mask_idx  # m_idx None if no mask channel
    gen = torch.Generator().manual_seed(int(seed))

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].clone()
            y = batch["y"]
            if mask_ratio > 0.0:
                keep = (torch.rand(x[:, :, q_idx].shape, generator=gen) >= mask_ratio).float()
                x[:, :, q_idx] = x[:, :, q_idx] * keep
                if m_idx is not None:
                    x[:, :, m_idx] = x[:, :, m_idx] * keep  # flag the masked entries
            out = model(x.to(device))
            preds.append(out[:, -1, :].cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return _inverse(scaler, preds, meta["log_transform"]), _inverse(scaler, targets, meta["log_transform"])


def robustness_sweep(model, loader, meta, mask_ratios, seed=0, device="cpu", extreme_pct=95):
    """Evaluate a trained model across a grid of mask ratios."""
    rows = []
    for mr in mask_ratios:
        preds, targets = evaluate_stress_test(model, loader, meta, mask_ratio=mr, seed=seed, device=device)
        m = compute_metrics(targets, preds, extreme_pct=extreme_pct)
        m["mask_ratio"] = float(mr)
        rows.append(m)
    return rows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
_PALETTE = ["#B03A2E", "#1F618D", "#1E8449", "#B7950B", "#6C3483"]


def _color_for(name, i):
    fixed = {"Control": "#B03A2E", "Physics-Informed": "#1F618D",
             "Control (orig repr)": "#B03A2E", "Control (improved)": "#E67E22"}
    return fixed.get(name, _PALETTE[i % len(_PALETTE)])


def plot_robustness_curve(agg, metric, outpath, title=None):
    """agg: {name: [{mask_ratio, mean, std}...], ...} for any set of models."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, rows) in enumerate(agg.items()):
        c = _color_for(name, i)
        xs = [r["mask_ratio"] * 100 for r in rows]
        ys = [r["mean"] for r in rows]
        es = [r.get("std", 0.0) for r in rows]
        ax.plot(xs, ys, "-o", label=name, color=c, linewidth=2.5, markersize=7)
        ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                        color=c, alpha=0.18)
    ax.set_xlabel("Missing discharge (%)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Robustness under sensor failure: {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_hydrograph(targets, preds_by_model, outpath, mask_ratio, n=300, title=None):
    fig, ax = plt.subplots(figsize=(13, 6))
    t = np.asarray(targets).ravel()[:n]
    ax.plot(t, color="black", linewidth=2.0, label="Observed", zorder=3)
    for i, (name, preds) in enumerate(preds_by_model.items()):
        ax.plot(np.asarray(preds).ravel()[:n], linewidth=1.6, alpha=0.9,
                label=name, color=_color_for(name, i))
    ax.set_xlabel("Test day")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title(title or f"Hydrograph at {int(mask_ratio*100)}% missing discharge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_training_history(histories, outpath):
    """histories: {label: history_list_of_dicts}"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for label, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["val"] for r in h], "-o", label=label, markersize=4)
    axes[0].set_title("Validation loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    # physics components if present (any key that is not bookkeeping)
    skip = {"epoch", "val", "data", "total", "lam"}
    for label, h in histories.items():
        comp_keys = [k for k in h[0] if k not in skip]
        ep = [r["epoch"] for r in h]
        for ck in comp_keys:
            if any(r.get(ck, 0) for r in h):
                axes[1].plot(ep, [r.get(ck, 0) for r in h], "--", label=f"{label}:{ck}")
    axes[1].set_title("Physics violations (train)"); axes[1].set_xlabel("epoch")
    if axes[1].has_data():
        axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
