"""Training loop and the physics-informed loss.

Fixes relative to the original repo
------------------------------------
* Constraint strength is a **fixed hyperparameter** (default) or a proper
  **Lagrange multiplier updated by dual ascent** (``adaptive_physics=True``) --
  never a plain parameter descended on the same loss (which collapses to 0).
* Thresholds are fixed model buffers (see :mod:`src.models`), not trained.
* Three physically-motivated soft constraints:
    1. **Non-negativity** -- predicted discharge must not fall below physical 0
       (unconditionally valid; was missing entirely before).
    2. **Lag-tolerant monotonicity** -- a sustained rise in *smoothed* rainfall
       should not be followed by *lower* discharge ``lag`` steps later
       (respects rainfall-runoff lag instead of demanding a same-step response).
    3. **Lagged flood threshold** -- when trailing-window rainfall is extreme,
       discharge over the following window should approach the flood level.
* Gradient clipping for RNN stability; per-batch-averaged loss history returned
  for plotting/inspection.
"""

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class PhysicsConfig:
    use_physics: bool = True
    mode: str = "precip"    # 'precip' (rain-based) or 'snow' (degree-day driven)
    # Fixed weights per constraint name (used when adaptive=False).
    weights: dict = field(default_factory=lambda: {
        # precip mode
        "mono": 0.1, "thresh": 0.1,
        # snow mode
        "recession": 0.1, "response": 0.1,
        # shared
        "nonneg": 0.5,
    })
    # Constraint shape parameters.
    lag: int = 4            # rainfall/snowmelt-runoff lag in days
    smooth: int = 5         # moving-average window applied to the forcing
    sharpness: float = 6.0  # sigmoid sharpness of the extreme-input / dry gates
    # Force-decay ceiling for the recession-rate band (only safe below the snow line;
    # snow baseflow can legitimately sit flat mid-winter). See _snow_losses.
    force_decay: bool = False
    # Adaptive (Lagrangian dual-ascent) mode.
    adaptive: bool = False
    dual_lr: float = 0.05
    lambda_max: float = 10.0

    def weight(self, key):
        return float(self.weights.get(key, 0.0))


def nse_loss(pred, target, eps=0.1):
    """Nash-Sutcliffe-style loss = MSE normalized by the target variance.

    Minimizing this maximizes NSE, which weights the fit by how much the flow
    actually varies (rather than absolute magnitude) -- the hydrology-standard
    objective (cf. Kratzert et al. 2019). On per-basin standardized targets the
    denominator is ~1, so it closely tracks MSE; the normalization matters most
    when target scale varies within/across batches.
    """
    denom = torch.mean((target - target.mean()) ** 2) + eps
    return torch.mean((pred - target) ** 2) / denom


def _data_loss(pred, target, loss_type):
    return nse_loss(pred, target) if loss_type == "nse" else F.mse_loss(pred, target)


def _moving_avg(x, k):
    """Causal-ish moving average over the time dim of (B, T, 1)."""
    if k <= 1:
        return x
    xt = x.transpose(1, 2)  # (B, 1, T)
    pad = k - 1
    xt = F.pad(xt, (pad, 0), mode="replicate")
    w = torch.ones(1, 1, k, device=x.device) / k
    return F.conv1d(xt, w).transpose(1, 2)


def physics_losses(outputs, x, model, cfg: PhysicsConfig):
    """Return the three soft-constraint violation means (each >= 0).

    ``outputs`` : (B, T, 1) predicted normalized discharge for the window.
    ``x``       : (B, T, F) inputs; rainfall at column ``model.precip_idx``.
    """
    if cfg.mode == "snow":
        return _snow_losses(outputs, x, model, cfg)

    q = outputs
    r = x[:, :, model.precip_idx : model.precip_idx + 1]
    r_s = _moving_avg(r, cfg.smooth)
    L = max(1, cfg.lag)

    # 1. Non-negativity: physical discharge must be >= 0.
    loss_nonneg = F.relu(model.q_zero_norm - q).mean()

    # 2. Lag-tolerant monotonicity: if smoothed rain rose over L steps,
    #    discharge L steps later should not be lower than before.
    if q.size(1) > L:
        rain_rise = (r_s[:, L:, :] > r_s[:, :-L, :]).float()
        mono_viol = F.relu(q[:, :-L, :] - q[:, L:, :]) * rain_rise
        loss_mono = mono_viol.mean()
    else:
        loss_mono = q.new_zeros(())

    # 3. Lagged flood threshold: extreme trailing-window rain should drive
    #    discharge toward (not below) the flood level within the next L steps.
    gate = torch.sigmoid(cfg.sharpness * (r_s - model.thresh_rain))
    if q.size(1) > L:
        thresh_viol = F.relu(model.thresh_flood - q[:, L:, :]) * gate[:, :-L, :]
        loss_thresh = thresh_viol.mean()
    else:
        loss_thresh = (F.relu(model.thresh_flood - q) * gate).mean()

    return {"mono": loss_mono, "thresh": loss_thresh, "nonneg": loss_nonneg}


def _snow_losses(outputs, x, model, cfg: PhysicsConfig):
    """Snow-driven physics: constraints keyed on the degree-day effective input
    ``w_eff`` (rain + snowmelt), which is available even when discharge is masked.

    * **nonneg**    -- predicted discharge stays above physical zero.
    * **recession** -- on low-input days (w_eff below its low level) discharge must
                       not rise: a soft baseflow-recession / storage-depletion prior.
    * **response**  -- when trailing effective input is extreme (snowmelt freshet),
                       discharge should approach the flood level within ``lag`` days.

    When an antecedent-moisture (API) feature is present (``model.api_idx``) the
    response prior is additionally **wetness-gated** -- a flood-level response is only
    demanded when the catchment is antecedently wet -- and a **dry_suppress** term
    penalizes discharge *rises* on extreme-rain-but-dry-antecedent days. This encodes
    the saturation-excess nonlinearity (rain runs off far more on already-wet soil)
    that governs the humid/Mediterranean basins where the unconditional response prior
    manufactured false flood peaks. All gates use fixed train-quantile thresholds.
    """
    q = outputs
    # Source the effective-input signal from the reconstructed w_eff = rain + melt
    # channel when it is present; below the snow line the router gates that feature
    # off, and w_eff reduces to rainfall, so fall back to the precipitation column.
    # The recession/response priors are thus identical in form everywhere -- only the
    # (redundant) snowpack input channel is dropped for non-snow basins.
    idx = model.w_eff_idx if getattr(model, "w_eff_idx", None) is not None else model.precip_idx
    w = x[:, :, idx : idx + 1]
    w_s = _moving_avg(w, cfg.smooth)
    L = max(1, cfg.lag)
    k = cfg.sharpness

    loss_nonneg = F.relu(model.q_zero_norm - q).mean()

    # Recession: penalize discharge increases on low-input days.
    dry_gate = torch.sigmoid(k * (model.weff_low - w_s))  # ~1 when input is low
    if q.size(1) > 1:
        rise = F.relu(q[:, 1:, :] - q[:, :-1, :])
        loss_recession = (rise * dry_gate[:, :-1, :]).mean()
    else:
        loss_recession = q.new_zeros(())

    # Snow-aware response: extreme trailing effective input -> elevated discharge later.
    hi_gate = torch.sigmoid(k * (w_s - model.weff_hi))

    # Antecedent-moisture gating (only where the API feature is present).
    api_idx = getattr(model, "api_idx", None)
    if api_idx is not None:
        a_s = _moving_avg(x[:, :, api_idx : api_idx + 1], cfg.smooth)
        wet_gate = torch.sigmoid(k * (a_s - model.api_hi))   # ~1 when antecedently wet
        dry_gate_api = torch.sigmoid(k * (model.api_lo - a_s))  # ~1 when antecedently dry
    else:
        wet_gate = None

    if q.size(1) > L:
        resp_gate = hi_gate[:, :-L, :]
        if wet_gate is not None:
            resp_gate = resp_gate * wet_gate[:, :-L, :]
        loss_response = (F.relu(model.thresh_flood - q[:, L:, :]) * resp_gate).mean()
    else:
        resp_gate = hi_gate if wet_gate is None else hi_gate * wet_gate
        loss_response = (F.relu(model.thresh_flood - q) * resp_gate).mean()

    losses = {"recession": loss_recession, "response": loss_response, "nonneg": loss_nonneg}

    # Anchored recession-rate band (opt-in via the 'recession_rate' weight). On dry days
    # discharge should follow the physical master recession Q_{t+1} ~= k*Q_t. Working in
    # p = q - q_zero_norm (affine anchor at physical zero) makes k*p a valid recession in
    # normalized space. Floor: forbid decaying FASTER than master (universal, safe even
    # for flat snow baseflow). Ceiling at k (force_decay, below snow line only): require
    # decay at least at the master rate.
    if cfg.weight("recession_rate") > 0 and float(model.recession_k) > 0 and q.size(1) > 1:
        kk = model.recession_k
        p = q - model.q_zero_norm
        floor = F.relu(kk * p[:, :-1, :] - p[:, 1:, :])
        band = floor
        if cfg.force_decay:
            band = band + F.relu(p[:, 1:, :] - kk * p[:, :-1, :])
        losses["recession_rate"] = (band * dry_gate[:, :-1, :]).mean()

    # Dry-antecedent suppression: on extreme-input days when the catchment is dry,
    # discharge should NOT surge (those are the false peaks the plain response prior made).
    if wet_gate is not None and q.size(1) > 1:
        q_rise = F.relu(q[:, 1:, :] - q[:, :-1, :])
        losses["dry_suppress"] = (q_rise * hi_gate[:, :-1, :] * dry_gate_api[:, :-1, :]).mean()

    return losses


def train_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=50,
    patience=10,
    device="cpu",
    physics: PhysicsConfig = None,
    grad_clip=1.0,
    loss_type="mse",
    verbose=True,
):
    """Train with early stopping; return (model, history). ``loss_type``: 'mse' | 'nse'."""
    if physics is None:
        physics = PhysicsConfig(use_physics=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lagrange multipliers (dual variables) -- plain tensors, NOT model params.
    # Keyed by whichever constraints the active mode emits (filled on first batch).
    lam = {}

    best_val = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    mode = "Physics-Informed" if physics.use_physics else "Control (MSE only)"
    if verbose:
        print(f"Training {type(model).__name__} [{mode}] on {device} ...")

    from collections import defaultdict
    for epoch in range(epochs):
        model.train()
        agg = defaultdict(float)
        nb = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            data_loss = _data_loss(outputs[:, -1, :], y, loss_type)
            loss = data_loss

            comps = {}
            if physics.use_physics:
                pl = physics_losses(outputs, x, model, physics)
                comps = {k: v.item() for k, v in pl.items()}
                for k in pl:
                    lam.setdefault(k, 0.0)
                if physics.adaptive:
                    # Primal: minimize data + Σ λ_i L_i (λ detached).
                    loss = loss + sum(lam[k] * pl[k] for k in pl)
                else:
                    loss = loss + sum(physics.weight(k) * pl[k] for k in pl)

            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Dual ascent on the multipliers (grow while violated, floor at 0).
            if physics.use_physics and physics.adaptive:
                for k in comps:
                    lam[k] = float(
                        min(physics.lambda_max, max(0.0, lam[k] + physics.dual_lr * comps[k]))
                    )

            agg["data"] += data_loss.item()
            agg["total"] += loss.item()
            for k in comps:
                agg[k] += comps[k]
            nb += 1

        for k in agg:
            agg[k] /= max(nb, 1)

        # Validation (data loss only -- physics is a training-time prior).
        model.eval()
        val_loss = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                val_loss += float(_data_loss(model(x)[:, -1, :], y, loss_type))
                vb += 1
        val_loss /= max(vb, 1)

        history.append(
            {"epoch": epoch + 1, "val": val_loss, "lam": dict(lam), **agg}
        )

        if val_loss < best_val:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} (val {val_loss:.5f})")
                break

        if verbose and (epoch + 1) % 10 == 0:
            extra = ""
            if physics.use_physics and comps:
                extra = " | " + " ".join(f"{k} {agg[k]:.4f}" for k in comps)
                if physics.adaptive:
                    extra += f" | λ {{{', '.join(f'{k}:{lam[k]:.2f}' for k in comps)}}}"
            print(f"  Epoch {epoch + 1}/{epochs} | val {val_loss:.5f}{extra}")

    model.load_state_dict(best_weights)
    if verbose:
        print(f"  Best val loss: {best_val:.5f}")
    return model, history
