"""Attribute-gated physics router.

Maps a basin's static attributes (from :mod:`src.attributes`) to a per-basin
physics prescription: which constraints to apply and with what weight. This is
the "location-dependent physics" mechanism — the right physics is a function of
the catchment, not a global constant.

Design rules carried over from the audit (see ASSESSMENT.md):
* The router gates are **fixed, monotone functions of static attributes** — never
  weights learned by the training loss (that self-nullifies). Learnable adaptation
  is only allowed via the dual-ascent path in :mod:`src.train`.
* Each constraint keeps that basin's own physical thresholds (fixed buffers).

The constraints live in the `snow` physics family (:func:`src.train._snow_losses`),
which is general: it is keyed on the degree-day effective input `w_eff = rain + melt`
(melt ≈ 0 where there is no snow, so `w_eff` reduces to rainfall). The router sets:

* ``recession`` weight ∝ ``baseflow_index`` — slow-release basins (karst, sand,
  glacial outwash) must recede on input-free days; flashy basins must not be forced to.
* ``response``  weight ∝ a flood-propensity gate — flood-prone / flashy basins get a
  stronger "extreme input ⇒ elevated discharge" prior.
* ``nonneg``    fixed — discharge ≥ 0 everywhere.
* ``lag`` grows with baseflow index and catchment area (slower routing/memory).
* the snowmelt **feature** (`w_eff`, reconstructed `swe`) is always available; it only
  carries signal where ``frac_snow`` is non-trivial.
"""

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PhysicsSpec:
    mode: str = "snow"
    weights: dict = field(default_factory=dict)
    lag: int = 4
    smooth: int = 5
    physics_features: bool = True   # degree-day snowpack (w_eff, swe) input channel
    api_feature: bool = False       # antecedent-moisture (API) input channel + wet-gate loss
    force_decay: bool = False       # force-decay ceiling on the recession-rate band
    regime: str = ""
    gates: dict = field(default_factory=dict)  # for logging/inspection


def classify_regime(a):
    """Coarse regime label from attributes (for reporting only)."""
    fs, ar, bfi, ps = a["frac_snow"], a["aridity"], a["baseflow_index"], a["p_seasonality"]
    if fs > 0.4:
        return "snowmelt-montane"
    if 0.15 < fs <= 0.4:
        return "mixed rain-snow"
    if ar > 1.8 and bfi < 0.35:
        return "arid/flashy"
    if bfi > 0.75:
        return "groundwater/baseflow"
    if ar < 0.4:
        return "energy-limited maritime"
    if ps < -0.6:
        return "Mediterranean"
    return "humid rain"


def route_physics(a, base=0.15, beta=6.0, enable_amc=False, enable_recession=False):
    """Return a :class:`PhysicsSpec` for a basin's attribute dict ``a``.

    All gates are fixed monotone functions of static attributes.

    ``enable_amc`` turns on the antecedent-soil-moisture physics (API feature +
    wetness-gated response + dry-antecedent suppression). It is OFF by default:
    evaluated on the two saturation-controlled basins (see ``src.exp_amc``) it gives
    only a modest, non-significant NSE gain under total sensor failure and does not
    beat the no-physics baseline on flood peaks, so it is not shipped on by default.
    The proven default is the location-gated snow feature (``physics_features``).
    """
    fs = float(a["frac_snow"])
    bfi = float(a["baseflow_index"])
    hpf = float(a.get("high_prec_freq", 15.0))
    area = float(a.get("area_km2", 300.0))

    # Gates in [0, 1].
    g_snow_feature = float(_sigmoid(beta * (fs - 0.15)))     # snow signal strength
    g_recession = float(np.clip(bfi, 0.0, 1.0))              # slow-release propensity
    # flood propensity: frequent high-precip days AND flashy (low baseflow)
    g_response = float(np.clip(0.4 + 0.5 * (hpf - 12.0) / 15.0, 0.0, 1.0)) * float(np.clip(1.0 - bfi + 0.3, 0.2, 1.0))

    weights = {
        "recession": round(base * (0.2 + g_recession), 4),   # 0.2 floor so it is never fully off
        "response": round(base * (0.3 + g_response), 4),
        "nonneg": 0.3,
    }

    # Antecedent-moisture physics (opt-in via enable_amc): below the snow line
    # (frac_snow<0.4) runoff is saturation-controlled -- a mm of rain runs off far more
    # on already-wet soil. Turns on the API feature channel and the dry-antecedent
    # suppression term so the response prior no longer manufactures flood peaks from
    # rain on dry soil. The 3 deep-snow basins (frac_snow>0.4) are never affected.
    amc = bool(enable_amc) and fs < 0.4
    if amc:
        weights["dry_suppress"] = round(base * (0.3 + g_response), 4)

    # Anchored recession-rate band (opt-in via enable_recession): pins dry-day discharge
    # to the physical master recession, strongest in storage/baseflow-dominated basins
    # (weight ~ baseflow_index). Restricted to non-snow basins; the force-decay ceiling
    # only turns on below the snow line, where flat mid-winter baseflow is not a concern.
    force_decay = False
    if bool(enable_recession) and fs < 0.4:
        weights["recession_rate"] = round(base * float(np.clip(bfi, 0.0, 1.0)), 4)
        force_decay = fs < 0.15

    # Memory / routing: groundwater and large basins are slower.
    lag = int(np.clip(round(2 + 5 * bfi + np.log10(max(area, 1)) - 1.5), 2, 9))
    smooth = int(np.clip(round(3 + 4 * bfi), 3, 9))

    # Location-gated FEATURES: the reconstructed snowpack channel (w_eff, swe) only
    # carries information where snow is a real part of the water balance. Below the
    # snow line it is redundant (w_eff ~= rain, swe ~= 0) and only adds a correlated
    # input, so we drop it. The recession/response priors are generic rainfall-runoff
    # constraints and still apply -- _snow_losses sources them from rainfall when the
    # w_eff channel is absent (see src/train.py). frac_snow>0.15 matches the mixed
    # rain-snow line used by classify_regime and the g_snow_feature sigmoid center.
    snow_influenced = fs > 0.15

    return PhysicsSpec(
        mode="snow", weights=weights, lag=lag, smooth=smooth,
        physics_features=bool(snow_influenced),
        api_feature=bool(amc),
        force_decay=bool(force_decay),
        regime=classify_regime(a),
        gates={"g_snow_feature": round(g_snow_feature, 3),
               "g_recession": round(g_recession, 3),
               "g_response": round(g_response, 3),
               "snow_feature_on": bool(snow_influenced),
               "amc_on": bool(amc),
               "recession_band_on": "recession_rate" in weights},
    )


if __name__ == "__main__":
    import os
    from src import attributes as A
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    print(f"{'gauge':<10} {'regime':<24} {'w_rec':>6} {'w_resp':>7} {'lag':>4} {'smooth':>7}  gates")
    for g in A.basins_in(data_dir):
        a = A.compute_attributes(g, data_dir)
        s = route_physics(a)
        print(f"{g:<10} {s.regime:<24} {s.weights['recession']:>6.3f} "
              f"{s.weights['response']:>7.3f} {s.lag:>4d} {s.smooth:>7d}  {s.gates}")
