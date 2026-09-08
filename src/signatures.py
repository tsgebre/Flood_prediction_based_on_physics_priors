"""Per-basin hydrological signatures for physics discovery.

Computes catchment signatures from raw CAMELS forcing + streamflow that reveal
which *additional* physical processes govern each basin's rainfall-runoff response
-- i.e. what physics beyond the degree-day snow family would help. Emphasis on
signatures that discriminate the failure modes we saw (Mediterranean / arid /
humid basins where the snow-family physics is neutral-to-harmful):

* water balance     -- runoff ratio Q/P, actual/ potential ET (Budyko position)
* storage / memory  -- baseflow index, master recession constant k
* flashiness        -- Richards-Baker index, P->Q peak lag
* nonlinearity      -- antecedent-moisture dependence of the event runoff coeff
                       (does a mm of rain yield more runoff when the catchment is
                        already wet? => soil-moisture threshold / saturation physics)
* seasonality       -- summer vs winter runoff ratio (Mediterranean dry-summer),
                       runoff-ratio vs PET (energy- vs water-limited)
* intermittency     -- fraction of near-zero-flow days (arid transmission losses)
* rain-on-snow      -- fraction of melt-season rain-on-snowpack days (mixed basins)

Run:  python -m src.signatures            # table + results/signatures.json
"""

import json
import os

import numpy as np
import pandas as pd

from src import attributes as A
from src.dataset import degree_day_snow

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CFS_TO_MM = 0.0283168 * 86400.0 * 1000.0  # cfs -> m^3/day -> mm over 1 m^2; divide by area_m2


def _load(gauge):
    sp = os.path.join(DATA, f"{gauge}_streamflow_qc.txt")
    fp = os.path.join(DATA, f"{gauge}_lump_cida_forcing_leap.txt")
    flow = pd.read_csv(sp, sep=r"\s+", header=None,
                       names=["basin", "year", "month", "day", "q_cfs", "qc"])
    flow["date"] = pd.to_datetime(flow[["year", "month", "day"]])
    forc = pd.read_csv(fp, skiprows=3, sep=r"\s+")
    forc.rename(columns={"Mnth": "month", "Year": "year", "Day": "day"}, inplace=True)
    forc["date"] = pd.to_datetime(forc[["year", "month", "day"]])
    pcol = [c for c in forc.columns if "prcp" in c.lower()][0]
    txc = [c for c in forc.columns if "tmax" in c.lower()][0]
    tnc = [c for c in forc.columns if "tmin" in c.lower()][0]
    df = pd.merge(flow[["date", "q_cfs"]], forc[["date", pcol, txc, tnc]], on="date")
    df.rename(columns={pcol: "prcp", txc: "tmax", tnc: "tmin"}, inplace=True)
    df = df[(df["date"] >= "1980-01-01") & (df["date"] <= "1999-12-31")].copy()
    df["q_cfs"] = df["q_cfs"].replace(-999, np.nan).interpolate(limit_direction="both")
    df = df.dropna().reset_index(drop=True)
    area_m2 = float(A.read_forcing_header(fp)["area_km2"]) * 1e6
    df["q_mm"] = df["q_cfs"] * CFS_TO_MM / area_m2
    df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0
    df["month_n"] = df["date"].dt.month
    return df, area_m2


def _recession_k(q, thr_quantile=0.6):
    """Master recession constant: median of Q_{t+1}/Q_t on falling limbs above a
    low-flow threshold. k in (0,1); closer to 1 => slower, more storage memory."""
    q = np.asarray(q, float)
    thr = np.quantile(q[q > 0], thr_quantile) if np.any(q > 0) else 0.0
    ratios = []
    for i in range(len(q) - 1):
        if q[i] > thr and q[i + 1] < q[i] and q[i] > 0:
            ratios.append(q[i + 1] / q[i])
    return float(np.median(ratios)) if ratios else float("nan")


def _rb_flashiness(q):
    q = np.asarray(q, float)
    s = np.sum(q)
    return float(np.sum(np.abs(np.diff(q))) / s) if s > 0 else float("nan")


def _peak_lag(p, q, maxlag=7):
    p = np.asarray(p, float) - np.mean(p)
    q = np.asarray(q, float) - np.mean(q)
    best, blag = -np.inf, 0
    for L in range(0, maxlag + 1):
        if L == 0:
            c = np.corrcoef(p, q)[0, 1]
        else:
            c = np.corrcoef(p[:-L], q[L:])[0, 1]
        if c > best:
            best, blag = c, L
    return blag, float(best)


def _api(precip, k=0.9):
    """Antecedent precipitation index: API_t = k*API_{t-1} + P_t (soil-wetness proxy)."""
    p = np.asarray(precip, float)
    api = np.zeros_like(p)
    acc = 0.0
    for i in range(len(p)):
        acc = k * acc + p[i]
        api[i] = acc
    return api


def _antecedent_dependence(df):
    """Does the event runoff coefficient rise with antecedent wetness?

    On wet days (P>5mm), regress next-3-day runoff increment / P on the *prior* API.
    Returns Spearman-like split: runoff coeff for dry-antecedent vs wet-antecedent
    events, and their ratio. Ratio >> 1 => strong soil-moisture/saturation control
    (a mm of rain runs off far more when the catchment is already wet)."""
    p = df["prcp"].values
    q = df["q_mm"].values
    api = _api(p)
    # shift API to be *antecedent* (exclude today's rain)
    api_prev = np.concatenate([[0.0], api[:-1]])
    ev = np.where(p > 5.0)[0]
    ev = ev[(ev > 2) & (ev < len(p) - 4)]
    if len(ev) < 30:
        return {"rc_dry": float("nan"), "rc_wet": float("nan"), "rc_ratio": float("nan"),
                "n_events": int(len(ev))}
    # runoff response = increase in q over the following 3 days above pre-event baseline
    resp = np.array([max(0.0, np.mean(q[i + 1:i + 4]) - q[i - 1]) for i in ev])
    rc = resp / p[ev]
    ap = api_prev[ev]
    med = np.median(ap)
    dry = rc[ap <= med]
    wet = rc[ap > med]
    rc_dry = float(np.median(dry)) if len(dry) else float("nan")
    rc_wet = float(np.median(wet)) if len(wet) else float("nan")
    ratio = float(rc_wet / rc_dry) if rc_dry and rc_dry > 1e-9 else float("nan")
    return {"rc_dry": rc_dry, "rc_wet": rc_wet, "rc_ratio": ratio, "n_events": int(len(ev))}


def _seasonal(df):
    """Summer vs winter runoff ratio and their contrast (Mediterranean signature)."""
    g = df.groupby("month_n").agg(p=("prcp", "sum"), q=("q_mm", "sum"))
    summer = [6, 7, 8, 9]
    winter = [11, 12, 1, 2, 3]
    rs = g.loc[g.index.isin(summer), "q"].sum() / max(g.loc[g.index.isin(summer), "p"].sum(), 1e-9)
    rw = g.loc[g.index.isin(winter), "q"].sum() / max(g.loc[g.index.isin(winter), "p"].sum(), 1e-9)
    return {"rr_summer": float(rs), "rr_winter": float(rw),
            "rr_winter_summer_ratio": float(rw / rs) if rs > 1e-9 else float("nan")}


def _rain_on_snow(df):
    """Fraction of days with rain falling on an existing snowpack during melt season."""
    snow = degree_day_snow(df["prcp"].values, df["tmean"].values)
    swe = snow["swe"]; rain = snow["rain"]
    ros = (rain > 5.0) & (swe > 10.0)
    return float(np.mean(ros))


def _low_flow_frac(q):
    q = np.asarray(q, float)
    thr = 0.1 * np.median(q[q > 0]) if np.any(q > 0) else 0.0
    return float(np.mean(q <= thr))


def signatures(gauge):
    df, area_m2 = _load(gauge)
    attrs = A.compute_attributes(gauge, DATA)
    P = df["prcp"].mean() * 365.25
    Q = df["q_mm"].mean() * 365.25
    rr = float(Q / P) if P > 0 else float("nan")
    lag, lagcorr = _peak_lag(df["prcp"].values, df["q_mm"].values)
    sig = {
        "gauge": gauge,
        "regime": None,  # filled by caller via router if desired
        "frac_snow": round(float(attrs["frac_snow"]), 3),
        "aridity_PET_P": round(float(attrs["aridity"]), 3),
        "baseflow_index": round(float(attrs["baseflow_index"]), 3),
        "p_seasonality": round(float(attrs.get("p_seasonality", float("nan"))), 3),
        "area_km2": round(area_m2 / 1e6, 1),
        "P_mm_yr": round(float(P), 1),
        "Q_mm_yr": round(float(Q), 1),
        "runoff_ratio": round(rr, 3),
        "et_fraction": round(float(1 - rr), 3),          # (P-Q)/P ~ ET/P
        "recession_k": round(_recession_k(df["q_mm"].values), 4),
        "rb_flashiness": round(_rb_flashiness(df["q_mm"].values), 3),
        "peak_lag_days": int(lag),
        "peak_lag_corr": round(lagcorr, 3),
        "low_flow_frac": round(_low_flow_frac(df["q_mm"].values), 3),
        "rain_on_snow_frac": round(_rain_on_snow(df), 4),
    }
    sig.update({k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in _antecedent_dependence(df).items()})
    sig.update({k: round(v, 3) for k, v in _seasonal(df).items()})
    return sig


def main():
    from src import router
    rows = []
    for g in A.basins_in(DATA):
        s = signatures(g)
        s["regime"] = router.route_physics(A.compute_attributes(g, DATA)).regime
        rows.append(s)
        print(f"{g} done", flush=True)
    rows.sort(key=lambda r: -r["frac_snow"])
    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    json.dump(rows, open(os.path.join(outdir, "signatures.json"), "w"), indent=2)

    cols = ["gauge", "regime", "frac_snow", "aridity_PET_P", "runoff_ratio", "et_fraction",
            "baseflow_index", "recession_k", "rb_flashiness", "peak_lag_days",
            "low_flow_frac", "rc_ratio", "rr_winter_summer_ratio", "rain_on_snow_frac"]
    print("\n" + " ".join(f"{c:>10}"[:10] for c in cols))
    for r in rows:
        print(" ".join(f"{str(r[c]):>10}"[:10] for c in cols))
    print("\nSaved -> results/signatures.json")


if __name__ == "__main__":
    main()
