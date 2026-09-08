"""Per-basin static catchment attributes computed directly from the CAMELS
forcing + streamflow files (no separate attributes download required).

These are the standard signatures used by the physics router (`src.router`):
`frac_snow`, `p_mean`, `pet_mean`, `aridity`, `p_seasonality`, `high_prec_freq`,
`baseflow_index`, plus `elev_mean`, `area_km2`, `lat` read from the forcing header.

They approximate the official CAMELS v2.0 attributes (Addor et al. 2017); PET here
is Hargreaves (temperature + extraterrestrial radiation) rather than the calibrated
Priestley-Taylor of CAMELS, so `aridity` is close but not identical. That is fine for
the router, whose gates are soft functions of these attributes. To use the official
values instead, drop in `camels_clim/hydro/topo.txt` and read the columns.
"""

import glob
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Raw per-basin loading
# --------------------------------------------------------------------------- #
def read_forcing_header(forcing_path):
    """First three metadata lines: latitude (deg), elevation (m), area (m^2)."""
    with open(forcing_path) as f:
        lat = float(f.readline().strip())
        elev = float(f.readline().strip())
        area_m2 = float(f.readline().strip())
    return {"lat": lat, "elev_mean": elev, "area_km2": area_m2 / 1e6}


def load_basin_raw(gauge_id, data_dir):
    """Return (df, meta) for one basin over its full overlapping record.

    df columns: date, prcp, tmax, tmin, srad, discharge (cfs, -999 -> NaN).
    """
    fp = os.path.join(data_dir, f"{gauge_id}_lump_cida_forcing_leap.txt")
    sp = os.path.join(data_dir, f"{gauge_id}_streamflow_qc.txt")
    meta = read_forcing_header(fp)
    meta["gauge_id"] = str(gauge_id)

    f = pd.read_csv(fp, skiprows=3, sep=r"\s+").rename(
        columns={"Mnth": "month", "Year": "year", "Day": "day"})
    f["date"] = pd.to_datetime(f[["year", "month", "day"]])
    prcp = [c for c in f.columns if "prcp" in c.lower()][0]
    srad = [c for c in f.columns if "srad" in c.lower()]
    cols = {prcp: "prcp", "tmax(C)": "tmax", "tmin(C)": "tmin"}
    if srad:
        cols[srad[0]] = "srad"
    f = f.rename(columns=cols)

    q = pd.read_csv(sp, sep=r"\s+", header=None,
                    names=["basin", "year", "month", "day", "discharge", "qc"])
    q["date"] = pd.to_datetime(q[["year", "month", "day"]])
    q["discharge"] = q["discharge"].replace(-999, np.nan)

    keep = ["date", "prcp", "tmax", "tmin"] + (["srad"] if srad else [])
    df = pd.merge(f[keep], q[["date", "discharge"]], on="date").sort_values("date")
    return df.reset_index(drop=True), meta


# --------------------------------------------------------------------------- #
# Signature computations
# --------------------------------------------------------------------------- #
def _extraterrestrial_radiation(lat_deg, doy):
    """Ra (mm/day equivalent) for Hargreaves PET. FAO-56 formulation."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    dec = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1, 1)
    ws = np.arccos(x)
    ra_mj = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.408 * ra_mj  # MJ/m2/day -> mm/day


def hargreaves_pet(tmax, tmin, lat_deg, doy):
    tmean = (tmax + tmin) / 2.0
    ra = _extraterrestrial_radiation(lat_deg, doy)
    return np.maximum(0.0, 0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)) * ra)


def _seasonality(dates, series, doy):
    """Fit series = mean * (1 + delta*sin(2pi(t - s)/365)); return (delta, phase_s)."""
    w = 2 * np.pi * doy / 365.25
    A = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(A, series, rcond=None)
    c0, a, b = coef
    amp = np.hypot(a, b)
    phase = np.arctan2(b, a)  # such that a*sin+b*cos = amp*sin(w+phase)
    delta = amp / c0 if c0 != 0 else 0.0
    return delta, phase


def precip_seasonality(df):
    """Woods (2009) precipitation seasonality index (Addor 2017 definition).

    >0 : precip in phase with temperature (summer/warm-season precip)
    <0 : precip out of phase (winter/cool-season precip)
    """
    doy = df["date"].dt.dayofyear.values
    p = df["prcp"].values
    t = ((df["tmax"] + df["tmin"]) / 2.0).values
    dp, sp = _seasonality(df["date"], p, doy)
    dt, st = _seasonality(df["date"], t, doy)
    return float(dp * np.sign(dt) * np.cos(sp - st))


def baseflow_index(q, alpha=0.925, passes=3):
    """Lyne-Hollick recursive digital filter (Ladson et al. 2013), BFI = Qb/Q."""
    q = pd.Series(q).interpolate(limit_direction="both").values.astype(float)
    if len(q) < 3 or np.nansum(q) == 0:
        return np.nan
    bf = q.copy()
    for k in range(passes):
        forward = (k % 2 == 0)
        idx = range(1, len(q)) if forward else range(len(q) - 2, -1, -1)
        prev = idx.start - 1 if forward else len(q) - 1
        qf = np.zeros_like(q)
        b = bf.copy()
        j0 = 0 if forward else len(q) - 1
        qf[j0] = 0.0
        step = 1 if forward else -1
        for i in idx:
            qf[i] = alpha * qf[i - step] + (1 + alpha) / 2 * (b[i] - b[i - step])
            cand = b[i] - qf[i]
            bf[i] = cand if qf[i] > 0 else b[i]
            bf[i] = min(bf[i], b[i])
    return float(np.nansum(bf) / np.nansum(q))


def compute_attributes(gauge_id, data_dir, start="1980-01-01", end="2010-12-31",
                       t_snow=0.0):
    """Compute the router attribute vector for one basin."""
    df, meta = load_basin_raw(gauge_id, data_dir)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    dfx = df.dropna(subset=["discharge"]).copy()

    p = df["prcp"].values
    tmean = ((df["tmax"] + df["tmin"]) / 2.0).values
    doy = df["date"].dt.dayofyear.values

    p_mean = float(np.nanmean(p))
    pet = hargreaves_pet(df["tmax"].values, df["tmin"].values, meta["lat"], doy)
    pet_mean = float(np.nanmean(pet))
    snow_p = float(np.nansum(p[tmean < t_snow]))
    frac_snow = snow_p / float(np.nansum(p)) if np.nansum(p) > 0 else 0.0
    high_prec_freq = float(np.mean(p >= 5 * p_mean) * 365.25)

    # hydrologic signatures on the gauged record (specific discharge, mm/day)
    area_km2 = meta["area_km2"]
    q_cfs = dfx["discharge"].values
    q_mm = q_cfs * 0.0283168 * 86400.0 / (area_km2 * 1e6) * 1000.0  # cfs -> mm/day
    q_mean = float(np.nanmean(q_mm))
    runoff_ratio = q_mean / p_mean if p_mean > 0 else np.nan
    bfi = baseflow_index(q_mm)

    return {
        "gauge_id": str(gauge_id),
        "lat": round(meta["lat"], 3),
        "elev_mean": round(meta["elev_mean"], 1),
        "area_km2": round(area_km2, 1),
        "p_mean": round(p_mean, 3),
        "pet_mean": round(pet_mean, 3),
        "aridity": round(pet_mean / p_mean, 3) if p_mean > 0 else np.nan,
        "frac_snow": round(frac_snow, 3),
        "p_seasonality": round(precip_seasonality(df), 3),
        "high_prec_freq": round(high_prec_freq, 1),
        "q_mean": round(q_mean, 3),
        "runoff_ratio": round(runoff_ratio, 3),
        "baseflow_index": round(bfi, 3) if bfi == bfi else np.nan,
    }


def basins_in(data_dir):
    ids = sorted({os.path.basename(p).split("_")[0]
                  for p in glob.glob(os.path.join(data_dir, "*_streamflow_qc.txt"))})
    return ids


def attribute_table(data_dir, **kw):
    rows = [compute_attributes(g, data_dir, **kw) for g in basins_in(data_dir)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    tab = attribute_table(data_dir)
    print(tab.to_string(index=False))
