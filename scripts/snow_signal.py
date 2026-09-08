"""Diagnostic: basin 01013500 is snowmelt-driven, so a degree-day snow
model turns a weak predictor (raw precipitation) into a strong one
(effective water input = rain + snowmelt).

    python scripts/snow_signal.py

Prints: correlation of discharge with raw precip vs reconstructed effective input
at several lags, and the monthly climatology showing the April-May freshet.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.dataset import degree_day_snow  # noqa: E402

STREAMFLOW = os.path.join(REPO, "data", "01013500_streamflow_qc.txt")
FORCING = os.path.join(REPO, "data", "01013500_lump_cida_forcing_leap.txt")


def load(start="1980-01-01", end="1999-12-31"):
    f = pd.read_csv(FORCING, skiprows=3, sep=r"\s+").rename(
        columns={"Mnth": "month", "Year": "year", "Day": "day"})
    f["date"] = pd.to_datetime(f[["year", "month", "day"]])
    q = pd.read_csv(STREAMFLOW, sep=r"\s+", header=None,
                    names=["b", "year", "month", "day", "discharge", "qc"])
    q["date"] = pd.to_datetime(q[["year", "month", "day"]])
    m = pd.merge(f, q[["date", "discharge"]], on="date")
    m = m[(m.date >= start) & (m.date <= end)].reset_index(drop=True)
    m["discharge"] = m["discharge"].replace(-999, np.nan).interpolate(limit_direction="both")
    return m


def main():
    m = load()
    tmean = (m["tmax(C)"].values + m["tmin(C)"].values) / 2.0
    snow = degree_day_snow(m["prcp(mm/day)"].values, tmean)
    m["w_eff"] = snow["w_eff"]
    m["melt"] = snow["melt"]
    m["swe_recon"] = snow["swe"]

    print("Correlation of discharge with predictor (by lag, days):")
    print(f"  {'lag':>3} | {'raw precip':>12} | {'effective input':>16}")
    for lag in range(0, 8):
        cP = m["discharge"].corr(m["prcp(mm/day)"].shift(lag))
        cW = m["discharge"].corr(m["w_eff"].shift(lag))
        print(f"  {lag:>3} | {cP:>12.3f} | {cW:>16.3f}")

    print("\nMonthly climatology (Q cfs, reconstructed melt mm/d, reconstructed SWE mm):")
    clim = m.groupby(m.date.dt.month).agg(
        Q=("discharge", "mean"), melt=("melt", "mean"), swe=("swe_recon", "mean"))
    print(clim.round(1).to_string())
    print("\nThe April-May discharge peak with flat precip is the snowmelt freshet:")
    print("raw precip is a poor predictor; the degree-day effective input is not.")


if __name__ == "__main__":
    main()
