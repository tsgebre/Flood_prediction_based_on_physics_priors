"""Data loading and windowing for the CAMELS flood-prediction study.

Key design points (see README "Methodology" section):

* Explicit missingness channel. A discharge sensor failure is represented by
  (a) zeroing the normalized discharge value *and* (b) setting a companion binary
  "observed" flag to 0. Feeding the flag to the network lets it *know* the input
  is missing instead of confusing a masked value (0 in normalized space == the
  training-mean discharge) with a genuine average-flow day.

* Meteorological drivers. The forcing file contains precipitation *and*
  temperature, solar radiation and vapour pressure. These "sensors" keep working
  when the discharge gauge fails, so they are exactly what the model should fall
  back on. They are included as always-observed inputs.

* Reproducible, configurable. Date window, feature set, sequence length and an
  optional log-transform of discharge are all parameters.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Meteorological drivers that remain available during a discharge-sensor failure.
# Matched against the forcing-file column names by case-insensitive substring.
DEFAULT_MET_FEATURES = ("prcp", "tmax", "tmin", "srad", "vp")

# Physics (degree-day snow) features appended when physics_features=True.
PHYSICS_FEATURES = ("w_eff", "swe_recon")

# Antecedent-moisture (soil-wetness) decay constant for the API feature. A FIXED
# physical constant (~3-week e-folding), NOT learned and NOT discharge-derived, so
# the API channel stays 100% forcing-derived (leakage-free) and survives a discharge
# sensor failure. See src/router.py (api_feature gate) and src/train.py (wet-gate).
API_DECAY = 0.93


def master_recession(discharge, thr_quantile=0.6):
    """Master recession constant k in (0,1): median Q_{t+1}/Q_t on falling limbs
    above a low-flow threshold. Fit on the TRAIN split only (leakage-free static
    scalar). Q_{t+1} ~= k*Q_t is the physical baseflow-depletion law; ->1 = slow.
    """
    import numpy as _np
    q = _np.asarray(discharge, float)
    q = q[_np.isfinite(q)]
    if len(q) < 3 or not _np.any(q > 0):
        return float("nan")
    thr = _np.quantile(q[q > 0], thr_quantile)
    r = [q[i + 1] / q[i] for i in range(len(q) - 1)
         if q[i] > thr and 0 < q[i + 1] < q[i]]
    return float(_np.median(r)) if r else float("nan")


def antecedent_index(w_eff, k=API_DECAY):
    """Causal antecedent-moisture index: API_t = k*API_{t-1} + w_eff_t.

    A leaky-bucket soil-wetness proxy driven by effective water input (rain+melt).
    Each day uses only present/past forcing -> causal, no leakage. Reveals the
    saturation state that gates runoff generation (a mm of rain runs off far more
    when the catchment is already wet); the degree-day melt makes it snow-aware.
    """
    import numpy as _np
    w = _np.asarray(w_eff, float)
    api = _np.zeros_like(w)
    acc = 0.0
    for i in range(len(w)):
        acc = k * acc + w[i]
        api[i] = acc
    return api


def degree_day_snow(precip, tmean, ddf=2.5, t_snow=1.0, t_melt=0.0):
    """Offline temperature-index (degree-day) snow model.

    A degree-day
    model reconstructs the snowpack from *temperature and precipitation alone*
    (both survive a discharge-sensor failure) and yields the physically-correct
    runoff driver -- effective water input = rain + snowmelt.

    The snowpack's measured snowfall->melt storage delay is ~60-100 days on the
    snow-dominated basins (snow season length ~160-230 days; see
    results/state_memory/state_memory.json), far longer than the model's 30-day
    input window, so this feature injects information a windowed LSTM cannot
    reconstruct on its own.

    Causal: each day uses only that day's (P, T), so computing it on the full
    series before the train/val/test split introduces no future leakage.

    Returns dict of arrays: ``rain``, ``melt``, ``swe`` (reconstructed snowpack),
    ``w_eff`` (= rain + melt), all in mm/day (swe in mm).
    """
    import numpy as _np
    precip = _np.asarray(precip, float)
    tmean = _np.asarray(tmean, float)
    n = len(precip)
    swe = _np.zeros(n)
    melt = _np.zeros(n)
    rain = _np.zeros(n)
    store = 0.0
    for i in range(n):
        if tmean[i] < t_snow:      # precipitation falls as snow -> accumulate
            store += precip[i]
        else:                       # precipitation falls as rain
            rain[i] = precip[i]
        m = min(store, max(0.0, ddf * (tmean[i] - t_melt)))  # temperature-index melt
        store -= m
        melt[i] = m
        swe[i] = store
    return {"rain": rain, "melt": melt, "swe": swe, "w_eff": rain + melt}


class CamelsDataset(Dataset):
    """Sliding-window dataset over a single basin.

    Each item is ``{'x': (seq_len, n_features), 'y': (1,), 'mask': (seq_len, 1)}``
    where ``x`` columns are ``[*met_features, discharge, discharge_observed]``.
    The final column is the binary observed-flag for the (maskable) discharge
    input. ``y`` is the normalized discharge one step after the window.

    Parameters
    ----------
    df : DataFrame with the ``*_norm`` columns produced by
        :func:`load_and_preprocess_data`.
    feature_cols : normalized met-feature column names (always observed).
    seq_len : window length.
    is_train : if True, simulate sensor failure during training by randomly
        masking the discharge channel (data augmentation). Deterministic given
        ``generator``.
    train_mask_range : (low, high) range for the per-window discharge dropout
        probability used when ``is_train``.
    generator : optional ``torch.Generator`` for reproducible train masking.
    """

    def __init__(
        self,
        df,
        feature_cols,
        seq_len=30,
        is_train=False,
        train_mask_range=(0.0, 0.5),
        generator=None,
        use_mask_channel=True,
    ):
        self.seq_len = seq_len
        self.is_train = is_train
        self.train_mask_range = train_mask_range
        self.generator = generator
        self.feature_cols = list(feature_cols)
        self.use_mask_channel = use_mask_channel

        met = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        discharge = torch.tensor(df["discharge_norm"].values, dtype=torch.float32).unsqueeze(-1)
        self.met = met
        self.discharge = discharge
        # Target: discharge one step ahead of the window.
        self.y = discharge.clone()
        self.n_met = met.shape[1]
        # Column indices in the produced input tensor.
        self.q_idx = self.n_met
        self.mask_idx = self.n_met + 1 if use_mask_channel else None

    @property
    def n_features(self):
        # met features + discharge (+ observed-flag if enabled)
        return self.n_met + (2 if self.use_mask_channel else 1)

    def __len__(self):
        return len(self.discharge) - self.seq_len

    def __getitem__(self, idx):
        met_seq = self.met[idx : idx + self.seq_len]
        q_seq = self.discharge[idx : idx + self.seq_len].clone()
        y_target = self.y[idx + self.seq_len]

        observed = torch.ones_like(q_seq)
        if self.is_train:
            lo, hi = self.train_mask_range
            if self.generator is not None:
                p = lo + (hi - lo) * torch.rand(1, generator=self.generator).item()
                keep = torch.rand(q_seq.shape, generator=self.generator) >= p
            else:
                p = lo + (hi - lo) * torch.rand(1).item()
                keep = torch.rand(q_seq.shape) >= p
            keep = keep.float()
            q_seq = q_seq * keep  # masked entries -> 0 (neutral in normalized space)
            observed = keep

        parts = [met_seq, q_seq]
        if self.use_mask_channel:
            parts.append(observed)
        x_seq = torch.cat(parts, dim=-1)
        return {"x": x_seq, "y": y_target, "mask": observed}


def _find_col(columns, key):
    """Return the first column whose name contains ``key`` (case-insensitive)."""
    matches = [c for c in columns if key.lower() in c.lower()]
    if not matches:
        raise KeyError(f"No forcing column matching '{key}'. Available: {list(columns)}")
    return matches[0]


def load_and_preprocess_data(
    streamflow_path,
    forcing_path,
    start_date="1980-01-01",
    end_date="1999-12-31",
    met_features=DEFAULT_MET_FEATURES,
    log_transform=False,
    split=(0.70, 0.15),
    physics_features=False,
    api_feature=False,
    snow_params=None,
):
    """Load, merge, filter, split and normalize the basin data.

    Returns
    -------
    train_df, val_df, test_df : DataFrames with ``<feat>_norm`` and
        ``discharge_norm`` columns.
    meta : dict with keys
        ``feature_cols`` (normalized met-feature column names, always observed),
        ``scaler_precip``, ``scaler_discharge``,
        ``init_rain`` / ``init_flood`` (normalized 95th-pct thresholds),
        ``q_zero_norm`` (physical discharge 0 in normalized space, for the
        non-negativity constraint),
        ``log_transform``.
    """
    # 1. Streamflow: Basin Year Month Day Discharge QC
    df_flow = pd.read_csv(
        streamflow_path,
        sep=r"\s+",
        header=None,
        names=["basin", "year", "month", "day", "discharge", "qc"],
    )
    df_flow["date"] = pd.to_datetime(df_flow[["year", "month", "day"]])

    # 2. Forcing: 3 metadata rows then a header row.
    df_forcing = pd.read_csv(forcing_path, skiprows=3, sep=r"\s+")
    df_forcing.rename(columns={"Mnth": "month", "Year": "year", "Day": "day"}, inplace=True)
    df_forcing["date"] = pd.to_datetime(df_forcing[["year", "month", "day"]])

    # 3. Resolve requested meteorological columns.
    met_cols = {key: _find_col(df_forcing.columns, key) for key in met_features}
    keep_forcing = ["date"] + list(met_cols.values())
    df_merged = pd.merge(df_flow[["date", "discharge"]], df_forcing[keep_forcing], on="date")

    # Canonical names: <key> (e.g. 'prcp', 'tmax', ...).
    rename_map = {v: k for k, v in met_cols.items()}
    df_merged.rename(columns=rename_map, inplace=True)
    met_keys = list(met_features)

    # 4. Filter to the study window.
    mask = (df_merged["date"] >= start_date) & (df_merged["date"] <= end_date)
    df = df_merged.loc[mask].copy().sort_values("date").reset_index(drop=True)

    # 5. Missing discharge (-999 sentinel) -> interpolate. Met forcing has no gaps.
    df["discharge"] = df["discharge"].replace(-999, np.nan)
    df["discharge"] = df["discharge"].interpolate(method="linear", limit_direction="both")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if log_transform:
        # log1p is monotone and keeps discharge >= 0 mapping to >= 0.
        df["discharge"] = np.log1p(df["discharge"].clip(lower=0))

    # 5b. Degree-day snow physics (causal -> safe to compute before the split).
    #     Requires temperature; falls back to raw precip if tmax/tmin absent.
    snow_params = snow_params or {}
    if "tmax" in met_keys and "tmin" in met_keys:
        tmean = (df["tmax"].values + df["tmin"].values) / 2.0
        snow = degree_day_snow(df["prcp"].values, tmean, snow_params)
        df["w_eff"] = snow["w_eff"]
        df["swe_recon"] = snow["swe"]
    else:
        df["w_eff"] = df["prcp"].values
        df["swe_recon"] = 0.0

    # Antecedent-moisture index (causal, forcing-only). Built on w_eff so it is
    # snow-aware: winter precip accumulates as snowpack (w_eff~0) and does not read
    # as "wet" until it melts.
    df["api"] = antecedent_index(df["w_eff"].values)

    # 6. Temporal split (default 70/15/15), fitting scalers on train only.
    n_total = len(df)
    f_train, f_val = split
    n_train = int(n_total * f_train)
    n_val = int(n_total * (f_train + f_val))
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    test_df = df.iloc[n_val:].copy()

    # 7. Normalize met features + discharge (fit on train).
    feature_cols = []
    scalers = {}
    for key in met_keys:
        sc = StandardScaler().fit(train_df[[key]])
        for d in (train_df, val_df, test_df):
            d[f"{key}_norm"] = sc.transform(d[[key]])
        feature_cols.append(f"{key}_norm")
        scalers[key] = sc

    # Physics features (normalized like the rest); appended to inputs only if requested.
    w_eff_idx = None
    for key in PHYSICS_FEATURES:
        sc = StandardScaler().fit(train_df[[key]])
        for d in (train_df, val_df, test_df):
            d[f"{key}_norm"] = sc.transform(d[[key]])
        scalers[key] = sc
        if physics_features:
            if key == "w_eff":
                w_eff_idx = len(feature_cols)
            feature_cols.append(f"{key}_norm")

    # Antecedent-moisture feature (normalized like the rest; appended only if requested).
    # Thresholds are computed regardless so the wet-gate soft loss can use them.
    sc_api = StandardScaler().fit(train_df[["api"]])
    for d in (train_df, val_df, test_df):
        d["api_norm"] = sc_api.transform(d[["api"]])
    scalers["api"] = sc_api
    api_idx = None
    if api_feature:
        api_idx = len(feature_cols)
        feature_cols.append("api_norm")
    init_api_hi = float(train_df["api_norm"].quantile(0.85))  # antecedently-wet level
    init_api_lo = float(train_df["api_norm"].quantile(0.40))  # antecedently-dry level

    # Master recession constant (train-only, physical space) for the recession-rate band.
    recession_k = master_recession(train_df["discharge"].values)

    scaler_discharge = StandardScaler().fit(train_df[["discharge"]])
    for d in (train_df, val_df, test_df):
        d["discharge_norm"] = scaler_discharge.transform(d[["discharge"]])

    scaler_precip = scalers.get("prcp")

    # 8. Physical priors used by the physics loss.
    init_rain = float(train_df["prcp_norm"].quantile(0.95))
    init_flood = float(train_df["discharge_norm"].quantile(0.95))
    # Extreme effective-input threshold (95th pct of normalized w_eff) for the
    # snow-aware response constraint; low-input level for the recession gate.
    init_weff_hi = float(train_df["w_eff_norm"].quantile(0.95))
    weff_low = float(train_df["w_eff_norm"].quantile(0.40))
    # Physical discharge 0 expressed in normalized (and possibly log) space.
    zero_phys = np.log1p(0.0) if log_transform else 0.0
    q_zero_norm = float(scaler_discharge.transform([[zero_phys]])[0, 0])

    meta = {
        "feature_cols": feature_cols,
        "met_features": met_keys,
        "physics_features": bool(physics_features),
        "api_feature": bool(api_feature),
        "w_eff_idx": w_eff_idx,
        "api_idx": api_idx,
        "init_api_hi": init_api_hi,
        "init_api_lo": init_api_lo,
        "recession_k": recession_k if recession_k == recession_k else 0.0,
        "scaler_precip": scaler_precip,
        "scaler_discharge": scaler_discharge,
        "init_rain": init_rain,
        "init_flood": init_flood,
        "init_weff_hi": init_weff_hi,
        "weff_low": weff_low,
        "q_zero_norm": q_zero_norm,
        "log_transform": log_transform,
        "precip_idx": 0,  # prcp is the first met feature -> input column 0
    }
    return train_df, val_df, test_df, meta
