from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class VannaConfig:
    """
    Convention:
      - If calls_positive_puts_negative=True: calls contribute +, puts contribute -
      - If False: calls -, puts +
    """
    calls_positive_puts_negative: bool = True
    contract_size_default: int = 100

def _option_sign(option_type: str, cfg: VannaConfig) -> int:
    ot = (option_type or "").lower()
    if cfg.calls_positive_puts_negative:
        return 1 if ot == "call" else -1
    return -1 if ot == "call" else 1

def _pick_iv_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "greeks.mid_iv",
        "greeks.implied_volatility",
        "greeks.iv",
        "implied_volatility",
        "iv",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def contract_signed_vega_exposure(df_vex: pd.DataFrame, cfg: VannaConfig) -> pd.Series:
    """
    Signed vega hedging exposure PER CONTRACT.
    Preference:
      - if 'vex' exists, use it (assumed already signed by your VEX convention)
      - else compute: sign * open_interest * greeks.vega * contract_size
    """
    if df_vex is None or df_vex.empty:
        return pd.Series(dtype=float)

    if "vex" in df_vex.columns:
        return pd.to_numeric(df_vex["vex"], errors="coerce").fillna(0.0)

    if "greeks.vega" not in df_vex.columns:
        return pd.Series([0.0] * len(df_vex), index=df_vex.index, dtype=float)

    oi = pd.to_numeric(df_vex.get("open_interest", 0), errors="coerce").fillna(0.0)
    vega = pd.to_numeric(df_vex.get("greeks.vega", 0), errors="coerce").fillna(0.0)
    cs = pd.to_numeric(
        df_vex.get("contract_size", cfg.contract_size_default),
        errors="coerce",
    ).fillna(cfg.contract_size_default)

    opt = df_vex.get("option_type", "")
    if isinstance(opt, pd.Series):
        sign = opt.apply(lambda t: _option_sign(t, cfg)).astype(float)
    else:
        sign = pd.Series([_option_sign(opt, cfg)] * len(df_vex), index=df_vex.index, dtype=float)

    out = sign * oi * vega * cs
    return pd.to_numeric(out, errors="coerce").fillna(0.0)

def vanna_curve_for_expiration(
    df_vex: pd.DataFrame,
    dte: int,
    expiration: str,
    cfg: VannaConfig,
    n_iv_bins: int = 18,
) -> Optional[pd.DataFrame]:
    """
    Build ONE vanna curve for ONE expiration (one DTE):
      X = IV (binned)
      Y = net signed vega hedging exposure (summed) within each IV bin
    """
    if df_vex is None or df_vex.empty:
        return None

    iv_col = _pick_iv_column(df_vex)
    if iv_col is None:
        return None

    iv = pd.to_numeric(df_vex[iv_col], errors="coerce")
    expo = contract_signed_vega_exposure(df_vex, cfg=cfg)

    mask = iv.notna() & np.isfinite(iv)
    if mask.sum() < 5:
        return None

    d = pd.DataFrame(
        {
            "iv": iv[mask].astype(float),
            "expo": expo[mask].astype(float),
        }
    )

    # If IV doesn't vary, we can't form a curve
    iv_min = float(d["iv"].min())
    iv_max = float(d["iv"].max())
    if not np.isfinite(iv_min) or not np.isfinite(iv_max) or iv_max <= iv_min:
        # degenerate: single "bin" point
        return pd.DataFrame(
            {
                "expiration": [str(expiration)],
                "dte": [int(dte)],
                "iv_bin": [0],
                "iv": [float(iv_min)],
                "net_vega_hedge_exposure": [float(d["expo"].sum())],
            }
        )

    # Bin IV, then net exposure inside each bin
    bins = int(max(3, n_iv_bins))
    d["iv_bin"] = pd.cut(d["iv"], bins=bins, labels=False, include_lowest=True)

    g = (
        d.groupby("iv_bin", as_index=False)
        .agg(
            iv=("iv", "mean"),
            net_vega_hedge_exposure=("expo", "sum"),
        )
        .sort_values("iv")
        .reset_index(drop=True)
    )

    g["expiration"] = str(expiration)
    g["dte"] = int(dte)
    return g[["expiration", "dte", "iv_bin", "iv", "net_vega_hedge_exposure"]]
