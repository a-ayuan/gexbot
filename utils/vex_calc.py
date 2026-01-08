from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class VexConfig:
    contract_size_default: int = 100
    # If True: calls +, puts -
    # If False: calls -, puts +
    calls_positive_puts_negative: bool = True

def _option_sign(option_type: str, cfg: VexConfig) -> int:
    ot = (option_type or "").lower()
    if cfg.calls_positive_puts_negative:
        return 1 if ot == "call" else -1
    return -1 if ot == "call" else 1

def compute_vex(df: pd.DataFrame, cfg: VexConfig) -> pd.DataFrame:
    """
    Compute per-contract vega exposure proxy ("vex") using Tradier greeks.vega.

    Definition used here:
        vex = sign * open_interest * vega * contract_size

    Notes:
    - Tradier's greeks.vega is used as-is. Different providers may scale vega differently
      (per 1 vol point vs per 1% vol). This chart is still useful directionally.
    - sign follows the same call/put convention toggle as GEX for consistency.
    """
    out = df.copy()
    out["contract_size"] = out.get("contract_size", np.nan)
    out["contract_size"] = out["contract_size"].fillna(cfg.contract_size_default)

    if "greeks.vega" not in out.columns:
        out["greeks.vega"] = 0.0
    out["greeks.vega"] = pd.to_numeric(out["greeks.vega"], errors="coerce").fillna(0.0)

    out["open_interest"] = pd.to_numeric(out.get("open_interest", pd.Series(0)), errors="coerce").fillna(0.0)

    if "option_type" not in out.columns:
        out["option_type"] = "call"

    out["sign"] = out["option_type"].map(lambda x: _option_sign(x, cfg)).astype(float)

    out["vex"] = (
        out["sign"]
        * out["open_interest"].astype(float)
        * out["greeks.vega"].astype(float)
        * out["contract_size"].astype(float)
    )

    return out

def vex_by_strike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vega exposure by strike.

    Output columns:
      strike, call_vex, put_vex, net_vex, total_oi
    """
    if df.empty:
        return pd.DataFrame(columns=["strike", "call_vex", "put_vex", "net_vex", "total_oi"])

    tmp = df.copy()
    tmp["strike"] = pd.to_numeric(tmp.get("strike", np.nan), errors="coerce")
    tmp = tmp.dropna(subset=["strike"])
    if tmp.empty:
        return pd.DataFrame(columns=["strike", "call_vex", "put_vex", "net_vex", "total_oi"])

    tmp["vex"] = pd.to_numeric(tmp["vex"] if "vex" in tmp.columns else pd.Series(0.0, index=tmp.index), errors="coerce").fillna(0.0)
    tmp["open_interest"] = pd.to_numeric(tmp.get("open_interest", pd.Series(0.0, index=tmp.index)), errors="coerce").fillna(0.0)

    call_mask = (pd.Series(tmp.get("option_type", ""), index=tmp.index).astype(str).str.lower() == "call")
    put_mask = (pd.Series(tmp.get("option_type", ""), index=tmp.index).astype(str).str.lower() == "put")

    calls = tmp[call_mask].groupby("strike", as_index=False).agg(
        call_vex=("vex", "sum"),
        call_oi=("open_interest", "sum"),
    )
    puts = tmp[put_mask].groupby("strike", as_index=False).agg(
        put_vex=("vex", "sum"),
        put_oi=("open_interest", "sum"),
    )

    out = pd.merge(calls, puts, on="strike", how="outer").fillna(0.0)
    out["net_vex"] = out["call_vex"] + out["put_vex"]
    out["total_oi"] = out.get("call_oi", 0.0) + out.get("put_oi", 0.0)

    out = out.sort_values("strike").reset_index(drop=True)
    return out