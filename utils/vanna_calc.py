from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

@dataclass
class VannaConfig:
    contract_size_default: int = 100
    # If True: calls +, puts -
    # If False: calls -, puts +
    calls_positive_puts_negative: bool = True

def _option_sign(option_type: str, cfg: VannaConfig) -> int:
    ot = (option_type or "").lower()
    if cfg.calls_positive_puts_negative:
        return 1 if ot == "call" else -1
    return -1 if ot == "call" else 1

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _coerce_iv(series: pd.Series) -> pd.Series:
    """
    Return IV as decimal (e.g., 0.20 for 20%).
    Accepts either 0.20 or 20.0 style input.
    """
    iv = pd.to_numeric(series, errors="coerce")
    # Heuristic: if > 3, treat as percent
    iv = np.where(iv > 3.0, iv / 100.0, iv)
    return pd.Series(iv, index=series.index, dtype="float64")

def _pick_iv_column(df: pd.DataFrame) -> str | None:
    """
    Try common IV field names across broker feeds.
    """
    candidates = [
        "greeks.iv",
        "greeks.mid_iv",
        "greeks.implied_volatility",
        "implied_volatility",
        "iv",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_vanna(
    df: pd.DataFrame,
    *,
    spot: float,
    t_years: float,
    r: float,
    q: float,
    cfg: VannaConfig,
) -> pd.DataFrame:
    """
    Compute vanna from known greeks inputs (IV + BS), not relying on greeks.vanna.

    Black-Scholes:
      d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
      vega = S * exp(-qT) * phi(d1) * sqrt(T)
      vanna = (vega / S) * (1 - d1/(sigma*sqrt(T)))

    We use provider greeks.vega if present; otherwise compute vega from BS.
    Exposure:
      vanna_ex = sign * open_interest * vanna * contract_size
    """
    out = df.copy()

    if out.empty:
        out["vanna_ex"] = []
        return out

    S = float(spot)
    T = float(max(t_years, 0.0))
    # avoid division by 0 for 0DTE chains (keeps numbers finite)
    T = max(T, 1e-6)

    # Contract size
    if "contract_size" not in out.columns:
        out["contract_size"] = cfg.contract_size_default
    out["contract_size"] = pd.to_numeric(out["contract_size"], errors="coerce").fillna(cfg.contract_size_default)

    # OI
    if "open_interest" not in out.columns:
        out["open_interest"] = 0
    out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce").fillna(0.0)

    # Option type
    if "option_type" not in out.columns:
        out["option_type"] = "call"

    # Strike
    if "strike" not in out.columns:
        out["strike"] = np.nan
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")

    # IV
    iv_col = _pick_iv_column(out)
    if iv_col is None:
        # no IV => cannot compute BS-based vanna; return zeros safely
        out["iv"] = 0.0
        out["d1"] = 0.0
        out["vanna"] = 0.0
        out["vanna_ex"] = 0.0
        return out

    out["iv"] = _coerce_iv(out[iv_col]).fillna(0.0)

    # Provider vega (optional)
    has_vega = "greeks.vega" in out.columns
    if has_vega:
        out["vega"] = pd.to_numeric(out["greeks.vega"], errors="coerce").fillna(np.nan)
    else:
        out["vega"] = np.nan

    sqrtT = math.sqrt(T)
    disc_q = math.exp(-float(q) * T)

    # Vectorized d1
    K = out["strike"].to_numpy(dtype=float)
    sigma = out["iv"].to_numpy(dtype=float)

    # Guard rails
    valid = np.isfinite(K) & (K > 0) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(S) & (S > 0)

    d1 = np.zeros_like(K, dtype=float)
    vega_bs = np.zeros_like(K, dtype=float)
    vanna = np.zeros_like(K, dtype=float)

    if valid.any():
        lnSK = np.log(S / K[valid])
        sig = sigma[valid]
        d1_valid = (lnSK + (float(r) - float(q) + 0.5 * sig * sig) * T) / (sig * sqrtT)
        d1[valid] = d1_valid

        # phi(d1)
        phi = np.array([_norm_pdf(x) for x in d1_valid], dtype=float)

        # vega (per 1.00 vol, i.e., per 100% IV move; matches BS definition)
        vega_bs_valid = S * disc_q * phi * sqrtT
        vega_bs[valid] = vega_bs_valid

        # choose provider vega when present AND finite, else BS vega
        if has_vega:
            vega_in = out["vega"].to_numpy(dtype=float)
            vega_use = np.where(np.isfinite(vega_in), vega_in, vega_bs)
        else:
            vega_use = vega_bs

        # vanna = (vega/S) * (1 - d1/(sigma*sqrtT))
        vanna_valid = (vega_use[valid] / S) * (1.0 - (d1_valid / (sig * sqrtT)))
        vanna[valid] = vanna_valid

    out["d1"] = d1
    out["vanna"] = vanna

    # Sign convention
    out["sign"] = out["option_type"].apply(lambda t: _option_sign(t, cfg)).astype(float)

    # Exposure
    out["vanna_ex"] = out["sign"] * out["open_interest"] * out["vanna"] * out["contract_size"]

    return out

def vanna_by_strike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vanna exposure by strike into call vs put columns (like vex_by_strike).

    Returns:
      strike, call_vanna, put_vanna, net_vanna
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["strike", "call_vanna", "put_vanna", "net_vanna"])

    if not {"strike", "option_type", "vanna_ex"}.issubset(df.columns):
        return pd.DataFrame(columns=["strike", "call_vanna", "put_vanna", "net_vanna"])

    tmp = df[["strike", "option_type", "vanna_ex"]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["strike", "call_vanna", "put_vanna", "net_vanna"])

    tmp["option_type"] = tmp["option_type"].astype(str).str.lower()
    tmp["strike"] = pd.to_numeric(tmp["strike"], errors="coerce")
    tmp["vanna_ex"] = pd.to_numeric(tmp["vanna_ex"], errors="coerce").fillna(0.0)

    calls = (
        tmp[tmp["option_type"] == "call"]
        .groupby("strike", as_index=False)["vanna_ex"]
        .sum()
        .rename(columns={"vanna_ex": "call_vanna"})
    )
    puts = (
        tmp[tmp["option_type"] == "put"]
        .groupby("strike", as_index=False)["vanna_ex"]
        .sum()
        .rename(columns={"vanna_ex": "put_vanna"})
    )

    out = pd.merge(calls, puts, on="strike", how="outer").fillna(0.0)
    out["net_vanna"] = out["call_vanna"] + out["put_vanna"]

    return out.sort_values("strike").reset_index(drop=True)
