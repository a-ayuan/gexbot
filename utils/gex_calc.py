from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class GexConfig:
    contract_size_default: int = 100
    # If True: calls +, puts -
    # If False: calls -, puts + (some desks flip “dealer positioning” assumptions)
    calls_positive_puts_negative: bool = True

def options_to_df(options: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(options).copy()

    # Normalize common fields (Tradier returns snake_case keys for many)
    # Expected keys include: strike, option_type, open_interest, volume, contract_size, greeks{gamma,...}
    if "greeks" in df.columns:
        greeks_df = pd.json_normalize(df["greeks"].tolist()).add_prefix("greeks.")
        df = df.drop(columns=["greeks"]).join(greeks_df)

    # Coerce numerics
    for col in ["strike", "open_interest", "volume", "contract_size", "greeks.gamma"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Some chains may omit contract_size; default to 100
    if "contract_size" not in df.columns:
        df["contract_size"] = np.nan

    return df

import pandas as pd

def compute_gex(df: pd.DataFrame, spot: float, cfg: "GexConfig") -> pd.DataFrame:
    out = df.copy()

    out["contract_size"] = out.get("contract_size", pd.Series(index=out.index)).fillna(cfg.contract_size_default)

    if "greeks.gamma" not in out.columns:
        out["greeks.gamma"] = 0.0
    out["greeks.gamma"] = out["greeks.gamma"].fillna(0.0)

    out["open_interest"] = out.get("open_interest", pd.Series(0, index=out.index)).fillna(0)

    out["volume"] = out.get("volume", pd.Series(0, index=out.index)).fillna(0)

    # Compute net gamma exposure per 1% move:
    # gex = ((call_oi*call_gamma) - (put_oi*put_gamma)) * contract_size * spot^2 * 0.01
    is_call = out["option_type"].astype(str).str.upper().isin(["C", "CALL"])
    is_put  = out["option_type"].astype(str).str.upper().isin(["P", "PUT"])

    call_term = (out["open_interest"] * out["greeks.gamma"]).where(is_call, 0.0)
    put_term  = (out["open_interest"] * out["greeks.gamma"]).where(is_put, 0.0)

    out["gex"] = (call_term - put_term) * out["contract_size"] * (spot ** 2) * 0.01

    return out

def gex_by_strike(gex_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        gex_df.groupby("strike", as_index=False)
        .agg(
            net_gex=("gex", "sum"),
            call_gex=("gex", lambda s: s[gex_df.loc[s.index, "option_type"].str.lower() == "call"].sum()),
            put_gex=("gex", lambda s: s[gex_df.loc[s.index, "option_type"].str.lower() == "put"].sum()),
            total_oi=("open_interest", "sum"),
            total_vol=("volume", "sum"),
        )
        .sort_values("strike")
    )
    return agg

def call_put_walls(strike_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Call wall: strike with maximum net_gex (largest positive).
    Put wall: strike with minimum net_gex (most negative).
    """
    if strike_df.empty:
        return None, None
    
    call_wall = pd.to_numeric(strike_df.loc[strike_df["net_gex"].idxmax(), "strike"], errors="coerce")
    call_wall = float(call_wall) if pd.notna(call_wall) else None
    put_wall = pd.to_numeric(strike_df.loc[strike_df["net_gex"].idxmin(), "strike"], errors="coerce")
    put_wall = float(put_wall) if pd.notna(put_wall) else None
    
    return call_wall, put_wall

from typing import Optional
import numpy as np
import pandas as pd

def find_zero_gamma(
    strike_df: pd.DataFrame,
    spot: float,
    *,
    tie_breaker: str = "closest_to_spot",  # or "lowest_strike" / "highest_strike"
    eps: float = 1e-12,
) -> Optional[float]:
    """
    Zero gamma = underlying level where net_gex crosses 0.
    We treat:
      - exact/near zeros as valid roots
      - sign-change intervals as roots via linear interpolation
    "Most significant" root defaults to closest-to-spot (matches standard dealer-regime interpretation).

    tie_breaker:
      - closest_to_spot (default)
      - lowest_strike
      - highest_strike
    """
    if strike_df is None or strike_df.empty:
        return None

    sdf = strike_df[["strike", "net_gex"]].dropna()
    if sdf.empty:
        return None

    # sort, coerce numeric, and collapse duplicate strikes safely
    sdf = (
        sdf.assign(
            strike=pd.to_numeric(sdf["strike"], errors="coerce"),
            net_gex=pd.to_numeric(sdf["net_gex"], errors="coerce"),
        )
        .dropna()
        .sort_values("strike")
        .groupby("strike", as_index=False)["net_gex"].sum()
    )

    if sdf.empty:
        return None

    x = sdf["strike"].to_numpy(dtype=float)
    y = sdf["net_gex"].to_numpy(dtype=float)

    # If everything is ~0, zero gamma isn't uniquely defined
    if np.all(np.isfinite(y)) and np.all(np.abs(y) <= eps):
        # best practical choice: closest strike to spot
        return float(x[np.argmin(np.abs(x - float(spot)))])

    roots = []

    # 1) exact / near zeros at discrete strikes
    near_zero_idx = np.where(np.abs(y) <= eps)[0]
    for j in near_zero_idx:
        roots.append(float(x[j]))

    # 2) sign-change crossings (bracketing intervals), ignore intervals touching ~0 already captured
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]

        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue

        # skip if either endpoint is ~0 (already added)
        if abs(y0) <= eps or abs(y1) <= eps:
            continue

        # strict sign change
        if y0 * y1 < 0:
            dy = (y1 - y0)
            dx = (x1 - x0)
            if abs(dx) <= eps or abs(dy) <= eps:
                continue
            # linear interpolation: x0 + (0 - y0) * (x1-x0)/(y1-y0)
            root = x0 + (-y0) * dx / dy
            roots.append(float(root))

    if not roots:
        return None

    roots = np.array(roots, dtype=float)

    # De-dup very-close roots (can happen if near-zero + crossing around same region)
    roots.sort()
    deduped = [roots[0]]
    for r in roots[1:]:
        if abs(r - deduped[-1]) > 1e-8:
            deduped.append(r)
    roots = np.array(deduped, dtype=float)

    # Choose "most significant" root:
    # default = closest to spot, else extreme strike.
    if tie_breaker == "closest_to_spot":
        return float(roots[np.argmin(np.abs(roots - float(spot)))])
    if tie_breaker == "lowest_strike":
        return float(np.min(roots))
    if tie_breaker == "highest_strike":
        return float(np.max(roots))

    # fallback
    return float(roots[np.argmin(np.abs(roots - float(spot)))])

def volume_triggers(strike_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Simple “volume triggers”: top strikes by total_vol, plus vol/oi ratio.
    """
    if strike_df.empty:
        return strike_df

    out = strike_df.copy()
    out["vol_oi"] = out["total_vol"] / out["total_oi"].replace(0, np.nan)
    out = out.sort_values("total_vol", ascending=False).head(top_n)
    return out

def calculate_max_pain(df_opts: pd.DataFrame) -> float | None:
    """
    Standard max pain:
    For each candidate settlement strike K:
      Sum over all calls: max(0, K - strike) * OI * contract_size
      Sum over all puts : max(0, strike - K) * OI * contract_size
    Max pain = K that minimizes total payout
    """
    if df_opts.empty or "strike" not in df_opts.columns:
        return None

    df = df_opts.copy()
    df["open_interest"] = df["open_interest"].fillna(0)
    df["contract_size"] = df["contract_size"].fillna(100)

    strikes = np.sort(df["strike"].unique())
    payouts = []

    # Prefilter once
    calls = df[df["option_type"].str.lower() == "call"]
    puts = df[df["option_type"].str.lower() == "put"]

    for K in strikes:
        call_payout = ((K - calls["strike"]).clip(lower=0) * calls["open_interest"] * calls["contract_size"]).sum()
        put_payout = ((puts["strike"] - K).clip(lower=0) * puts["open_interest"] * puts["contract_size"]).sum()
        payouts.append(call_payout + put_payout)

    if not payouts:
        return None

    return float(strikes[int(np.argmin(payouts))])
