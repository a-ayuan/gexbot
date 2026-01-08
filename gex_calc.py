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

def find_zero_gamma(
    strike_df: pd.DataFrame,
    spot: float,
    *,
    tie_breaker: str = "closest_to_spot",  # or "lowest_strike" / "highest_strike"
) -> Optional[float]:
    """
    "Zero gamma" = strike where net_gex crosses 0.
    "Optimal" here = crossing with the strongest local flip (max |slope|) rather than closest to spot.

    slope ≈ (y1 - y0) / (x1 - x0) for the two strikes bracketing the sign change.
    """
    if strike_df.empty:
        return None

    sdf = strike_df[["strike", "net_gex"]].dropna().sort_values("strike")
    if sdf.empty:
        return None

    x = sdf["strike"].to_numpy(dtype=float)
    y = sdf["net_gex"].to_numpy(dtype=float)

    # If any strike is exactly zero net_gex, treat those as valid roots and choose the "best" among them
    zero_idx = np.where(y == 0)[0]
    if zero_idx.size:
        roots = x[zero_idx]
        if tie_breaker == "closest_to_spot":
            return float(roots[np.argmin(np.abs(roots - spot))])
        if tie_breaker == "lowest_strike":
            return float(np.min(roots))
        if tie_breaker == "highest_strike":
            return float(np.max(roots))
        return float(roots[0])

    # Find sign changes (bracketing intervals)
    sign = np.sign(y)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(idx) == 0:
        return None

    best_root = None
    best_score = -float("inf")  # higher is better (|slope|)
    best_dist = float("inf")

    for i in idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]

        dx = (x1 - x0)
        dy = (y1 - y0)
        if dx == 0 or dy == 0:
            continue  # degenerate (shouldn't happen with sorted distinct strikes, but safe)

        # Linear interpolation root
        root = x0 + (0 - y0) * dx / dy

        # "Optimal" score: strongest local flip
        slope_mag = abs(dy / dx)

        # Tie-breakers
        dist = abs(root - spot)

        better = False
        if slope_mag > best_score:
            better = True
        elif slope_mag == best_score:
            if tie_breaker == "closest_to_spot" and dist < best_dist:
                better = True
            elif tie_breaker == "lowest_strike" and (best_root is None or root < best_root):
                better = True
            elif tie_breaker == "highest_strike" and (best_root is None or root > best_root):
                better = True

        if better:
            best_score = slope_mag
            best_dist = dist
            best_root = float(root)

    return best_root

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
