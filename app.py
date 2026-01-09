import os
from datetime import datetime, date
from dateutil import tz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from tradier_client import TradierClient
from utils.gex_calc import (
    GexConfig,
    compute_gex,
    options_to_df,
    gex_by_strike,
    call_put_walls,
    find_zero_gamma,
    calculate_max_pain,
)

from utils.vex_calc import (
    VexConfig,
    compute_vex,
    vex_by_strike,
)

from utils.vanna_calc import (
    VannaConfig,
    vanna_curve_for_expiration,
)

st.set_page_config(page_title="GEX Dashboard", layout="wide")
st.title("GEX Dashboard")

# ---------------------------
# Helpers
# ---------------------------
def get_token() -> str:
    if "TRADIER_TOKEN" in st.secrets:
        return st.secrets["TRADIER_TOKEN"]
    return os.getenv("TRADIER_TOKEN", "")

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def fmt_compact(n: float | int | None) -> str:
    """Format numbers like 1.12M, 100.23k, -2.30B, with 2 decimals."""
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return "—"
    if not np.isfinite(x):
        return "—"

    sign = "-" if x < 0 else ""
    ax = abs(x)

    if ax >= 1_000_000_000_000:
        return f"{sign}{ax/1_000_000_000_000:.2f}T"
    if ax >= 1_000_000_000:
        return f"{sign}{ax/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{sign}{ax/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{sign}{ax/1_000:.2f}K"
    return f"{x:,.2f}"

def strikes_window_around_spot(all_strikes: np.ndarray, spot: float, n_each_side: int) -> tuple[float, float]:
    """
    Pick a strike window that includes the nearest strike to spot,
    plus n_each_side strikes below and above (total = 2*n_each_side + 1).
    Returns (strike_lo, strike_hi).
    """
    if all_strikes.size == 0:
        return (spot, spot)

    s = np.sort(np.unique(all_strikes.astype(float)))
    i = int(np.argmin(np.abs(s - float(spot))))
    n_each_side = max(1, int(n_each_side))

    lo_i = clamp_int(i - n_each_side, 0, len(s) - 1)
    hi_i = clamp_int(i + n_each_side, 0, len(s) - 1)
    return float(s[lo_i]), float(s[hi_i])

def _extract_spot_from_quote(q: dict) -> float:
    try:
        return float(q.get("last") or q.get("bid") or q.get("ask") or np.nan)
    except Exception:
        return np.nan

def _is_valid_quote_for_symbol(q: object, symbol: str) -> bool:
    """
    Best-effort validation that a ticker is real/recognized.
    - If Tradier returns a quote with a usable price => valid
    - If the quote has a symbol field matching input => likely valid
    - If quote indicates an error/unknown symbol => invalid
    """
    if not isinstance(q, dict) or not q:
        return False

    # Common error shapes
    err_txt = ""
    for k in ("error", "errors", "message", "fault", "detail", "description"):
        v = q.get(k)
        if isinstance(v, (str, int, float)):
            err_txt += f" {v}"
        elif isinstance(v, dict):
            err_txt += " " + " ".join(str(x) for x in v.values())
        elif isinstance(v, list):
            err_txt += " " + " ".join(str(x) for x in v)

    if err_txt:
        lowered = err_txt.lower()
        if "unknown" in lowered and "symbol" in lowered:
            return False
        if "invalid" in lowered and ("symbol" in lowered or "ticker" in lowered):
            return False
        if "not found" in lowered and ("symbol" in lowered or "ticker" in lowered):
            return False

    # If spot is present and finite, that's a strong signal the symbol is valid
    spot = _extract_spot_from_quote(q)
    if np.isfinite(spot) and spot > 0:
        return True

    # Fallback: some responses include the resolved symbol
    qs = str(q.get("symbol") or "").upper().strip()
    if qs and qs == str(symbol).upper().strip():
        return True

    return False

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote(token: str, symbol: str):
    c = TradierClient(token=token)
    try:
        return c.get_quote(symbol)
    except Exception:
        return {}

@st.cache_data(ttl=5, show_spinner=False)
def fetch_snapshot(token: str, symbol: str, expiration: str):
    c = TradierClient(token=token)
    try:
        q = c.get_quote(symbol)
    except Exception:
        q = {}
    spot = _extract_spot_from_quote(q)

    try:
        chain = c.get_chain(symbol=symbol, expiration=expiration, greeks=True)
    except Exception:
        chain = []

    return spot, chain, q

@st.cache_data(ttl=60, show_spinner=False)
def fetch_expirations(token: str, symbol: str):
    c = TradierClient(token=token)
    try:
        return c.get_expirations(symbol)
    except Exception:
        return []

@st.cache_data(ttl=60, show_spinner=False)
def fetch_chain(token: str, symbol: str, expiration: str):
    c = TradierClient(token=token)
    try:
        return c.get_chain(symbol=symbol, expiration=expiration, greeks=True)
    except Exception:
        return []

def _safe_parse_expiration(expiration: str) -> date | None:
    try:
        return datetime.strptime(str(expiration), "%Y-%m-%d").date()
    except Exception:
        return None

def _closest_expiration_by_dte(exp_df: pd.DataFrame, target_dte: int) -> str:
    """
    Pick the expiration whose DTE is closest to target_dte.
    exp_df must have columns: expiration (str), dte (int)
    """
    if exp_df.empty:
        return ""
    target_dte = int(target_dte)
    i = (exp_df["dte"].astype(int) - target_dte).abs().idxmin()
    return str(exp_df.loc[i, "expiration"])

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    token = get_token()
    if not token:
        st.error("Missing Tradier token. Set TRADIER_TOKEN env var or .streamlit/secrets.toml.")
        st.stop()

    # Session state for persistent DTE across ticker changes
    if "prev_symbol" not in st.session_state:
        st.session_state.prev_symbol = "SPY"
    if "selected_dte" not in st.session_state:
        st.session_state.selected_dte = 0  # persists across tickers

    symbol = st.text_input("Ticker", value=st.session_state.prev_symbol).upper().strip()

    q_check = fetch_quote(token, symbol)
    if not _is_valid_quote_for_symbol(q_check, symbol):
        st.warning(
            f"Invalid ticker: **{symbol}**. Please enter a valid symbol (e.g., SPY, QQQ, AAPL)."
        )
        st.stop()

    expirations = fetch_expirations(token, symbol)
    if not expirations:
        st.error(
            "No options expirations returned for this ticker. "
            "This may be an options entitlement issue on Tradier, or the symbol has no listed options."
        )
        st.stop()

    # Build expiration->DTE table
    today_et = datetime.now(tz=tz.gettz("America/New_York")).date()
    exp_meta = []
    for exp in expirations:
        d = _safe_parse_expiration(exp)
        if d is None:
            continue
        dte = (d - today_et).days
        if dte < 0:
            continue
        exp_meta.append((str(exp), int(dte)))

    exp_df = pd.DataFrame(exp_meta, columns=["expiration", "dte"]).sort_values(["dte", "expiration"]).reset_index(drop=True)
    if exp_df.empty:
        st.error("No non-expired expirations available.")
        st.stop()

    # Determine default expiration index:
    # - Keep user's DTE across ticker changes by choosing closest expiration by DTE
    # - Still show date-based selection only
    ticker_changed = symbol != st.session_state.prev_symbol
    if ticker_changed:
        st.session_state.prev_symbol = symbol
        # keep selected_dte as-is, choose closest expiration
        default_exp = _closest_expiration_by_dte(exp_df, st.session_state.selected_dte)
    else:
        # if we already have an expiration selected, keep it if valid; otherwise choose closest by DTE
        prev_exp = st.session_state.get("expiration_select", "")
        if prev_exp and (prev_exp in exp_df["expiration"].tolist()):
            default_exp = prev_exp
        else:
            default_exp = _closest_expiration_by_dte(exp_df, st.session_state.selected_dte)

    expiration_options = exp_df["expiration"].tolist()
    default_index = expiration_options.index(default_exp) if default_exp in expiration_options else 0

    expiration = st.selectbox(
        "Expiration (date)",
        options=expiration_options,
        index=default_index,
        key="expiration_select",
        help="Date-based expiration selector. Your last DTE preference is preserved when you change tickers.",
    )

    # Update persisted DTE whenever user picks a new date
    try:
        st.session_state.selected_dte = int(exp_df.loc[exp_df["expiration"] == expiration, "dte"].iloc[0])
    except Exception:
        pass

    st.subheader("GEX / VEX Convention")
    calls_pos = st.toggle("Calls + / Puts - (common)", value=True)

    st.subheader("Strike Window")
    n_each_side = st.number_input(
        "Strikes each side of spot",
        min_value=1,
        value=25,
        step=1,
        help="Used for GEX/VEX strike charts AND as the 'near-the-money' filter for Vanna curves.",
        key="n_each_side",
    )
    st.caption(f"Window: {2 * int(n_each_side) + 1} strikes total (nearest to spot).")

    st.subheader("Vanna settings")
    vanna_max_dte = st.number_input(
        "Max DTE for Vanna curves",
        min_value=0,
        value=190,
        step=1,
        help="Only expirations with DTE <= this value are included in Vanna curves.",
        key="vanna_max_dte",
    )

    st.subheader("Refresh")
    refresh = st.button("Refresh now", key="refresh_now")

    live = st.toggle("Live updates", value=False, key="live_updates")
    interval_s = st.number_input(
        "Refresh interval (seconds)",
        min_value=5,
        value=10,
        step=1,
        disabled=not live,
        key="refresh_interval_s",
    )

# Live autorefresh (keeps user’s Plotly pan/zoom via uirevision below)
if live:
    st_autorefresh(interval=int(interval_s * 1000), key="auto_refresh_key")

if refresh:
    st.cache_data.clear()

# ---------------------------
# Fetch + compute (selected expiration)
# ---------------------------
spot, chain, quote = fetch_snapshot(token, symbol, expiration)
if np.isnan(spot):
    st.error("Could not determine spot price from quote.")
    st.stop()

cfg_gex = GexConfig(calls_positive_puts_negative=calls_pos)
cfg_vex = VexConfig(calls_positive_puts_negative=calls_pos)
cfg_vanna = VannaConfig(calls_positive_puts_negative=calls_pos)

df_raw = options_to_df(chain)

# GEX
df_gex_all = compute_gex(df_raw, spot=spot, cfg=cfg_gex)

# VEX
df_vex_all = compute_vex(df_raw, cfg=cfg_vex)

# Validate
if df_gex_all.empty or "strike" not in df_gex_all.columns:
    st.error("No options data returned for this expiration.")
    st.stop()

# Strike window around spot (N each side) - use strikes from GEX df
all_strikes = df_gex_all["strike"].dropna().to_numpy(dtype=float)
strike_lo, strike_hi = strikes_window_around_spot(all_strikes, spot=float(spot), n_each_side=int(n_each_side))

df_gex = df_gex_all[(df_gex_all["strike"] >= strike_lo) & (df_gex_all["strike"] <= strike_hi)].copy()
df_vex = df_vex_all[(df_vex_all["strike"] >= strike_lo) & (df_vex_all["strike"] <= strike_hi)].copy()

# Aggregate
df_strike = gex_by_strike(df_gex)
df_vstrike = vex_by_strike(df_vex)

# Ensure net_gex exists (and is numeric)
if "net_gex" not in df_strike.columns:
    tmp = df_gex.copy()
    tmp["gex"] = tmp.get("gex", 0.0)
    df_strike = df_strike.merge(
        tmp.groupby("strike", as_index=False)["gex"].sum().rename(columns={"gex": "net_gex"}),
        on="strike",
        how="left",
    )
df_strike["net_gex"] = pd.to_numeric(df_strike["net_gex"], errors="coerce").fillna(0.0)

call_wall, put_wall = call_put_walls(df_strike)
zero_gamma = find_zero_gamma(df_strike, spot=spot)
net_gex = float(df_strike["net_gex"].sum()) if not df_strike.empty else 0.0

# Max Pain (use full chain df_raw by default) - still used for GEX metrics + GEX chart
max_pain = calculate_max_pain(df_raw)

# Current selected DTE for display
selected_dte = int(st.session_state.get("selected_dte", 0))

# ---------------------------
# Header metrics
# ---------------------------
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Selected DTE", f"{selected_dte}")
c3.metric("Net GEX", fmt_compact(net_gex))
c4.metric("Call Wall", f"{call_wall:,.0f}" if call_wall is not None else "—")
c5.metric("Put Wall", f"{put_wall:,.0f}" if put_wall is not None else "—")
c6.metric("Zero Gamma", f"{zero_gamma:,.2f}" if zero_gamma is not None else "—")
c7.metric("Max Pain", f"{max_pain:,.0f}" if max_pain is not None else "—")

st.divider()

# ---------------------------
# GEX Chart
# ---------------------------
fig = go.Figure()

if not df_strike.empty:
    bar_colors = np.where(df_strike["net_gex"].to_numpy() >= 0, "green", "red")

    fig.add_trace(
        go.Bar(
            x=df_strike["net_gex"],
            y=df_strike["strike"],
            orientation="h",
            name="Net GEX",
            marker=dict(color=bar_colors),
        )
    )

    max_pos = float(df_strike["net_gex"].max())
    min_neg = float(df_strike["net_gex"].min())
    max_abs_value = max(abs(min_neg), abs(max_pos)) or 1.0
    chart_range = max_abs_value * 1.3

    strike_min = float(df_strike["strike"].min())
    strike_max = float(df_strike["strike"].max())
    ypad = (strike_max - strike_min) * 0.02 if strike_max > strike_min else 1.0

    fig.add_hline(y=spot, line_width=2, line_dash="dash", line_color="blue", annotation_text="Spot")

    if call_wall is not None:
        fig.add_hline(y=call_wall, line_width=2, line_dash="dot", annotation_text="Call Wall")
    if put_wall is not None:
        fig.add_hline(y=put_wall, line_width=2, line_dash="dot", annotation_text="Put Wall")
    if zero_gamma is not None:
        fig.add_hline(y=zero_gamma, line_width=2, line_dash="dashdot", annotation_text="Zero Γ")
    if max_pain is not None:
        fig.add_hline(y=max_pain, line_width=2, line_dash="solid", line_color="purple", annotation_text="Max Pain")

    fig.update_layout(
        title=f"{symbol} Net GEX by Strike ({expiration})",
        xaxis_title="Net GEX ($Gamma per 1% move)",
        yaxis_title="Strike",
        height=720,
        xaxis=dict(range=[-chart_range, chart_range], zeroline=True, zerolinewidth=2),
        uirevision=f"{symbol}-{expiration}-{int(calls_pos)}-gex",
    )
    fig.update_yaxes(range=[strike_min - ypad, strike_max + ypad])

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# VEX Chart
# ---------------------------
fig_vex = go.Figure()

if not df_vstrike.empty:
    fig_vex.add_trace(
        go.Bar(
            x=df_vstrike["call_vex"],
            y=df_vstrike["strike"],
            orientation="h",
            name="Call VEX",
        )
    )
    fig_vex.add_trace(
        go.Bar(
            x=-df_vstrike["put_vex"].abs(),
            y=df_vstrike["strike"],
            orientation="h",
            name="Put VEX",
        )
    )

    max_call = float(df_vstrike["call_vex"].abs().max())
    max_put = float(df_vstrike["put_vex"].abs().max())
    max_abs_v = max(max_call, max_put) or 1.0
    vex_range = max_abs_v * 1.3

    strike_min_v = float(df_vstrike["strike"].min())
    strike_max_v = float(df_vstrike["strike"].max())
    ypad_v = (strike_max_v - strike_min_v) * 0.02 if strike_max_v > strike_min_v else 1.0

    fig_vex.add_hline(y=spot, line_width=2, line_dash="dash", line_color="blue", annotation_text="Spot")

    fig_vex.update_layout(
        barmode="overlay",
        title=f"{symbol} Vega Exposure by Strike (Calls vs Puts) ({expiration})",
        xaxis_title="VEX",
        yaxis_title="Strike",
        height=720,
        xaxis=dict(range=[-vex_range, vex_range], zeroline=True, zerolinewidth=2),
        uirevision=f"{symbol}-{expiration}-{int(calls_pos)}-vex",
    )
    fig_vex.update_yaxes(range=[strike_min_v - ypad_v, strike_max_v + ypad_v])

st.plotly_chart(fig_vex, use_container_width=True)

# ---------------------------
# VANNA Curves
# ---------------------------
st.subheader("Vanna Curves: Net Vega Hedging Exposure vs Implied Volatility")

IV_BINS = 18

# restrict expirations used for vanna curves
vanna_exp_df = exp_df[exp_df["dte"].astype(int) <= int(vanna_max_dte)].copy()

curves = []
for exp, dte_val in vanna_exp_df[["expiration", "dte"]].itertuples(index=False, name=None):
    chain_i = fetch_chain(token, symbol, exp)
    df_i = options_to_df(chain_i)
    if df_i.empty or "strike" not in df_i.columns:
        continue

    df_vex_i = compute_vex(df_i, cfg=cfg_vex)

    # Near-the-money filter using strike window around spot (same N each side setting)
    strikes_i = df_vex_i["strike"].dropna().to_numpy(dtype=float) if "strike" in df_vex_i.columns else np.array([])
    lo_i, hi_i = strikes_window_around_spot(strikes_i, spot=float(spot), n_each_side=int(n_each_side))
    df_vex_i = df_vex_i[(df_vex_i["strike"] >= lo_i) & (df_vex_i["strike"] <= hi_i)].copy()

    curve_df = vanna_curve_for_expiration(
        df_vex=df_vex_i,
        dte=int(dte_val),
        expiration=str(exp),
        cfg=cfg_vanna,
        n_iv_bins=IV_BINS,
    )
    if curve_df is not None and not curve_df.empty:
        curves.append(curve_df)

df_vanna_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

fig_vanna = go.Figure()

if not df_vanna_curves.empty:
    # One trace per DTE (different color per curve)
    for dte_val, g in df_vanna_curves.groupby("dte", sort=True):
        g = g.sort_values("iv")
        fig_vanna.add_trace(
            go.Scatter(
                x=g["iv"],
                y=g["net_vega_hedge_exposure"],
                mode="lines+markers",
                name=f"{int(dte_val)} DTE",
                text=g.apply(lambda r: f"Exp: {r['expiration']}<br>DTE: {r['dte']}<br>IV bin: {r['iv_bin']}", axis=1),
                hovertemplate=(
                    "IV: %{x:.4f}<br>"
                    "Net Vega Hedge: %{y:,.0f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    fig_vanna.update_layout(
        title=f"{symbol} Vanna Curves — Near-ATM only (Max DTE: {int(vanna_max_dte)})",
        xaxis_title="Implied Volatility (binned; netted across near-ATM contracts within each bin)",
        yaxis_title="Net Vega Hedging Exposure (netted across near-ATM contracts in each IV bin)",
        height=620,
        legend_title="Curve DTE",
        uirevision=f"{symbol}-{int(calls_pos)}-vanna-curves",
    )
else:
    fig_vanna.update_layout(
        title="No vanna curves available (missing IV / vega / OI or no chains).",
        height=620,
        uirevision=f"{symbol}-{int(calls_pos)}-vanna-curves",
    )

st.plotly_chart(fig_vanna, use_container_width=True)

st.divider()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab4, tab5, tab3 = st.tabs(
    ["By strike (GEX)", "Contracts (filtered)", "By strike (VEX)", "Vanna curves (binned)", "Quote"]
)

with tab1:
    st.dataframe(df_strike.reset_index(drop=True), use_container_width=True, height=520)

with tab2:
    cols = [
        c
        for c in ["symbol", "option_type", "strike", "open_interest", "volume", "greeks.gamma", "gex", "greeks.vega", "vex"]
        if c in df_gex.columns
    ]
    merged = df_gex.copy()
    if "vex" not in merged.columns and "vex" in df_vex.columns:
        merged = merged.merge(
            df_vex[["symbol", "option_type", "strike", "vex"]],
            on=["symbol", "option_type", "strike"],
            how="left",
        )
        cols = [c for c in cols if c != "vex"] + (["vex"] if "vex" in merged.columns else [])

    st.dataframe(
        merged[cols].sort_values(["strike", "option_type"]).reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

with tab3:
    st.json(quote)

with tab4:
    st.dataframe(df_vstrike.reset_index(drop=True), use_container_width=True, height=520)

with tab5:
    st.dataframe(df_vanna_curves.reset_index(drop=True), use_container_width=True, height=520)

eastern = tz.gettz("America/New_York")
st.caption(f"Last updated: {datetime.now(tz=eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}")
