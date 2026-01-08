import os
from datetime import datetime
from dateutil import tz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from tradier_client import TradierClient
from gex_calc import (
    GexConfig,
    compute_gex,
    options_to_df,
    gex_by_strike,
    call_put_walls,
    find_zero_gamma,
    calculate_max_pain,
    # volume_triggers,
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


@st.cache_data(ttl=5, show_spinner=False)
def fetch_snapshot(token: str, symbol: str, expiration: str):
    c = TradierClient(token=token)
    q = c.get_quote(symbol)
    spot = float(q.get("last") or q.get("bid") or q.get("ask") or np.nan)
    chain = c.get_chain(symbol=symbol, expiration=expiration, greeks=True)
    return spot, chain, q


@st.cache_data(ttl=60, show_spinner=False)
def fetch_expirations(token: str, symbol: str):
    c = TradierClient(token=token)
    return c.get_expirations(symbol)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    token = get_token()
    if not token:
        st.error("Missing Tradier token. Set TRADIER_TOKEN env var or .streamlit/secrets.toml.")
        st.stop()

    symbol = st.text_input("Ticker", value="SPY").upper().strip()

    expirations = fetch_expirations(token, symbol)
    if not expirations:
        st.error("No expirations returned. Check symbol or Tradier entitlements.")
        st.stop()

    expiration = st.selectbox("Expiration", options=expirations, index=0)

    st.subheader("GEX Convention")
    calls_pos = st.toggle("Calls + / Puts - (common)", value=True)

    st.subheader("Strike Window")
    n_each_side = st.number_input(
        "Strikes each side of spot",
        min_value=1,
        value=25,
        step=1,
        help="Shows N strikes below + N strikes above the nearest strike to spot.",
        key="n_each_side",
    )
    st.caption(f"Window: {2 * int(n_each_side) + 1} strikes total (nearest to spot).")

    st.subheader("Refresh")
    refresh = st.button("Refresh now", key="refresh_now")

    live = st.toggle("Live updates", value=False, key="live_updates")
    interval_s = st.number_input(
        "Refresh interval (seconds)",
        min_value=1,
        value=5,
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
# Fetch + compute
# ---------------------------
spot, chain, quote = fetch_snapshot(token, symbol, expiration)
if np.isnan(spot):
    st.error("Could not determine spot price from quote.")
    st.stop()

cfg = GexConfig(calls_positive_puts_negative=calls_pos)

df_raw = options_to_df(chain)
df_gex_all = compute_gex(df_raw, spot=spot, cfg=cfg)

# Validate
if df_gex_all.empty or "strike" not in df_gex_all.columns:
    st.error("No options data returned for this expiration.")
    st.stop()

# Strike window around spot (N each side)
all_strikes = df_gex_all["strike"].dropna().to_numpy(dtype=float)
strike_lo, strike_hi = strikes_window_around_spot(all_strikes, spot=float(spot), n_each_side=int(n_each_side))
df_gex = df_gex_all[(df_gex_all["strike"] >= strike_lo) & (df_gex_all["strike"] <= strike_hi)].copy()

# Aggregate
df_strike = gex_by_strike(df_gex)

# Ensure net_gex exists (and is numeric)
if "net_gex" not in df_strike.columns:
    # Fallback: compute from contract-level gex
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

# Max Pain (use full chain df_raw by default)
max_pain = calculate_max_pain(df_raw)

# ---------------------------
# Header metrics
# ---------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Net GEX", f"{net_gex:,.0f}")
c3.metric("Call Wall", f"{call_wall:,.0f}" if call_wall is not None else "—")
c4.metric("Put Wall", f"{put_wall:,.0f}" if put_wall is not None else "—")
c5.metric("Zero Gamma", f"{zero_gamma:,.2f}" if zero_gamma is not None else "—")
c6.metric("Max Pain", f"{max_pain:,.0f}" if max_pain is not None else "—")

st.divider()

# ---------------------------
# Chart
# ---------------------------
fig = go.Figure()

if not df_strike.empty:
    # Color net bars by sign: positive = green, negative = red
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

    # Spot line (blue)
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
        # Keep user zoom/pan on reruns (refresh/live)
        uirevision=f"{symbol}-{expiration}-{int(calls_pos)}",
    )
    fig.update_yaxes(range=[strike_min - ypad, strike_max + ypad])

st.plotly_chart(fig, use_container_width=True)

# # ---------------------------
# # Volume Triggers
# # ---------------------------
# st.subheader("Volume Triggers")
# vt = volume_triggers(df_strike, top_n=12)
# st.dataframe(
#     vt[["strike", "total_vol", "total_oi", "vol_oi", "net_gex"]].reset_index(drop=True),
#     use_container_width=True,
#     height=320,
# )

st.divider()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["By strike", "Contracts (filtered)", "Quote"])

with tab1:
    st.dataframe(df_strike.reset_index(drop=True), use_container_width=True, height=520)

with tab2:
    cols = [
        c
        for c in ["symbol", "option_type", "strike", "open_interest", "volume", "greeks.gamma", "gex"]
        if c in df_gex.columns
    ]
    st.dataframe(
        df_gex[cols].sort_values(["strike", "option_type"]).reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

with tab3:
    st.json(quote)

eastern = tz.gettz("America/New_York")
st.caption(f"Last updated: {datetime.now(tz=eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}")
