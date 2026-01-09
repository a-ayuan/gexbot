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
    compute_vanna,
    vanna_by_strike,
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

    spot = _extract_spot_from_quote(q)
    if np.isfinite(spot) and spot > 0:
        return True

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

def _safe_parse_expiration(expiration: str) -> date | None:
    try:
        return datetime.strptime(str(expiration), "%Y-%m-%d").date()
    except Exception:
        return None

def _closest_expiration_by_dte(exp_df: pd.DataFrame, target_dte: int) -> str:
    if exp_df.empty:
        return ""
    target_dte = int(target_dte)
    i = (exp_df["dte"].astype(int) - target_dte).abs().idxmin()
    return str(exp_df.loc[i, "expiration"])

def _tte_years(expiration: str) -> float:
    """
    Time-to-expiration in years, computed in America/New_York timezone.
    Uses end-of-day assumption for expiration (16:00 ET) to avoid 0 on same-day.
    """
    eastern = tz.gettz("America/New_York")
    now = datetime.now(tz=eastern)

    try:
        d = datetime.strptime(str(expiration), "%Y-%m-%d").date()
    except Exception:
        return 0.0

    exp_dt = datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=eastern)
    t = (exp_dt - now).total_seconds() / (365.0 * 24.0 * 3600.0)
    return float(max(t, 0.0))

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    token = get_token()
    if not token:
        st.error("Missing Tradier token. Set TRADIER_TOKEN env var or .streamlit/secrets.toml.")
        st.stop()

    if "prev_symbol" not in st.session_state:
        st.session_state.prev_symbol = "SPY"
    if "selected_dte" not in st.session_state:
        st.session_state.selected_dte = 0

    symbol = st.text_input("Ticker", value=st.session_state.prev_symbol).upper().strip()

    q_check = fetch_quote(token, symbol)
    if not _is_valid_quote_for_symbol(q_check, symbol):
        st.warning(f"Invalid ticker: **{symbol}**. Please enter a valid symbol (e.g., SPY, QQQ, AAPL).")
        st.stop()

    expirations = fetch_expirations(token, symbol)
    if not expirations:
        st.error(
            "No options expirations returned for this ticker. "
            "This may be an options entitlement issue on Tradier, or the symbol has no listed options."
        )
        st.stop()

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

    ticker_changed = symbol != st.session_state.prev_symbol
    if ticker_changed:
        st.session_state.prev_symbol = symbol
        default_exp = _closest_expiration_by_dte(exp_df, st.session_state.selected_dte)
    else:
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

    try:
        st.session_state.selected_dte = int(exp_df.loc[exp_df["expiration"] == expiration, "dte"].iloc[0])
    except Exception:
        pass
    
    live = st.toggle("Live updates", value=False, key="live_updates")
    interval_s = st.number_input(
        "Refresh interval (seconds)",
        min_value=5,
        value=10,
        step=1,
        disabled=not live,
        key="refresh_interval_s",
    )

    st.subheader("GEX / VEX Convention")
    calls_pos = st.toggle("Calls + / Puts - (common)", value=True)

    st.subheader("Strike Window")
    n_each_side = st.number_input(
        "Strikes each side of spot",
        min_value=1,
        value=25,
        step=1,
        help="Used for GEX/VEX/Vanna strike charts.",
        key="n_each_side",
    )
    st.caption(f"Window: {2 * int(n_each_side) + 1} strikes total (nearest to spot).")

    st.subheader("Vanna model inputs")
    r_rate = st.number_input(
        "Risk-free rate (annual, decimal)",
        min_value=0.0,
        value=0.00,
        step=0.01,
        help="Used for BS d1 (optional). Leave 0.00 if you don't want to model rates.",
        key="rf_rate",
    )
    q_yield = st.number_input(
        "Dividend yield (annual, decimal)",
        min_value=0.0,
        value=0.00,
        step=0.01,
        help="Used for BS d1 (optional). Leave 0.00 if you don't want to model dividends.",
        key="div_yield",
    )

    st.subheader("Refresh")
    refresh = st.button("Refresh now", key="refresh_now")

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

df_raw = options_to_df(chain)

cfg_gex = GexConfig(calls_positive_puts_negative=calls_pos)
cfg_vex = VexConfig(calls_positive_puts_negative=calls_pos)
cfg_vanna = VannaConfig(calls_positive_puts_negative=calls_pos)

# GEX
df_gex_all = compute_gex(df_raw, spot=spot, cfg=cfg_gex)

# VEX
df_vex_all = compute_vex(df_raw, cfg=cfg_vex)

# VANNA (computed from IV + BS + (optional) provider vega)
T = _tte_years(expiration)
df_vanna_all = compute_vanna(
    df_raw,
    spot=float(spot),
    t_years=float(T),
    r=float(r_rate),
    q=float(q_yield),
    cfg=cfg_vanna,
)

if df_gex_all.empty or "strike" not in df_gex_all.columns:
    st.error("No options data returned for this expiration.")
    st.stop()

all_strikes = df_gex_all["strike"].dropna().to_numpy(dtype=float)
strike_lo, strike_hi = strikes_window_around_spot(all_strikes, spot=float(spot), n_each_side=int(n_each_side))

df_gex = df_gex_all[(df_gex_all["strike"] >= strike_lo) & (df_gex_all["strike"] <= strike_hi)].copy()
df_vex = df_vex_all[(df_vex_all["strike"] >= strike_lo) & (df_vex_all["strike"] <= strike_hi)].copy()
df_vanna = df_vanna_all[(df_vanna_all["strike"] >= strike_lo) & (df_vanna_all["strike"] <= strike_hi)].copy()

df_strike = gex_by_strike(df_gex)
df_vstrike = vex_by_strike(df_vex)
df_vanna_strike = vanna_by_strike(df_vanna)

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

max_pain = calculate_max_pain(df_raw)
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
        title=f"${symbol} Net GEX by Strike ({expiration})",
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
        title=f"${symbol} Vega Exposure by Strike ({expiration})",
        xaxis_title="VEX",
        yaxis_title="Strike",
        height=720,
        xaxis=dict(range=[-vex_range, vex_range], zeroline=True, zerolinewidth=2),
        uirevision=f"{symbol}-{expiration}-{int(calls_pos)}-vex",
    )
    fig_vex.update_yaxes(range=[strike_min_v - ypad_v, strike_max_v + ypad_v])

st.plotly_chart(fig_vex, use_container_width=True)

# ---------------------------
# VANNA Chart
# ---------------------------
vanna_chart_slot = st.empty()

# Toggle shown below the chart (render chart via placeholder after)
vanna_view = st.radio(
    "Display Vanna exposure",
    options=["Both", "Calls", "Puts"],
    horizontal=True,
    key="vanna_view",
)

fig_vanna = go.Figure()

if not df_vanna_strike.empty:
    max_call = float(df_vanna_strike["call_vanna"].abs().max()) if "call_vanna" in df_vanna_strike.columns else 0.0
    max_put = float(df_vanna_strike["put_vanna"].abs().max()) if "put_vanna" in df_vanna_strike.columns else 0.0
    max_abs = max(max_call, max_put) or 1.0
    vanna_range = max_abs * 1.3

    strike_min_v = float(df_vanna_strike["strike"].min())
    strike_max_v = float(df_vanna_strike["strike"].max())
    ypad_v = (strike_max_v - strike_min_v) * 0.02 if strike_max_v > strike_min_v else 1.0

    show_calls = vanna_view in ("Both", "Calls")
    show_puts = vanna_view in ("Both", "Puts")

    fig_vanna.add_trace(
        go.Bar(
            x=df_vanna_strike["call_vanna"],
            y=df_vanna_strike["strike"],
            orientation="h",
            name="Call Vanna",
            marker_color='rgba(200, 0, 0, 0.5)',
            visible=show_calls,
        )
    )
    fig_vanna.add_trace(
        go.Bar(
            x=-df_vanna_strike["put_vanna"].abs(),
            y=df_vanna_strike["strike"],
            orientation="h",
            name="Put Vanna",
            marker_color='rgba(0, 200, 0, 0.5)',
            visible=show_puts,
        )
    )

    fig_vanna.add_hline(y=spot, line_width=2, line_dash="dash", line_color="blue", annotation_text="Spot")

    fig_vanna.update_layout(
        barmode="overlay",
        title=f"${symbol} Vanna Exposure by Strike ({expiration})",
        xaxis_title="Vanna Exposure",
        yaxis_title="Strike",
        height=720,
        xaxis=dict(range=[-vanna_range, vanna_range], zeroline=True, zerolinewidth=2),
        # Keep uirevision stable so Plotly UI state doesn't jump around
        uirevision=f"{symbol}-{expiration}-{int(calls_pos)}-vanna",
    )
    fig_vanna.update_yaxes(range=[strike_min_v - ypad_v, strike_max_v + ypad_v])

# Render chart above toggle
vanna_chart_slot.plotly_chart(fig_vanna, use_container_width=True)

st.divider()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["By strike (GEX)", "Contracts (filtered)", "By strike (VEX)", "By strike (Vanna)", "Quote"]
)

with tab1:
    st.dataframe(df_strike.reset_index(drop=True), use_container_width=True, height=520)

with tab2:
    merged = df_gex.copy()

    if "vex" not in merged.columns and "vex" in df_vex.columns:
        merged = merged.merge(
            df_vex[["symbol", "option_type", "strike", "vex"]],
            on=["symbol", "option_type", "strike"],
            how="left",
        )

    if "vanna_ex" not in merged.columns and "vanna_ex" in df_vanna.columns:
        keep = ["symbol", "option_type", "strike", "vanna_ex", "iv", "d1", "vanna"]
        keep = [c for c in keep if c in df_vanna.columns]
        merged = merged.merge(
            df_vanna[keep],
            on=["symbol", "option_type", "strike"],
            how="left",
        )

    cols_pref = [
        "symbol",
        "option_type",
        "strike",
        "open_interest",
        "volume",
        "greeks.gamma",
        "gex",
        "greeks.vega",
        "vex",
        "iv",
        "d1",
        "vanna",
        "vanna_ex",
    ]
    cols = [c for c in cols_pref if c in merged.columns]

    st.dataframe(
        merged[cols].sort_values(["strike", "option_type"]).reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

with tab3:
    st.dataframe(df_vstrike.reset_index(drop=True), use_container_width=True, height=520)

with tab4:
    st.dataframe(df_vanna_strike.reset_index(drop=True), use_container_width=True, height=520)

with tab5:
    st.json(quote)

eastern = tz.gettz("America/New_York")
st.caption(f"Last updated: {datetime.now(tz=eastern).strftime('%Y-%m-%d %H:%M:%S %Z')}")
