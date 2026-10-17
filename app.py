import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dotenv import load_dotenv

load_dotenv()

from core.agent import IndianStockRAGAgent
from core.nse_client import NSEClient
from core.technicals import TechnicalAnalyzer
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore

# Page configuration
st.set_page_config(
    page_title="Artha Terminal | Indian Equities Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Institutional Fintech CSS styling (Kite / TradingView Dark Mode Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    code, pre, .number-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background & Container */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }

    /* Pulsing Live Dot */
    .live-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 6px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
    }
    @keyframes pulse {
        to {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
    }

    /* Ticker Banner */
    .ticker-bar {
        background: linear-gradient(90deg, #121824 0%, #171f30 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Metric Cards */
    .terminal-card {
        background: #131926;
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .terminal-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Custom Button Style */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #171f30;
        color: #f1f5f9;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #2563eb;
        color: #ffffff;
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    /* Chat UI Customization */
    .stChatMessage {
        background-color: #111724;
        border: 1px solid #1e283d;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent" not in st.session_state or not hasattr(st.session_state.agent.nse, "get_live_stock_news"):
    st.session_state.agent = IndianStockRAGAgent()
if "portfolio" not in st.session_state:
    st.session_state.portfolio = [
        {"symbol": "RELIANCE", "qty": 10, "buy_price": 1220.0, "sector": "Energy"},
        {"symbol": "TCS", "qty": 5, "buy_price": 2400.0, "sector": "IT"},
        {"symbol": "TMPV", "qty": 20, "buy_price": 285.0, "sector": "Auto"},
        {"symbol": "HAL", "qty": 8, "buy_price": 4300.0, "sector": "Defense"}
    ]
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "### ⚡ Artha Quantitative Intelligence Copilot\n\n"
                       "Welcome to **Artha Terminal**. I am an autonomous quant research copilot connected to live **NSE India** order feeds, technical indicator matrices, valuation multiples, real-time financial news, and persistent SQLite memory.\n\n"
                       "**Quick Actions:**\n"
                       "- Ask about any stock (e.g. *Reliance, TCS, HAL, Zomato, Trent, IRFC*)\n"
                       "- Compare metrics: *'Compare TCS and Infosys'*\n"
                       "- Conversational context: *'How does its RSI compare to Tata Motors?'*\n"
                       "- Valuation & News: *'What are the fundamentals and latest news on Zomato?'*"
        }
    ]
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "RELIANCE"

agent: IndianStockRAGAgent = st.session_state.agent
memory: DualTierMemoryManager = agent.memory
nse: NSEClient = agent.nse

# Auto-migrate any legacy portfolio symbols
for p in st.session_state.portfolio:
    p["symbol"] = nse.resolve_symbol(p["symbol"])

# --- TOP LIVE TICKER RIBBON ---
indices = nse.get_major_indices()
status = nse.get_market_status()

# Visual Top Bar
cols = st.columns([1.6] + [1] * len(indices))
with cols[0]:
    st.markdown(
        f"<div style='padding-top:4px; font-weight:700; font-size:1.05rem;'>"
        f"<span class='live-pulse'></span>NSE INDIA <span style='font-size:0.75rem; color:#94a3b8; font-weight:500;'>[{status.get('status', 'Open')}]</span>"
        f"</div>",
        unsafe_allow_html=True
    )

for i, (idx_name, d) in enumerate(indices.items()):
    with cols[i + 1]:
        pchg = d.get("percent_change", 0)
        badge = "badge-green" if pchg >= 0 else "badge-red"
        sign = "+" if pchg >= 0 else ""
        st.markdown(
            f"<div style='font-size:0.85rem; font-weight:600;'>{idx_name}</div>"
            f"<div class='number-font' style='font-size:0.95rem; font-weight:700;'>{d.get('value', 0):,.2f} "
            f"<span class='badge-pill {badge}'>{sign}{pchg:.2f}%</span></div>",
            unsafe_allow_html=True
        )

st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin-top:12px; margin-bottom:18px;'>", unsafe_allow_html=True)

# --- SIDEBAR: TERMINAL CONTROLS & MEMORY RADAR ---
with st.sidebar:
    st.markdown("## ⚡ ARTHA TERMINAL")
    st.caption("Institutional AI Copilot for Indian Equities")

    # API Key Configuration
    with st.expander("🔑 AI Engine Configuration", expanded=False):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=agent.api_key or "",
            placeholder="AI Studio Key (Free)"
        )
        if st.button("Apply Key", width="stretch"):
            st.session_state.agent = IndianStockRAGAgent(api_key=api_key_input)
            st.success("API Key updated!")
            st.rerun()

    st.markdown("### 🧠 Dual-Tier Memory")
    
    # Tier 1: Working Context
    with st.container():
        st.markdown(f"**Active Ticker**: `{memory.last_mentioned_symbol or 'None'}`")
        if memory.recent_symbols:
            st.caption(f"Context Thread: {', '.join(memory.recent_symbols)}")

    # Tier 2: Persistent SQLite Watchlist
    with st.expander("📋 SQLite Watchlist", expanded=True):
        watchlist = memory.get_watchlist()
        if watchlist:
            for item in watchlist:
                wc1, wc2 = st.columns([3, 1])
                with wc1:
                    st.markdown(f"**{item['symbol']}**")
                    if item.get("notes"):
                        st.caption(f"📌 {item['notes']}")
                with wc2:
                    if st.button("×", key=f"del_side_{item['symbol']}"):
                        memory.remove_from_watchlist(item['symbol'])
                        st.rerun()
        else:
            st.caption("Watchlist empty. Tell copilot: *'Add TCS to my watchlist'*")

    # Quick Add to Watchlist
    with st.expander("➕ Pin Stock to Watchlist"):
        add_sym_in = st.text_input("Symbol (e.g. HAL, IRFC, TRENT)")
        add_note_in = st.text_input("Thesis / Notes")
        if st.button("Save to Persistent Memory", width="stretch"):
            if add_sym_in:
                res_sym = nse.resolve_symbol(add_sym_in)
                memory.add_to_watchlist(res_sym, notes=add_note_in)
                st.success(f"Pinned {res_sym} to SQLite!")
                st.rerun()

# --- MAIN NAVIGATION TABS ---
tab_chat, tab_chart, tab_portfolio, tab_radar = st.tabs([
    "💬 Research Copilot",
    "📈 Interactive Charting & Radar",
    "💼 Portfolio & P&L Analytics",
    "📰 Live Intelligence & Disclosures"
])

# ================= TAB 1: RESEARCH COPILOT =================
with tab_chat:
    st.markdown("##### ⚡ Command Shortcuts:")
    sc1, sc2, sc3, sc4 = st.columns(4)
    if sc1.button("📊 Reliance Live & Pivots"):
        st.session_state.pending_prompt = "What is the live price, RSI, and pivot levels for Reliance?"
    if sc2.button("⚖️ Compare TCS vs INFY"):
        st.session_state.pending_prompt = "Compare TCS and Infosys in terms of valuation, P/E, and dividend yield."
    if sc3.button("🚀 HAL Defense Order Book"):
        st.session_state.pending_prompt = "Explain HAL defense order book, Tejas fighter aircraft, and outlook."
    if sc4.button("📰 Zomato News & Blinkit"):
        st.session_state.pending_prompt = "What is the latest news and growth drivers for Zomato?"

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # Chat Feed
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "traces" in msg and msg["traces"]:
                with st.expander("🛠️ Inspect Autonomous Tool Execution Traces", expanded=False):
                    for t in msg["traces"]:
                        st.markdown(f"**Executed Tool**: `{t.get('tool')}`")
                        st.json({"arguments": t.get("args"), "data_payload": t.get("output")})

    # User Input
    user_query = st.chat_input("Enter command or question (e.g. 'Analyze Tata Motors technicals and stop-loss')...")
    if getattr(st.session_state, "pending_prompt", None):
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Executing real-time tools & synthesizing intelligence..."):
                resp, traces = agent.run(user_query)
                st.markdown(resp)
                if traces:
                    with st.expander("🛠️ Inspect Autonomous Tool Execution Traces", expanded=False):
                        for t in traces:
                            st.markdown(f"**Executed Tool**: `{t.get('tool')}`")
                            st.json({"arguments": t.get("args"), "data_payload": t.get("output")})

        st.session_state.messages.append({
            "role": "assistant",
            "content": resp,
            "traces": traces
        })

# ================= TAB 2: INTERACTIVE CHART & GAUGES =================
with tab_chart:
    st.subheader("📈 Institutional Technical Radar")
    
    col_sym, col_tf, col_action = st.columns([3, 2, 2])
    with col_sym:
        target_symbol = st.text_input(
            "Ticker Symbol",
            value=memory.last_mentioned_symbol or st.session_state.selected_symbol,
            key="tab_chart_ticker_input"
        )
    with col_tf:
        timeframe = st.selectbox("Candle History", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    with col_action:
        st.write("")
        st.write("")
        pin_btn = st.button("📌 Pin to Watchlist", width="stretch")

    resolved_ticker = nse.resolve_symbol(target_symbol)
    st.session_state.selected_symbol = resolved_ticker

    if pin_btn:
        memory.add_to_watchlist(resolved_ticker)
        st.success(f"Added {resolved_ticker} to watchlist!")

    quote = nse.get_live_quote(resolved_ticker)
    df = nse.get_historical_ohlcv(resolved_ticker, period=timeframe, interval="1d")

    if df is not None and not df.empty and len(df) > 12:
        # Technical indicators calculation
        indicators = TechnicalAnalyzer.calculate_indicators(df)

        # Top Metric Stat-Grid
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        ltp = quote.get("last_price", df["Close"].iloc[-1])
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)

        m_col1.metric("Live LTP", f"₹{ltp:,.2f}", f"{chg:+.2f} ({pchg:+.2f}%)")
        m_col2.metric("200 EMA Benchmark", f"₹{indicators.get('ema_200', 'N/A')}")
        m_col3.metric("RSI (14)", f"{indicators.get('rsi_14')}", indicators.get('rsi_condition'))
        m_col4.metric("Pivot (P)", f"₹{indicators.get('pivot_point')}")
        m_col5.metric("Support (S1)", f"₹{indicators.get('support_1')}")
        m_col6.metric("Resistance (R1)", f"₹{indicators.get('resistance_1')}")

        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        # Split layout: 70% Candlestick + Volume, 30% RSI Speedometer Gauge + ATR Setup
        chart_col, gauge_col = st.columns([3, 1])

        with chart_col:
            # Candlestick + Volume Subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.78, 0.22],
                subplot_titles=[f"{quote.get('company_name', resolved_ticker)} ({resolved_ticker})", "Volume"]
            )

            # Candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#ef4444"
                ),
                row=1, col=1
            )

            # EMAs
            ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
            ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
            ema_200 = df["Close"].ewm(span=200, adjust=False).mean() if len(df) >= 40 else None

            fig.add_trace(go.Scatter(x=df["Date"], y=ema_20, line=dict(color="#38bdf8", width=1.5), name="EMA 20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=ema_50, line=dict(color="#fbbf24", width=1.5), name="EMA 50"), row=1, col=1)
            if ema_200 is not None:
                fig.add_trace(go.Scatter(x=df["Date"], y=ema_200, line=dict(color="#c084fc", width=2), name="EMA 200"), row=1, col=1)

            # Volume Bars
            vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(
                go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume"),
                row=2, col=1
            )

            fig.update_layout(
                height=560,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                paper_bgcolor="#0e131d",
                plot_bgcolor="#0e131d",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width="stretch")

        with gauge_col:
            # RSI Speedometer Gauge
            rsi_val = indicators.get("rsi_14", 50)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_val,
                title={"text": "RSI (14) Momentum", "font": {"size": 15, "color": "#cbd5e1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": "#60a5fa", "thickness": 0.25},
                    "bgcolor": "#131926",
                    "borderwidth": 1,
                    "bordercolor": "#1f293d",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.3)"},
                        {"range": [30, 70], "color": "rgba(100, 116, 139, 0.2)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "#f43f5e", "width": 3},
                        "thickness": 0.8,
                        "value": rsi_val
                    }
                }
            ))
            gauge_fig.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="#0e131d",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"}
            )
            st.plotly_chart(gauge_fig, width="stretch")

            # Volatility & ATR Trade Setup Card
            st.markdown(
                f"<div class='terminal-card'>"
                f"<div style='font-size:0.8rem; color:#94a3b8; font-weight:600;'>ATR (14) VOLATILITY SETUP</div>"
                f"<div style='margin-top:6px; font-size:0.9rem;'>Daily ATR Range: <b class='number-font'>₹{indicators.get('atr_14')}</b></div>"
                f"<div style='margin-top:4px; font-size:0.9rem; color:#ef4444;'>Stop-Loss: <b class='number-font'>₹{indicators.get('suggested_stop_loss')}</b></div>"
                f"<div style='margin-top:4px; font-size:0.9rem; color:#10b981;'>Target: <b class='number-font'>₹{indicators.get('suggested_target')}</b></div>"
                f"<div style='margin-top:6px; font-size:0.8rem; color:#60a5fa;'>Stance: <b>{indicators.get('overall_bias')}</b></div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning(f"Could not load chart data for {resolved_ticker}. Please verify the symbol.")

# ================= TAB 3: PORTFOLIO & P&L ANALYTICS =================
with tab_portfolio:
    st.subheader("💼 Real-Time Portfolio Performance & Risk Distribution")

    # Add holding form
    with st.expander("➕ Add Position to Portfolio"):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            in_sym = st.text_input("NSE Symbol (e.g. INFY, HAL, BEL)")
        with p2:
            in_qty = st.number_input("Quantity", min_value=1, value=10, step=1)
        with p3:
            in_buy = st.number_input("Average Buy Price (₹)", min_value=0.5, value=1000.0, step=10.0)
        with p4:
            in_sector = st.selectbox("Sector", ["IT", "Banking", "Auto", "Defense", "Energy", "FMCG", "Railways", "Other"])

        if st.button("Log Position", width="stretch"):
            if in_sym:
                res_sym = nse.resolve_symbol(in_sym)
                st.session_state.portfolio.append({
                    "symbol": res_sym,
                    "qty": in_qty,
                    "buy_price": in_buy,
                    "sector": in_sector
                })
                st.success(f"Added {in_qty} of {res_sym}!")
                st.rerun()

    # Portfolio Analytics Engine
    if st.session_state.portfolio:
        rows = []
        total_inv = 0.0
        total_cur = 0.0
        sector_weights = {}

        for pos in st.session_state.portfolio:
            s = pos["symbol"]
            q = pos["qty"]
            bp = pos["buy_price"]
            sec = pos.get("sector", "Other")

            live_q = nse.get_live_quote(s)
            cur_p = live_q.get("last_price", bp)

            inv_val = q * bp
            cur_val = q * cur_p
            pnl_val = cur_val - inv_val
            pnl_pct = (pnl_val / inv_val * 100) if inv_val else 0.0

            total_inv += inv_val
            total_cur += cur_val
            sector_weights[sec] = sector_weights.get(sec, 0.0) + cur_val

            rows.append({
                "Symbol": s,
                "Sector": sec,
                "Qty": q,
                "Avg Buy": f"₹{bp:,.2f}",
                "Live LTP": f"₹{cur_p:,.2f}",
                "Invested": f"₹{inv_val:,.2f}",
                "Current": f"₹{cur_val:,.2f}",
                "P&L (₹)": f"{'+' if pnl_val >= 0 else ''}₹{pnl_val:,.2f}",
                "Return (%)": f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            })

        net_pnl = total_cur - total_inv
        net_pct = (net_pnl / total_inv * 100) if total_inv else 0.0

        # High-impact summary cards
        c_inv, c_cur, c_pnl = st.columns(3)
        c_inv.metric("Capital Invested", f"₹{total_inv:,.2f}")
        c_cur.metric("Current Portfolio Value", f"₹{total_cur:,.2f}")
        c_pnl.metric("Total Unrealized P&L", f"{'+' if net_pnl >= 0 else ''}₹{net_pnl:,.2f}", f"{'+' if net_pct >= 0 else ''}{net_pct:.2f}%")

        p_chart_col, p_table_col = st.columns([1, 2])

        with p_chart_col:
            # Sector Allocation Donut Chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=list(sector_weights.keys()),
                values=list(sector_weights.values()),
                hole=0.6,
                marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]),
                textinfo="label+percent"
            )])
            donut_fig.update_layout(
                title={"text": "Sector Weighting Distribution", "font": {"size": 14, "color": "#cbd5e1"}},
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#0e131d",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"},
                showlegend=False
            )
            st.plotly_chart(donut_fig, width="stretch")

        with p_table_col:
            st.markdown("##### 📋 Holdings Breakdown")
            st.dataframe(pd.DataFrame(rows), width="stretch")

        if st.button("🗑️ Reset Portfolio Holdings"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Portfolio is empty. Log a position above to track real-time P&L.")

# ================= TAB 4: LIVE INTELLIGENCE & NEWS =================
with tab_radar:
    st.subheader("📰 Market Intelligence & Regulatory Disclosures")

    n_col1, n_col2 = st.columns([1, 1])

    with n_col1:
        st.markdown("#### 🔴 Live Stock Media Stream")
        target_news_stock = st.text_input("Stream News for Stock", value="RELIANCE", key="news_tab_input")
        if target_news_stock:
            news_items = nse.get_live_stock_news(target_news_stock, limit=5)
            for item in news_items:
                st.markdown(
                    f"<div class='terminal-card'>"
                    f"<div style='font-weight:600; font-size:0.95rem;'><a href='{item['link']}' target='_blank' style='color:#60a5fa; text-decoration:none;'>{item['title']}</a></div>"
                    f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:4px;'>Source: <b>{item['source']}</b> • <i>{item['published']}</i></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    with n_col2:
        st.markdown("#### 🔍 Corporate Filings & Knowledge RAG")
        k_search = st.text_input("Semantic Retrieval Query", placeholder="e.g. defense capex, Blinkit dark stores, green hydrogen...")
        if k_search:
            results = agent.knowledge.search(k_search)
            for r in results:
                st.markdown(
                    f"<div class='terminal-card'>"
                    f"<div style='font-size:0.8rem;'><span class='badge-pill badge-blue'>{r.get('symbol')}</span> <span style='color:#94a3b8;'>{r.get('category')}</span></div>"
                    f"<div style='font-weight:700; margin-top:6px;'>{r.get('title')}</div>"
                    f"<div style='font-size:0.85rem; color:#cbd5e1; margin-top:4px;'>{r.get('content')}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Enter a query above to semantically search across pre-seeded corporate filings.")
