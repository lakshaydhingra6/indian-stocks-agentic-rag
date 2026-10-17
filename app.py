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
    page_title="Artha // Quantum Terminal (NSE)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# ULTRA-PREMIUM NEO-FUTURISTIC FINTECH CSS DESIGN SYSTEM
# =========================================================================
st.markdown("""
<style>
/* Animated gradient background */
@keyframes gradientShift {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
.stApp {
  background: linear-gradient(270deg, #0f172a, #111827, #1e293b, #0f172a);
  background-size: 600% 600%;
  animation: gradientShift 12s ease infinite;
}
/* Global typography */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}
/* Neon glow for glass cards */
.glass-card {
  background: rgba(15, 23, 42, 0.65);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 14px;
  padding: 16px 20px;
  box-shadow: 0 0 10px rgba(59,130,246,0.6), 0 0 20px rgba(168,85,247,0.4);
  transition: all 0.25s cubic-bezier(0.16,1,0.3,1);
}
.glass-card:hover {
  border-color: rgba(96,165,250,0.4);
  transform: translateY(-2px);
  box-shadow: 0 0 15px rgba(59,130,246,0.8);
}
/* HUD card with pulsating neon border */
.hud-card {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(20,27,45,0.7) 0%, rgba(13,18,30,0.7) 100%);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 12px 16px;
}
.hud-card::after {
  content: "";
  position: absolute;
  inset: 0;
  border: 2px solid rgba(59,130,246,0.4);
  box-shadow: 0 0 8px rgba(59,130,246,0.6);
  animation: pulseGlow 3s infinite;
}
@keyframes pulseGlow {
  0% {border-color: rgba(59,130,246,0.2); box-shadow: 0 0 4px rgba(59,130,246,0.2);}
  50% {border-color: rgba(59,130,246,0.6); box-shadow: 0 0 12px rgba(59,130,246,0.6);}
  100% {border-color: rgba(59,130,246,0.2); box-shadow: 0 0 4px rgba(59,130,246,0.2);}
}
/* Neon button hover */
.stButton>button:hover {
  background: linear-gradient(180deg, #34d399, #2563eb);
  box-shadow: 0 0 12px #34d399;
}
/* Live pulse badge */
.pulse-live {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background-color: #10b981;
  box-shadow: 0 0 12px #10b981;
  animation: pulseAnimation 2s infinite;
}
@keyframes pulseAnimation {
  0% {transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16,185,129,0.7);}
  70% {transform: scale(1.15); box-shadow: 0 0 0 10px rgba(16,185,129,0);}
  100% {transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16,185,129,0);}
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
            "content": "### ⚡ Artha Quantum Intelligence Engine\n\n"
                       "Autonomous quant copilot connected to live **NSE India** high-frequency order book feeds, "
                       "technical indicator tensors, valuation ratio matrices, and dual-tier SQLite memory.\n\n"
                       "**Direct Command Examples:**\n"
                       "- 📊 *'What is Reliance live quote, pivot points, and technical trend?'*\n"
                       "- ⚖️ *'Compare TCS and Infosys valuation, P/E, and dividend yield'*\n"
                       "- 🛡️ *'Explain HAL defense order book and Tejas fighter aircraft outlook'*\n"
                       "- 📰 *'Latest news and sentiment for Zomato / Blinkit'*\n"
                       "- 🔄 *'How does its RSI compare to Tata Motors?'* (Contextual Memory)"
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

# =========================================================================
# TOP HERO TELEMETRY RIBBON
# =========================================================================
indices = nse.get_major_indices()
status = nse.get_market_status()

# Hero Header Bar
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; padding: 4px 0;'>
    <div>
        <div class='display-font' style='font-size:1.6rem; font-weight:800; background:linear-gradient(90deg, #60a5fa 0%, #a855f7 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ARTHA // QUANTUM TERMINAL
        </div>
        <div style='font-size:0.78rem; color:#94a3b8; font-weight:500; letter-spacing:0.05em;'>
            AUTONOMOUS AGENTIC RAG - REAL-TIME NSE FEED - DUAL-TIER MEMORY
        </div>
    </div>
    <div style='text-align:right;'>
        <span class='pulse-live'></span>
        <span style='font-weight:700; font-size:0.85rem; color:#10b981;'>LIVE NSE FEED</span>
        <span class='cyber-badge badge-cyan' style='margin-left:8px;'>MEMORY ACTIVE</span>
        <span class='cyber-badge badge-purple' style='margin-left:4px;'>QUANTUM V2</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Live Ticker Cards Grid
index_cols = st.columns(len(indices))
for i, (idx_name, d) in enumerate(indices.items()):
    with index_cols[i]:
        pchg = d.get("percent_change", 0)
        badge_cls = "badge-gain" if pchg >= 0 else "badge-loss"
        sign = "+" if pchg >= 0 else ""
        st.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8; font-weight:600;'>{idx_name}</div>"
            f"<div class='number-font' style='font-size:1.1rem; font-weight:700; margin-top:2px;'>"
            f"{d.get('value', 0):,.2f} "
            f"<span class='cyber-badge {badge_cls}' style='font-size:0.7rem;'>{sign}{pchg:.2f}%</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin-top:12px; margin-bottom:16px;'>", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR: RADAR CONTROLS & SQLITE WATCHLIST
# =========================================================================
with st.sidebar:
    st.markdown("### ⚡ RADAR CONTROLS")
    st.caption("Quantum Agent State & Memory Ledger")

    with st.expander("🔑 AI Intelligence Credentials", expanded=False):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=agent.api_key or "",
            placeholder="AI Studio Key (Free Tier)"
        )
        if st.button("Update AI Core", width="stretch"):
            st.session_state.agent = IndianStockRAGAgent(api_key=api_key_input)
            st.success("AI Core Synchronized!")
            st.rerun()

    # Active Memory Context Radar
    st.markdown(
        f"<div class='glass-card' style='padding:12px 14px; margin-top:10px;'>"
        f"<div style='font-size:0.75rem; color:#94a3b8; font-weight:600;'>⚡ WORKING CONTEXT (SHORT-TERM)</div>"
        f"<div style='margin-top:4px; font-weight:700; color:#38bdf8;'>Active: <span class='number-font'>{memory.last_mentioned_symbol or 'None'}</span></div>"
        f"<div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>Thread: {', '.join(memory.recent_symbols) if memory.recent_symbols else 'Idle'}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # SQLite Persistent Watchlist
    with st.expander("📋 SQLite Persistent Watchlist", expanded=True):
        watchlist = memory.get_watchlist()
        if watchlist:
            for item in watchlist:
                w_c1, w_c2 = st.columns([3, 1])
                with w_c1:
                    st.markdown(f"**{item['symbol']}**")
                    if item.get("notes"):
                        st.caption(f"📌 {item['notes']}")
                with w_c2:
                    if st.button("×", key=f"wl_del_{item['symbol']}"):
                        memory.remove_from_watchlist(item['symbol'])
                        st.rerun()
        else:
            st.caption("Watchlist empty. Tell agent: *'Add TCS to my watchlist'*")

    # Fast Pin to Watchlist
    with st.expander("➕ Pin Stock to Watchlist"):
        add_sym_side = st.text_input("NSE Symbol (e.g. HAL, IRFC, ZOMATO)")
        add_note_side = st.text_input("Research Thesis / Note")
        if st.button("Pin to Persistent Memory", width="stretch"):
            if add_sym_side:
                res_sym = nse.resolve_symbol(add_sym_side)
                memory.add_to_watchlist(res_sym, notes=add_note_side)
                st.success(f"Pinned {res_sym} to SQLite Database!")
                st.rerun()

# =========================================================================
# MAIN APP NAVIGATION TABS
# =========================================================================
# ----------------- MAIN APP NAVIGATION TABS -----------------
# IPO tab removed for streamlined UI
tab_copilot, tab_treemap, tab_chart, tab_portfolio, tab_wire = st.tabs([
    "💬 Quantum Copilot",
    "🗺️ Market Sector Treemap",
    "📊 Institutional Technical Radar",
    "💼 Portfolio Cockpit",
    "📰 Live Intelligence Wire"
])

# ----------------- TAB 1: QUANTUM COPILOT -----------------
with tab_copilot:
    st.markdown("##### ⚡ Tactical Command Matrix:")
    cmd1, cmd2, cmd3, cmd4, cmd5, cmd6 = st.columns(6)
    if cmd1.button("📊 Reliance Pivots"):
        st.session_state.pending_prompt = "What is the live price, RSI, and pivot levels for Reliance?"
    if cmd2.button("⚖️ TCS vs INFY"):
        st.session_state.pending_prompt = "Compare TCS and Infosys in terms of valuation, P/E, and dividend yield."
    if cmd3.button("🛡️ HAL Defense"):
        st.session_state.pending_prompt = "Explain HAL defense order book, Tejas fighter aircraft, and outlook."
    if cmd4.button("⚡ Zomato & Blinkit"):
        st.session_state.pending_prompt = "What is the latest news and growth drivers for Zomato?"
    if cmd5.button("🚂 IRFC Rail Trend"):
        st.session_state.pending_prompt = "What is the live quote, 200 EMA status, and technical bias of IRFC?"
    if cmd6.button("💰 INFY Valuation"):
        st.session_state.pending_prompt = "What are the fundamental valuation ratios, P/E, and ROE of Infosys?"

    # IPO command buttons removed to simplify UI

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "traces" in msg and msg["traces"]:
                with st.expander("🛠️ Autonomous Tool Execution Telemetry", expanded=False):
                    for t in msg["traces"]:
                        st.markdown(f"**Tool Invoked**: `{t.get('tool')}`")
                        st.json({"arguments": t.get("args"), "payload": t.get("output")})

    # Chat Input
    query = st.chat_input("Enter market prompt, comparison, or stock question (e.g. 'What is Swiggy IPO turnover and profit?')...")
    if getattr(st.session_state, "pending_prompt", None):
        query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing live market data & reasoning..."):
                ans, traces = agent.run(query)
                st.markdown(ans)
                if traces:
                    with st.expander("🛠️ Autonomous Tool Execution Telemetry", expanded=False):
                        for t in traces:
                            st.markdown(f"**Tool Invoked**: `{t.get('tool')}`")
                            st.json({"arguments": t.get("args"), "payload": t.get("output")})

        st.session_state.messages.append({
            "role": "assistant",
            "content": ans,
            "traces": traces
        })

# IPO UI block removed

# ----------------- TAB 3: MARKET SECTOR TREEMAP -----------------
with tab_treemap:
    st.subheader("🗺️ Indian Market Sector Treemap (NIFTY Heatmap)")
    st.caption("Visual heat map of key Indian equities grouped by industry sectors, sized by capitalization and color-coded by daily performance.")

    # Data for the interactive market treemap
    sample_stocks = [
        # Sector, Symbol, Company, Weight/Cap, % Change
        ("IT", "TCS", "Tata Consultancy", 140, 1.45),
        ("IT", "INFY", "Infosys", 110, 0.82),
        ("IT", "WIPRO", "Wipro Ltd", 45, -0.42),
        ("IT", "HCLTECH", "HCL Technologies", 55, 1.10),
        ("IT", "TECHM", "Tech Mahindra", 35, -0.15),
        
        ("Banking & Finance", "HDFCBANK", "HDFC Bank", 160, -0.35),
        ("Banking & Finance", "ICICIBANK", "ICICI Bank", 130, 1.25),
        ("Banking & Finance", "SBIN", "State Bank of India", 90, 0.65),
        ("Banking & Finance", "KOTAKBANK", "Kotak Mahindra", 70, -0.10),
        ("Banking & Finance", "BAJFINANCE", "Bajaj Finance", 80, 2.10),
        
        ("Energy & Oil", "RELIANCE", "Reliance Industries", 190, -0.31),
        ("Energy & Oil", "ONGC", "ONGC Ltd", 50, 0.95),
        ("Energy & Oil", "NTPC", "NTPC Ltd", 65, 1.40),
        ("Energy & Oil", "POWERGRID", "Power Grid Corp", 60, 0.20),
        
        ("Automobile", "TMPV", "Tata Motors PV", 70, 2.45),
        ("Automobile", "MARUTI", "Maruti Suzuki", 85, -0.80),
        ("Automobile", "M&M", "Mahindra & Mahindra", 75, 1.65),
        ("Automobile", "BAJAJ-AUTO", "Bajaj Auto", 60, 0.50),
        
        ("Defense & Infra", "HAL", "Hindustan Aeronautics", 65, 3.20),
        ("Defense & Infra", "BEL", "Bharat Electronics", 55, 1.80),
        ("Defense & Infra", "LT", "Larsen & Toubro", 100, 0.75),
        ("Defense & Infra", "IRFC", "Indian Railway Fin", 45, 2.10),
        
        ("FMCG & Consumer", "ITC", "ITC Limited", 95, 0.30),
        ("FMCG & Consumer", "HINDUNILVR", "Hindustan Unilever", 115, -0.90),
        ("FMCG & Consumer", "TRENT", "Trent Ltd (Zudio)", 60, 3.80),
        ("FMCG & Consumer", "ZOMATO", "Zomato Ltd", 70, 1.95),
    ]

    labels = ["NSE MARKET"]
    parents = [""]
    values = [0]
    colors = [0]
    custom_texts = ["Market Overview"]

    sectors = list(set([s[0] for s in sample_stocks]))
    for sec in sectors:
        labels.append(sec)
        parents.append("NSE MARKET")
        sec_stocks = [s for s in sample_stocks if s[0] == sec]
        sec_val = sum([s[3] for s in sec_stocks])
        sec_avg_chg = np.mean([s[4] for s in sec_stocks])
        values.append(sec_val)
        colors.append(sec_avg_chg)
        custom_texts.append(f"{sec} Sector: {sec_avg_chg:+.2f}%")

    for sec, sym, name, cap, chg in sample_stocks:
        labels.append(sym)
        parents.append(sec)
        values.append(cap)
        colors.append(chg)
        custom_texts.append(f"{name} ({sym})<br>Change: {chg:+.2f}%")

    tree_fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=[
                [0.0, "#ef4444"],
                [0.45, "#7f1d1d"],
                [0.5, "#1e293b"],
                [0.55, "#064e3b"],
                [1.0, "#10b981"]
            ],
            cmid=0.0,
            colorbar=dict(title="Day % Change", tickfont=dict(color="#f8fafc"))
        ),
        textinfo="label+value",
        hovertext=custom_texts,
        hoverinfo="text"
    ))

    tree_fig.update_layout(
        height=580,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#07090e",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc", size=13)
    )
    st.plotly_chart(tree_fig, width="stretch")

# ----------------- TAB 3: INSTITUTIONAL CHART & RADAR -----------------
with tab_chart:
    st.subheader("📊 Quantitative Technical Radar")
    
    t_c1, t_c2, t_c3 = st.columns([3, 2, 2])
    with t_c1:
        target_sym = st.text_input(
            "Ticker Symbol",
            value=memory.last_mentioned_symbol or st.session_state.selected_symbol,
            key="tab_chart_ticker_box"
        )
    with t_c2:
        chart_tf = st.selectbox("Candle History", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    with t_c3:
        st.write("")
        st.write("")
        pin_btn = st.button("📌 Pin to Watchlist", width="stretch")

    res_ticker = nse.resolve_symbol(target_sym)
    st.session_state.selected_symbol = res_ticker

    if pin_btn:
        memory.add_to_watchlist(res_ticker)
        st.success(f"Pinned {res_ticker} to SQLite Watchlist!")

    quote = nse.get_live_quote(res_ticker)
    df = nse.get_historical_ohlcv(res_ticker, period=chart_tf, interval="1d")

    if df is not None and not df.empty and len(df) > 12:
        indicators = TechnicalAnalyzer.calculate_indicators(df)

        # Telemetry HUD Row
        hud1, hud2, hud3, hud4, hud5, hud6 = st.columns(6)
        ltp = quote.get("last_price", df["Close"].iloc[-1])
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)
        badge_style = "badge-gain" if pchg >= 0 else "badge-loss"
        sign_char = "+" if pchg >= 0 else ""

        hud1.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>LIVE LTP</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700;'>₹{ltp:,.2f}</div>"
            f"<div class='cyber-badge {badge_style}'>{sign_char}{pchg:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hud2.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>200 EMA (BENCHMARK)</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700;'>₹{indicators.get('ema_200', 'N/A')}</div>"
            f"<div style='font-size:0.75rem; color:#c084fc;'>{'Above Trend' if indicators.get('price_above_ema200') else 'Below Trend'}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hud3.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>RSI (14) MOMENTUM</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700;'>{indicators.get('rsi_14')}</div>"
            f"<div style='font-size:0.75rem; color:#38bdf8;'>{indicators.get('rsi_condition')[:12]}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hud4.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>PIVOT POINT (P)</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700;'>₹{indicators.get('pivot_point')}</div>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>Classic Floor</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hud5.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>SUPPORT (S1)</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700; color:#34d399;'>₹{indicators.get('support_1')}</div>"
            f"<div style='font-size:0.75rem; color:#10b981;'>Key Demand</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hud6.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>RESISTANCE (R1)</div>"
            f"<div class='number-font' style='font-size:1.15rem; font-weight:700; color:#fb7185;'>₹{indicators.get('resistance_1')}</div>"
            f"<div style='font-size:0.75rem; color:#f43f5e;'>Key Supply</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        chart_side, dial_side = st.columns([3, 1])

        with chart_side:
            # Candlestick Subplots
            c_fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=[f"{quote.get('company_name', res_ticker)} ({res_ticker}) // Real-Time Chart", "Volume Distribution"]
            )

            # Candlestick
            c_fig.add_trace(
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Candles",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#f43f5e"
                ),
                row=1, col=1
            )

            # Overlays
            ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
            ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
            ema_200 = df["Close"].ewm(span=200, adjust=False).mean() if len(df) >= 40 else None

            c_fig.add_trace(go.Scatter(x=df["Date"], y=ema_20, line=dict(color="#38bdf8", width=1.5), name="EMA 20"), row=1, col=1)
            c_fig.add_trace(go.Scatter(x=df["Date"], y=ema_50, line=dict(color="#fbbf24", width=1.5), name="EMA 50"), row=1, col=1)
            if ema_200 is not None:
                c_fig.add_trace(go.Scatter(x=df["Date"], y=ema_200, line=dict(color="#c084fc", width=2), name="EMA 200"), row=1, col=1)

            # Volume
            vol_colors = ["#10b981" if c >= o else "#f43f5e" for c, o in zip(df["Close"], df["Open"])]
            c_fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

            c_fig.update_layout(
                height=560,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                paper_bgcolor="#0b0e14",
                plot_bgcolor="#0b0e14",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(c_fig, width="stretch")

        with dial_side:
            # RSI Speedometer Gauge
            rsi_score = indicators.get("rsi_14", 50)
            dial_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_score,
                title={"text": "RSI (14) MOMENTUM", "font": {"size": 13, "color": "#94a3b8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#38bdf8", "thickness": 0.3},
                    "bgcolor": "#131926",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16, 185, 129, 0.35)"},
                        {"range": [30, 70], "color": "rgba(71, 85, 105, 0.25)"},
                        {"range": [70, 100], "color": "rgba(244, 63, 94, 0.35)"}
                    ],
                    "threshold": {
                        "line": {"color": "#f43f5e", "width": 3},
                        "thickness": 0.8,
                        "value": rsi_score
                    }
                }
            ))
            dial_fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="#0b0e14",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"}
            )
            st.plotly_chart(dial_fig, width="stretch")

            # Trade Setup Matrix Card
            st.markdown(
                f"<div class='glass-card' style='padding:14px; margin-top:-10px;'>"
                f"<div style='font-size:0.75rem; color:#94a3b8; font-weight:700; letter-spacing:0.04em;'>ATR VOLATILITY TARGETS</div>"
                f"<div style='margin-top:6px; font-size:0.85rem;'>Daily True Range: <b class='number-font'>₹{indicators.get('atr_14')}</b></div>"
                f"<div style='margin-top:4px; font-size:0.85rem; color:#f43f5e;'>Stop-Loss: <b class='number-font'>₹{indicators.get('suggested_stop_loss')}</b></div>"
                f"<div style='margin-top:4px; font-size:0.85rem; color:#10b981;'>Profit Target: <b class='number-font'>₹{indicators.get('suggested_target')}</b></div>"
                f"<div style='margin-top:8px; font-size:0.8rem; color:#38bdf8;'>Stance: <b>{indicators.get('overall_bias')}</b></div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning(f"Could not load chart candles for {res_ticker}.")

# ----------------- TAB 4: PORTFOLIO COCKPIT -----------------
with tab_portfolio:
    st.subheader("💼 Real-Time Portfolio Cockpit & Risk Distribution")

    # Add position drawer
    with st.expander("➕ Add Position to Portfolio"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sym_in = st.text_input("NSE Symbol (e.g. INFY, HAL, BEL, ZOMATO)")
        with f2:
            qty_in = st.number_input("Shares Quantity", min_value=1, value=10, step=1)
        with f3:
            buy_in = st.number_input("Average Buy Price (₹)", min_value=0.5, value=1000.0, step=10.0)
        with f4:
            sec_in = st.selectbox("Sector Classification", ["IT", "Banking", "Auto", "Defense", "Energy", "FMCG", "Railways", "Other"])

        if st.button("Log Position to Cockpit", width="stretch"):
            if sym_in:
                r_sym = nse.resolve_symbol(sym_in)
                st.session_state.portfolio.append({
                    "symbol": r_sym,
                    "qty": qty_in,
                    "buy_price": buy_in,
                    "sector": sec_in
                })
                st.success(f"Position in {r_sym} logged!")
                st.rerun()

    if st.session_state.portfolio:
        p_rows = []
        tot_inv = 0.0
        tot_cur = 0.0
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

            tot_inv += inv_val
            tot_cur += cur_val
            sector_weights[sec] = sector_weights.get(sec, 0.0) + cur_val

            p_rows.append({
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

        net_pnl = tot_cur - tot_inv
        net_pct = (net_pnl / tot_inv * 100) if tot_inv else 0.0

        # Telemetry cards
        pt1, pt2, pt3 = st.columns(3)
        pt1.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>TOTAL INVESTED CAPITAL</div>"
            f"<div class='number-font' style='font-size:1.35rem; font-weight:700;'>₹{tot_inv:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        pt2.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>CURRENT PORTFOLIO VALUE</div>"
            f"<div class='number-font' style='font-size:1.35rem; font-weight:700; color:#38bdf8;'>₹{tot_cur:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        pnl_badge = "badge-gain" if net_pnl >= 0 else "badge-loss"
        sign_p = "+" if net_pnl >= 0 else ""
        pt3.markdown(
            f"<div class='hud-card'>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>TOTAL UNREALIZED P&L</div>"
            f"<div class='number-font' style='font-size:1.35rem; font-weight:700;'>"
            f"{sign_p}₹{net_pnl:,.2f} <span class='cyber-badge {pnl_badge}'>{sign_p}{net_pct:.2f}%</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        chart_w_col, table_w_col = st.columns([1, 2])

        with chart_w_col:
            donut_fig = go.Figure(data=[go.Pie(
                labels=list(sector_weights.keys()),
                values=list(sector_weights.values()),
                hole=0.6,
                marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]),
                textinfo="label+percent"
            )])
            donut_fig.update_layout(
                title={"text": "Sector Weighting Distribution", "font": {"size": 13, "color": "#94a3b8"}},
                height=300,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="#0b0e14",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"},
                showlegend=False
            )
            st.plotly_chart(donut_fig, width="stretch")

        with table_w_col:
            st.markdown("##### 📋 Active Positions Ledger")
            st.dataframe(pd.DataFrame(p_rows), width="stretch")

        if st.button("🗑️ Reset Entire Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Portfolio ledger is empty. Add a holding above.")

# ----------------- TAB 5: LIVE INTELLIGENCE WIRE -----------------
with tab_wire:
    st.subheader("📰 Market Intelligence Wire & Regulatory Disclosures")

    w1, w2 = st.columns([1, 1])

    with w1:
        st.markdown("#### 🔴 Real-Time Media Stream")
        stream_stock = st.text_input("Stream News for Stock", value="RELIANCE", key="wire_stock_input")
        if stream_stock:
            news_feed = nse.get_live_stock_news(stream_stock, limit=5)
            for item in news_feed:
                st.markdown(
                    f"<div class='glass-card' style='padding:14px; margin-bottom:10px;'>"
                    f"<div style='font-weight:600; font-size:0.92rem;'><a href='{item['link']}' target='_blank' style='color:#60a5fa; text-decoration:none;'>{item['title']}</a></div>"
                    f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:4px;'>Source: <b style='color:#cbd5e1;'>{item['source']}</b> • <i>{item['published']}</i></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    with w2:
        st.markdown("#### 🔍 Semantic Corporate Filings RAG")
        filing_query = st.text_input("Semantic Retrieval Query", placeholder="e.g. Tejas defense capex, Blinkit dark stores, green hydrogen...")
        if filing_query:
            k_results = agent.knowledge.search(filing_query)
            for r in k_results:
                st.markdown(
                    f"<div class='glass-card' style='padding:14px; margin-bottom:10px;'>"
                    f"<div><span class='cyber-badge badge-cyan'>{r.get('symbol')}</span> <span style='font-size:0.75rem; color:#94a3b8; margin-left:6px;'>{r.get('category')}</span></div>"
                    f"<div style='font-weight:700; margin-top:6px; font-size:0.95rem;'>{r.get('title')}</div>"
                    f"<div style='font-size:0.85rem; color:#cbd5e1; margin-top:4px;'>{r.get('content')}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Search across corporate announcements and business profiles.")



