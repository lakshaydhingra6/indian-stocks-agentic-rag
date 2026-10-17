import os
import streamlit as st
import pandas as pd
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
    page_title="Artha-Agent: Advanced Indian Stocks Agentic RAG",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .metric-card {
        background: #1e222d;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #2a2e39;
        margin-bottom: 8px;
    }
    .ticker-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .pill-green {
        background-color: rgba(38, 166, 154, 0.2);
        color: #26a69a;
        border: 1px solid #26a69a;
    }
    .pill-red {
        background-color: rgba(239, 83, 80, 0.2);
        color: #ef5350;
        border: 1px solid #ef5350;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = IndianStockRAGAgent()
if "portfolio" not in st.session_state:
    # Sample starter portfolio
    st.session_state.portfolio = [
        {"symbol": "RELIANCE", "qty": 10, "buy_price": 1220.0},
        {"symbol": "TCS", "qty": 5, "buy_price": 2400.0},
        {"symbol": "TATAMOTORS", "qty": 25, "buy_price": 850.0}
    ]
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Artha-Agent**, your Quantitative AI Research Copilot for the **Indian Stock Market (NSE/BSE)**.\n\n"
                       "Equipped with live real-time feeds, technical engines, fundamental valuation ratios, live financial news, "
                       "and dual-tier memory.\n\n"
                       "**Questions you can ask me right now:**\n"
                       "- 📊 *'What is Reliance's current price, pivot points, and technical trend?'*\n"
                       "- 💰 *'What are the fundamentals, P/E ratio, and valuation of Infosys?'*\n"
                       "- ⚖️ *'Compare TCS and Infosys'* (or *'How does that compare to TCS?'*)\n"
                       "- 📰 *'What is the latest news and sentiment for Zomato / Tata Motors?'*\n"
                       "- 📋 *'Add HAL to my watchlist with note: High defense order book'*"
        }
    ]
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "RELIANCE"

agent: IndianStockRAGAgent = st.session_state.agent
memory: DualTierMemoryManager = agent.memory
nse: NSEClient = agent.nse

# Top Live Ticker Tape
indices = nse.get_major_indices()
status = nse.get_market_status()

tape_cols = st.columns(len(indices) + 1 if indices else 1)
with tape_cols[0]:
    st.markdown(f"**NSE Status**: `{status.get('status', 'Open')}`")

for idx, (name, val) in enumerate(indices.items()):
    with tape_cols[idx + 1]:
        pchg = val.get("percent_change", 0)
        color = "pill-green" if pchg >= 0 else "pill-red"
        symbol_prefix = "+" if pchg >= 0 else ""
        st.markdown(
            f"**{name}**: {val.get('value', 0):,.2f} "
            f"<span class='ticker-pill {color}'>{symbol_prefix}{pchg:.2f}%</span>",
            unsafe_allow_html=True
        )

st.divider()

# Sidebar: Config, Dual-Tier Memory & Watchlist
with st.sidebar:
    st.title("🇮🇳 Artha-Agent")
    st.caption("Agentic RAG with Live NSE Data & Dual-Tier Memory")
    
    with st.expander("🔑 API Key & Model Settings", expanded=False):
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=agent.api_key or "",
            help="Free from https://aistudio.google.com/app/apikey. If empty, deterministic smart agent mode runs!"
        )
        if st.button("Update Key"):
            st.session_state.agent = IndianStockRAGAgent(api_key=api_key_input)
            st.success("API key updated!")
            st.rerun()

    # Memory Inspector Section
    st.subheader("🧠 Dual-Tier Memory")
    
    with st.expander("⚡ Working Memory (In-Session)", expanded=True):
        st.write(f"**Active Ticker**: `{memory.last_mentioned_symbol or 'None'}`")
        if memory.recent_symbols:
            st.write(f"**Recent Tickers**: {', '.join(memory.recent_symbols)}")
        else:
            st.caption("No tickers mentioned yet.")

    with st.expander("📋 Persistent Watchlist (SQLite)", expanded=True):
        watchlist = memory.get_watchlist()
        if watchlist:
            for item in watchlist:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{item['symbol']}** ({item['company_name']})")
                    if item.get("notes"):
                        st.caption(f"📝 {item['notes']}")
                with c2:
                    if st.button("❌", key=f"del_{item['symbol']}", help=f"Remove {item['symbol']}"):
                        memory.remove_from_watchlist(item['symbol'])
                        st.rerun()
        else:
            st.caption("Watchlist is empty. Ask: *'Add Reliance to my watchlist'*.")

    with st.expander("➕ Add Stock to Watchlist"):
        add_sym = st.text_input("NSE Symbol (e.g. INFY, HAL, IRFC)", key="add_sym_input")
        add_note = st.text_input("Notes (optional)", key="add_note_input")
        if st.button("Save to Memory"):
            if add_sym:
                resolved = nse.resolve_symbol(add_sym)
                memory.add_to_watchlist(resolved, notes=add_note)
                st.success(f"Saved {resolved} to persistent database!")
                st.rerun()

# Main Application Tabs
tab_chat, tab_chart, tab_portfolio, tab_knowledge = st.tabs([
    "💬 Agentic RAG Copilot",
    "📊 Live Interactive Chart & Technicals",
    "💼 Portfolio P&L Tracker",
    "📚 Vector Knowledge Store & News"
])

# ----------------- TAB 1: AGENTIC RAG CHAT -----------------
with tab_chat:
    st.write("##### 💡 Quick Prompt Triggers:")
    row1 = st.columns(4)
    if row1[0].button("📊 Reliance Live & Technicals"):
        st.session_state.pending_prompt = "What is the live price, RSI, and pivot levels for Reliance?"
    if row1[1].button("💰 Infosys Fundamentals & P/E"):
        st.session_state.pending_prompt = "What are the fundamental valuation ratios, P/E, and ROE of Infosys?"
    if row1[2].button("⚖️ Compare TCS vs Infosys"):
        st.session_state.pending_prompt = "Compare TCS and Infosys in terms of valuation, price, and dividend yield."
    if row1[3].button("📰 Zomato News & Blinkit"):
        st.session_state.pending_prompt = "What is the latest news and growth drivers for Zomato?"

    row2 = st.columns(4)
    if row2[0].button("🚀 HAL Defense Order Book"):
        st.session_state.pending_prompt = "Explain Hindustan Aeronautics (HAL) defense order book and outlook."
    if row2[1].button("🚂 IRFC / Railways Trend"):
        st.session_state.pending_prompt = "What is the live quote and technical bias of IRFC?"
    if row2[2].button("🔄 Context: Compare it with TCS"):
        st.session_state.pending_prompt = "How does that compare to TCS?"
    if row2[3].button("🌐 Major Indian Market Status"):
        st.session_state.pending_prompt = "Give me a live overview of the Indian stock market and indices."

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "traces" in msg and msg["traces"]:
                with st.expander("🛠️ View Agent Tool Calls & Data Retrieval", expanded=False):
                    for t in msg["traces"]:
                        st.markdown(f"**Tool Invoked**: `{t.get('tool')}`")
                        st.json({"arguments": t.get("args"), "response_preview": t.get("output")})

    # Chat Input
    prompt = st.chat_input("Ask about Indian stocks, fundamentals, technicals, news, or memory...")
    if getattr(st.session_state, "pending_prompt", None):
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Artha-Agent is executing tools & synthesizing analysis..."):
                response_text, traces = agent.run(prompt)
                st.markdown(response_text)
                if traces:
                    with st.expander("🛠️ View Agent Tool Calls & Data Retrieval", expanded=False):
                        for t in traces:
                            st.markdown(f"**Tool Invoked**: `{t.get('tool')}`")
                            st.json({"arguments": t.get("args"), "response_preview": t.get("output")})

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "traces": traces
        })

# ----------------- TAB 2: INTERACTIVE CHART & TECHNICALS -----------------
with tab_chart:
    st.subheader("📈 Live Technical Charting & Quantitative Setup")
    col_input, col_period = st.columns([3, 2])
    with col_input:
        chart_symbol_input = st.text_input(
            "Stock Symbol (e.g. RELIANCE, TCS, HAL, ZOMATO, TRENT)",
            value=memory.last_mentioned_symbol or st.session_state.selected_symbol,
            key="chart_symbol_key"
        )
    with col_period:
        chart_period = st.selectbox(
            "Timeframe",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )

    resolved_sym = nse.resolve_symbol(chart_symbol_input)
    st.session_state.selected_symbol = resolved_sym

    quote = nse.get_live_quote(resolved_sym)
    df = nse.get_historical_ohlcv(resolved_sym, period=chart_period, interval="1d")

    if df is not None and not df.empty and len(df) > 10:
        m1, m2, m3, m4, m5 = st.columns(5)
        ltp = quote.get("last_price", df["Close"].iloc[-1])
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)
        
        m1.metric("LTP (₹)", f"{ltp:,.2f}", f"{chg:+.2f} ({pchg:+.2f}%)")
        m2.metric("Day High", f"₹{quote.get('day_high', df['High'].iloc[-1]):,.2f}")
        m3.metric("Day Low", f"₹{quote.get('day_low', df['Low'].iloc[-1]):,.2f}")
        m4.metric("52W High", f"₹{quote.get('week_52_high', 0):,.2f}")
        m5.metric("52W Low", f"₹{quote.get('week_52_low', 0):,.2f}")

        indicators = TechnicalAnalyzer.calculate_indicators(df)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25],
            subplot_titles=[f"{quote.get('company_name', resolved_sym)} ({resolved_sym}) Price & EMAs", "RSI (14) Momentum Indicator"]
        )

        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC"
            ),
            row=1, col=1
        )

        ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["Close"].ewm(span=200, adjust=False).mean() if len(df) >= 40 else None

        fig.add_trace(go.Scatter(x=df["Date"], y=ema_20, line=dict(color="#29b6f6", width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=ema_50, line=dict(color="#ffa726", width=1.5), name="EMA 50"), row=1, col=1)
        if ema_200 is not None:
            fig.add_trace(go.Scatter(x=df["Date"], y=ema_200, line=dict(color="#ab47bc", width=2), name="EMA 200"), row=1, col=1)

        rsi_series = TechnicalAnalyzer.calculate_rsi(df["Close"], 14)
        fig.add_trace(go.Scatter(x=df["Date"], y=rsi_series, line=dict(color="#e040fb", width=1.5), name="RSI 14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(
            height=620,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        c_t1.info(f"**Overall Bias**: {indicators.get('overall_bias')}")
        c_t2.info(f"**MA Crossover**: {indicators.get('ma_crossover_status')}")
        c_t3.info(f"**RSI (14)**: {indicators.get('rsi_14')} ({indicators.get('rsi_condition')})")
        c_t4.info(f"**Pivot Levels**: Pivot ₹{indicators.get('pivot_point')} | S1: ₹{indicators.get('support_1')} | R1: ₹{indicators.get('resistance_1')}")
    else:
        st.warning(f"Could not load historical candles for '{resolved_sym}'.")

# ----------------- TAB 3: PORTFOLIO TRACKER -----------------
with tab_portfolio:
    st.subheader("💼 Real-Time Portfolio P&L Tracker")
    st.caption("Track your actual Indian equity holdings with live LTP, P&L, and returns.")

    # Add holding
    with st.expander("➕ Add New Stock Holding to Portfolio"):
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            p_sym = st.text_input("Stock Symbol (e.g. TCS, RELIANCE)", key="p_sym_in")
        with p_c2:
            p_qty = st.number_input("Quantity", min_value=1, value=10, step=1, key="p_qty_in")
        with p_c3:
            p_buy = st.number_input("Average Buy Price (₹)", min_value=0.1, value=1000.0, step=10.0, key="p_buy_in")

        if st.button("Add to Portfolio"):
            if p_sym:
                res_sym = nse.resolve_symbol(p_sym)
                st.session_state.portfolio.append({"symbol": res_sym, "qty": p_qty, "buy_price": p_buy})
                st.success(f"Added {p_qty} shares of {res_sym} to your portfolio!")
                st.rerun()

    # Portfolio Calculation
    if st.session_state.portfolio:
        portfolio_rows = []
        total_invested = 0.0
        total_current = 0.0

        for h in st.session_state.portfolio:
            sym = h["symbol"]
            qty = h["qty"]
            buy_p = h["buy_price"]
            
            q = nse.get_live_quote(sym)
            cur_p = q.get("last_price", buy_p)
            
            invested = qty * buy_p
            curr_val = qty * cur_p
            pnl = curr_val - invested
            pnl_pct = (pnl / invested * 100) if invested else 0.0

            total_invested += invested
            total_current += curr_val

            portfolio_rows.append({
                "Symbol": sym,
                "Qty": qty,
                "Avg Buy (₹)": f"₹{buy_p:,.2f}",
                "Live LTP (₹)": f"₹{cur_p:,.2f}",
                "Invested (₹)": f"₹{invested:,.2f}",
                "Current Value (₹)": f"₹{curr_val:,.2f}",
                "P&L (₹)": f"{'+' if pnl >= 0 else ''}₹{pnl:,.2f}",
                "Return (%)": f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            })

        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

        pt_c1, pt_c2, pt_c3 = st.columns(3)
        pt_c1.metric("Total Invested", f"₹{total_invested:,.2f}")
        pt_c2.metric("Current Portfolio Value", f"₹{total_current:,.2f}")
        pt_c3.metric("Total P&L", f"{'+' if total_pnl >= 0 else ''}₹{total_pnl:,.2f}", f"{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct:.2f}%")

        st.dataframe(pd.DataFrame(portfolio_rows), use_container_width=True)

        if st.button("🗑️ Clear Entire Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Portfolio is empty. Add a holding above to start tracking real-time P&L!")

# ----------------- TAB 4: KNOWLEDGE STORE & NEWS -----------------
with tab_knowledge:
    st.subheader("📚 Corporate Knowledge Store & Real-Time News")

    # Section 1: Live News Fetcher
    st.markdown("#### 📰 Live Indian Financial News by Stock")
    news_stock_input = st.text_input("Enter Stock for Real-Time Headlines", value="RELIANCE", key="news_sym_key")
    if news_stock_input:
        n_res = nse.get_live_stock_news(news_stock_input, limit=5)
        for article in n_res:
            st.markdown(f"**[{article['title']}]({article['link']})**")
            st.caption(f"Source: {article['source']} • Published: {article['published']}")

    st.divider()

    # Section 2: Semantic Knowledge Search
    st.markdown("#### 🔍 Semantic Search across Ingested Corporate Knowledge")
    k_query = st.text_input("Search corporate disclosures, defense order books, capex plans...", placeholder="e.g. Tejas defense order, Blinkit quick commerce, green hydrogen...")
    if k_query:
        k_results = agent.knowledge.search(k_query)
        st.write(f"Found **{len(k_results)}** matching disclosures:")
        for r in k_results:
            st.markdown(f"### {r.get('symbol')} — {r.get('title')}")
            st.caption(f"Category: {r.get('category')}")
            st.write(r.get("content"))
            st.divider()
    else:
        st.write("##### Existing Ingested Corporate Knowledge Base:")
        for doc in agent.knowledge.documents:
            with st.expander(f"{doc.get('symbol')} — {doc.get('title')}"):
                st.caption(f"Category: {doc.get('category')}")
                st.write(doc.get("content"))
