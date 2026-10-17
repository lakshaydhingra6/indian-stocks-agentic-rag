import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dotenv import load_dotenv

# Load local environment
load_dotenv()

from core.agent import IndianStockRAGAgent
from core.nse_client import NSEClient
from core.technicals import TechnicalAnalyzer
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore

# Page configuration
st.set_page_config(
    page_title="Artha-Agent: Indian Stocks Agentic RAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern sleek financial UI
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
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
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
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🙏 I am **Artha-Agent**, your Agentic RAG assistant for the **Indian Stock Market (NSE)**.\n\n"
                       "I have real-time live data feeds, technical analysis calculation engines, corporate filings RAG, "
                       "and dual-tier memory.\n\n"
                       "Try asking me:\n"
                       "- *'What is the current price and technical trend of Reliance?'*\n"
                       "- *'How does that compare to TCS?'* (Tests short-term pronoun memory!)\n"
                       "- *'Add Tata Motors to my watchlist and tell me its 200 EMA'* (Tests persistent SQLite memory!)\n"
                       "- *'Give me an overview of the Indian market today'*"
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

# Sidebar: Configuration, Memory Inspector & Watchlist
with st.sidebar:
    st.title("🇮🇳 Artha-Agent")
    st.caption("Agentic RAG with Live NSE Data & Dual-Tier Memory")
    
    # API Key Config
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
    
    # 1. Short-term Working Memory
    with st.expander("⚡ Working Memory (In-Session)", expanded=True):
        st.write(f"**Active Ticker**: `{memory.last_mentioned_symbol or 'None'}`")
        if memory.recent_symbols:
            st.write(f"**Recent Tickers**: {', '.join(memory.recent_symbols)}")
        else:
            st.caption("No tickers mentioned yet in this session.")

    # 2. Long-term Persistent Memory (Watchlist)
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
            st.caption("Watchlist is empty. Ask the agent: *'Add Reliance to my watchlist'*.")

    # Quick Action to add stock
    with st.expander("➕ Add Stock to Watchlist"):
        add_sym = st.text_input("NSE Symbol (e.g. INFY, TCS)", key="add_sym_input")
        add_note = st.text_input("Notes (optional)", key="add_note_input")
        if st.button("Save to Memory"):
            if add_sym:
                resolved = nse.resolve_symbol(add_sym)
                memory.add_to_watchlist(resolved, notes=add_note)
                st.success(f"Saved {resolved} to persistent database!")
                st.rerun()

# Main Application Tabs
tab_chat, tab_chart, tab_knowledge = st.tabs([
    "💬 Agentic RAG Copilot",
    "📊 Live Interactive Chart & Technicals",
    "📚 Vector Knowledge Store"
])

# ----------------- TAB 1: AGENTIC RAG CHAT -----------------
with tab_chat:
    # Quick prompt chips
    st.write("##### 💡 Suggested Questions:")
    chip_cols = st.columns(4)
    if chip_cols[0].button("📊 Reliance LTP & Technicals"):
        st.session_state.pending_prompt = "What is the live price and technical indicators for Reliance?"
    if chip_cols[1].button("🔄 Compare it with TCS"):
        st.session_state.pending_prompt = "How does that compare to TCS in terms of price and RSI?"
    if chip_cols[2].button("📋 Add Tata Motors to Watchlist"):
        st.session_state.pending_prompt = "Add Tata Motors to my watchlist with note 'Bullish EV growth'."
    if chip_cols[3].button("🌐 Indian Market Overview"):
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

    # Chat Input Handling
    prompt = st.chat_input("Ask about Indian stocks, technical levels, memory, or market status...")
    if getattr(st.session_state, "pending_prompt", None):
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent reasoning & response
        with st.chat_message("assistant"):
            with st.spinner("Artha-Agent is querying live NSE data & memory..."):
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
    st.subheader("📈 Live Technical Charting")
    col_input, col_period = st.columns([3, 2])
    with col_input:
        chart_symbol_input = st.text_input(
            "Stock Symbol",
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

    # Fetch live quote and historical OHLCV
    quote = nse.get_live_quote(resolved_sym)
    df = nse.get_historical_ohlcv(resolved_sym, period=chart_period, interval="1d")

    if df is not None and not df.empty and len(df) > 10:
        # Metric tiles
        m1, m2, m3, m4, m5 = st.columns(5)
        ltp = quote.get("last_price", df["Close"].iloc[-1])
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)
        
        m1.metric("LTP (₹)", f"{ltp:,.2f}", f"{chg:+.2f} ({pchg:+.2f}%)")
        m2.metric("Day High", f"₹{quote.get('day_high', df['High'].iloc[-1]):,.2f}")
        m3.metric("Day Low", f"₹{quote.get('day_low', df['Low'].iloc[-1]):,.2f}")
        m4.metric("52W High", f"₹{quote.get('week_52_high', 0):,.2f}")
        m5.metric("52W Low", f"₹{quote.get('week_52_low', 0):,.2f}")

        # Compute Technicals
        indicators = TechnicalAnalyzer.calculate_indicators(df)

        # Plotly Candlestick + Volume + RSI Chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25],
            subplot_titles=[f"{quote.get('company_name', resolved_sym)} ({resolved_sym}) Price & EMAs", "RSI (14) Indicator"]
        )

        # Candlestick
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

        # EMAs
        ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["Close"].ewm(span=200, adjust=False).mean() if len(df) >= 50 else None

        fig.add_trace(go.Scatter(x=df["Date"], y=ema_20, line=dict(color="#29b6f6", width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=ema_50, line=dict(color="#ffa726", width=1.5), name="EMA 50"), row=1, col=1)
        if ema_200 is not None:
            fig.add_trace(go.Scatter(x=df["Date"], y=ema_200, line=dict(color="#ab47bc", width=2), name="EMA 200"), row=1, col=1)

        # RSI
        rsi_series = TechnicalAnalyzer.calculate_rsi(df["Close"], 14)
        fig.add_trace(go.Scatter(x=df["Date"], y=rsi_series, line=dict(color="#e040fb", width=1.5), name="RSI 14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Indicator Summary Badges
        c_tech1, c_tech2, c_tech3, c_tech4 = st.columns(4)
        c_tech1.info(f"**Overall Bias**: {indicators.get('overall_bias')}")
        c_tech2.info(f"**RSI (14)**: {indicators.get('rsi_14')} ({indicators.get('rsi_condition')})")
        c_tech3.info(f"**MACD Signal**: {indicators.get('macd_crossover')}")
        c_tech4.info(f"**Support / Resist**: ₹{indicators.get('support_level')} / ₹{indicators.get('resistance_level')}")
    else:
        st.warning(f"Could not load chart data for '{resolved_sym}'. Please verify the stock ticker.")

# ----------------- TAB 3: KNOWLEDGE STORE -----------------
with tab_knowledge:
    st.subheader("📚 Corporate Knowledge Store & Disclosures")
    st.caption("Local semantic knowledge base for corporate developments, business overviews, and capex notes.")

    # Search Bar
    k_query = st.text_input("Semantic Search within Corporate Knowledge", placeholder="e.g. EV battery gigafactory, cloud margins, merger...")
    if k_query:
        k_results = agent.knowledge.search(k_query)
        st.write(f"Found **{len(k_results)}** matching disclosures:")
        for r in k_results:
            st.markdown(f"### {r.get('symbol')} — {r.get('title')}")
            st.caption(f"Category: {r.get('category')}")
            st.write(r.get("content"))
            st.divider()
    else:
        st.write("#### Existing Ingested Corporate Knowledge:")
        for doc in agent.knowledge.documents:
            with st.expander(f"{doc.get('symbol')} — {doc.get('title')}"):
                st.caption(f"Category: {doc.get('category')}")
                st.write(doc.get("content"))

    # Add new note/filing
    st.write("#### ➕ Ingest New Corporate Announcement / Disclosure")
    with st.form("new_doc_form"):
        new_sym = st.text_input("Stock Symbol (e.g. RELIANCE, ZOMATO)")
        new_title = st.text_input("Disclosure Title")
        new_category = st.selectbox("Category", ["Quarterly Results", "Corporate Action", "Capex & Strategy", "Management Commentary"])
        new_content = st.text_area("Content / Full Text")
        submitted = st.form_submit_button("Ingest into Vector Store")
        if submitted:
            if new_sym and new_title and new_content:
                agent.knowledge.add_document(
                    symbol=new_sym,
                    title=new_title,
                    content=new_content,
                    category=new_category
                )
                st.success(f"Ingested disclosure for {new_sym.upper()} into local vector store!")
                st.rerun()
            else:
                st.error("Please fill in symbol, title, and content.")
