# 🇮🇳 Artha-Agent: Agentic RAG for Indian Stocks with Live NSE Data & Memory

An autonomous, multi-turn AI Financial Research Agent and RAG system for the **Indian Stock Market (NSE/BSE)**. It combines real-time stock feeds, technical indicator engines, local corporate knowledge retrieval, and an efficient **dual-tier memory system**—running entirely on a **100% free-of-cost stack**.

---

## 🚀 Key Features

1. **🔴 Live NSE Market Feeds (Real-Time)**:
   - Live quotes for all NSE-listed equities (LTP, % change, open, day high/low, 52-week high/low, circuit limits, order book depth).
   - Market status indicator (Capital Market Open/Close) and major indices tracking (NIFTY 50, BANK NIFTY, SENSEX, NIFTY IT).
   - Resilient automatic fallback mechanism to Yahoo Finance (`.NS` tickers) ensuring 100% uninterrupted uptime.

2. **📊 Quantitative Technical Analysis Engine**:
   - Relative Strength Index (RSI 14) with overbought/oversold boundaries.
   - Exponential Moving Averages: EMA 20, EMA 50, and the institutional benchmark EMA 200.
   - MACD (12, 26, 9) signal line crossovers & histogram.
   - Dynamic 30-day Support & Resistance level calculation and overall trend bias (Bullish / Neutral / Bearish).

3. **🧠 Dual-Tier Memory System (Zero Cost, Zero Cloud DB)**:
   - **Tier 1: Short-Term Working Memory (In-Session)**:
     - Tracks active tickers and resolves conversational pronouns (e.g. *"What is its 200 EMA?"* or *"Compare it with TCS"* resolves *"it"* to the stock you just discussed).
   - **Tier 2: Long-Term Persistent Memory (Local SQLite)**:
     - Saves your custom watchlist, stock research notes, and user preferences across app restarts in `data/memory.db`.

4. **📚 Local Vector Knowledge Store (RAG Engine)**:
   - Ingests corporate disclosures, earnings summaries, capex updates, and business segment profiles.
   - Semantic term-scoring search that can be filtered by stock ticker.
   - Ability to ingest custom company announcements directly through the UI.

5. **📈 Interactive Streamlit Dashboard**:
   - Live ticker ribbon at the top.
   - Interactive Plotly Candlestick + Volume + RSI charts with customizable timeframes (1M, 3M, 6M, 1Y, 5Y).
   - Chat Copilot with full **Agent Tool Execution Transparency** (expandable view of tools invoked, parameters passed, and raw JSON outputs).

6. **💸 100% Free Stack**:
   - Free Google Gemini API tier (`gemini-2.5-flash` / `gemini-1.5-flash`).
   - SQLite for persistence (no cloud database bill).
   - Streamlit for dashboard (can be run locally or deployed for free on Streamlit Community Cloud).
   - Intelligent fallback agent mode that works out-of-the-box even before entering an API key.

---

## 📂 Project Structure

```
Project/
├── .env.example              # Template for API keys
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings, aliases, headers
│
├── core/
│   ├── __init__.py
│   ├── nse_client.py         # Real-time NSE API client + resilient fallback
│   ├── technicals.py         # Technical analysis (RSI, EMAs, MACD, Support/Resistance)
│   ├── memory_manager.py     # Dual-tier memory (Working context + SQLite persistence)
│   ├── vector_store.py       # Local semantic knowledge store & RAG
│   └── agent.py              # Autonomous Agentic RAG engine & tool calling loop
│
├── tools/
│   ├── __init__.py
│   └── stock_tools.py        # Tool registry exposed to the LLM agent
│
├── data/
│   ├── memory.db             # Local SQLite database
│   └── knowledge_store.json  # Pre-seeded Indian corporate disclosures
│
├── app.py                    # Streamlit interactive web application
├── test_pipeline.py          # Automated verification script
└── README.md
```

---

## 🛠️ Quickstart Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
To enable full generative reasoning with Google Gemini:
1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `.env.example` to `.env` and set your key:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
*(Note: If no API key is provided, the system seamlessly runs in Smart Deterministic Agent mode, allowing you to test live NSE quotes, charts, and memory immediately!)*

---

## 🚀 Running the Application

### Launch the Streamlit Web Dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Automated Tests:
```bash
python test_pipeline.py
```

---

## 🧪 Example Queries to Test

- **Live Quote & Technicals**: *"What is the current price and technical trend of Reliance?"*
- **Pronoun Resolution (Working Memory)**: *"How does that compare to TCS?"*
- **Persistent Watchlist (Long-Term Memory)**: *"Add Tata Motors to my watchlist with note 'Bullish EV sales'."*
- **Market Overview**: *"Give me a live overview of the Indian stock market and indices."*
- **Corporate Filings (RAG)**: *"Tell me about Infosys AI platforms and revenue mix."*

---

## ⚖️ Disclaimer
*This project is built strictly for educational, informational, and research purposes. It does not provide SEBI-registered investment advice or buy/sell recommendations. Always conduct independent research before making financial decisions.*
