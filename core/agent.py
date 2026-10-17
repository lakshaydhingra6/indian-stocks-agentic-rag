import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from google import genai
from google.genai import types

from core.nse_client import NSEClient
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore
from tools.stock_tools import StockToolsRegistry
from config import GEMINI_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are 'Artha-Agent', an expert AI Quantitative Research Agent and Financial Assistant specializing in the Indian Stock Market (NSE / BSE).
You have real-time access to live NSE market quotes, historical OHLCV data, technical indicators (RSI, 20/50/200 EMAs, Golden/Death crosses, Pivot points, ATR stop-loss), fundamental valuation ratios (P/E, P/B, Market Cap, ROE, Dividend Yield), real-time financial news, corporate disclosures, and a dual-tier memory system.

TOOL CAPABILITIES:
1. 'tool_get_live_quote': Real-time LTP, 52W high/low, day range, circuit limits.
2. 'tool_get_stock_technicals': RSI (14), EMA crossovers (Golden Cross/Death Cross), Pivot Points (S1/S2/R1/R2), ATR stop-loss and targets.
3. 'tool_get_stock_fundamentals': P/E, P/B, Market Cap (Cr), ROE %, Dividend Yield %, Trailing EPS, Debt-to-Equity.
4. 'tool_get_live_stock_news': Live news headlines and media updates.
5. 'tool_compare_stocks': Head-to-head comparison between two Indian stocks.
6. 'tool_search_company_knowledge': Corporate capex, business models, earnings call insights.
7. 'tool_manage_watchlist_and_memory': Add, remove, view persistent watchlist and notes in SQLite.
8. 'tool_get_market_overview': NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT live tracking.

GUIDELINES:
1. Always base answers on verified data fetched via your tools.
2. If the user refers to pronouns like 'it', 'the stock', or 'its price', use active working memory context to resolve the stock symbol.
3. Be structured, concise, and insightful. Highlight key numbers (₹, %, ratios) in bold.
4. Mandatory SEBI Disclaimer: Always append at the end:
   "> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
"""

class IndianStockRAGAgent:
    """
    Autonomous Agentic RAG Controller for Indian Stock Market.
    Supports tool execution, multi-turn reasoning, dual-tier memory integration,
    and transparent tool call tracing.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.nse = NSEClient()
        self.memory = DualTierMemoryManager()
        self.knowledge = StockKnowledgeStore()
        self.tools_registry = StockToolsRegistry(self.nse, self.memory, self.knowledge)
        
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key.strip())
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")

    def _get_tool_declarations(self) -> List[Any]:
        """Exposes functions to Gemini SDK."""
        return [
            self.tools_registry.tool_get_live_quote,
            self.tools_registry.tool_get_stock_technicals,
            self.tools_registry.tool_get_stock_fundamentals,
            self.tools_registry.tool_get_live_stock_news,
            self.tools_registry.tool_compare_stocks,
            self.tools_registry.tool_search_company_knowledge,
            self.tools_registry.tool_manage_watchlist_and_memory,
            self.tools_registry.tool_get_market_overview
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Executes the matching tool from the registry."""
        tool_map = {
            "tool_get_live_quote": self.tools_registry.tool_get_live_quote,
            "tool_get_stock_technicals": self.tools_registry.tool_get_stock_technicals,
            "tool_get_stock_fundamentals": self.tools_registry.tool_get_stock_fundamentals,
            "tool_get_live_stock_news": self.tools_registry.tool_get_live_stock_news,
            "tool_compare_stocks": self.tools_registry.tool_compare_stocks,
            "tool_search_company_knowledge": self.tools_registry.tool_search_company_knowledge,
            "tool_manage_watchlist_and_memory": self.tools_registry.tool_manage_watchlist_and_memory,
            "tool_get_market_overview": self.tools_registry.tool_get_market_overview,
        }
        fn = tool_map.get(name)
        if not fn:
            for k, v in tool_map.items():
                if k.endswith(name) or name.endswith(k):
                    fn = v
                    break

        if fn:
            try:
                return fn(**args)
            except Exception as e:
                return {"error": f"Error executing {name}: {str(e)}"}
        return {"error": f"Tool '{name}' not found."}

    def run(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes an agentic reasoning loop for the user query.
        Returns:
            Tuple of (final_response_markdown, list_of_tool_traces)
        """
        tool_traces = []

        if not self.client:
            return self._run_deterministic_agent(user_query)

        context_summary = self.memory.get_working_context_summary()
        preferences = self.memory.get_user_preferences()
        watchlist = self.memory.get_watchlist()
        
        pref_str = json.dumps(preferences) if preferences else "Standard / General"
        watchlist_str = ", ".join([w["symbol"] for w in watchlist[:8]]) if watchlist else "Empty"

        augmented_system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"[ACTIVE USER MEMORY]\n"
            f"- {context_summary}\n"
            f"- Watchlist Stocks: {watchlist_str}\n"
            f"- User Preferences: {pref_str}\n"
        )

        try:
            chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=augmented_system_prompt,
                    temperature=0.2,
                    tools=self._get_tool_declarations()
                )
            )

            response = chat.send_message(user_query)

            max_iterations = 6
            iteration = 0
            while response.function_calls and iteration < max_iterations:
                iteration += 1
                function_responses = []
                for call in response.function_calls:
                    call_name = call.name
                    call_args = call.args
                    tool_output = self._execute_tool(call_name, call_args)

                    tool_traces.append({
                        "tool": call_name,
                        "args": call_args,
                        "output": tool_output
                    })

                    function_responses.append(
                        types.Part.from_function_response(
                            name=call_name,
                            response={"result": tool_output}
                        )
                    )

                response = chat.send_message(function_responses)

            final_text = response.text or "I processed your request with live market data."
            
            symbols = [t["args"].get("symbol") for t in tool_traces if "symbol" in t.get("args", {}) and t["args"]["symbol"]]
            self.memory.record_interaction(user_query, final_text, [s for s in symbols if s])
            self.memory.update_working_context(user_msg=user_query, assistant_msg=final_text)

            return final_text, tool_traces

        except Exception as e:
            logger.error(f"Gemini API error during agent execution: {e}. Falling back to rule-based agent.")
            return self._run_deterministic_agent(user_query)

    def _run_deterministic_agent(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Intelligent deterministic agent that executes tools based on intent recognition.
        Guarantees 100% functionality even when no API key is set or offline.
        """
        tool_traces = []
        lower = user_query.lower()

        # 1. Market Overview Intent
        if any(w in lower for w in ["market status", "nifty", "sensex", "market today", "how is market", "indices"]):
            res = self.tools_registry.tool_get_market_overview()
            tool_traces.append({"tool": "tool_get_market_overview", "args": {}, "output": res})
            status = res.get("market_status", {})
            indices = res.get("indices", {})
            
            ind_text = "\n".join([f"- **{k}**: {v['value']} ({'+' if v['percent_change']>=0 else ''}{v['percent_change']}%)" for k, v in indices.items()])
            text = (
                f"### 📈 Live Indian Market Overview\n\n"
                f"**Market Status**: `{status.get('status', 'Open')}` (Date: {status.get('trade_date', 'Today')})\n\n"
                f"#### Major Indices:\n{ind_text}\n\n"
                f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
            )
            return text, tool_traces

        # 2. Watchlist View Intent
        if any(w in lower for w in ["my watchlist", "view watchlist", "show watchlist"]):
            res = self.tools_registry.tool_manage_watchlist_and_memory("view_watchlist")
            tool_traces.append({"tool": "tool_manage_watchlist_and_memory", "args": {"action": "view_watchlist"}, "output": res})
            items = res.get("watchlist", [])
            if not items:
                text = "Your watchlist is currently empty. You can add stocks by asking e.g. *'Add Reliance to my watchlist'*."
            else:
                table_lines = ["| Symbol | Company | Added On | Notes |", "|---|---|---|---|"]
                for i in items:
                    table_lines.append(f"| **{i['symbol']}** | {i['company_name']} | {i['added_at'][:10]} | {i['notes'] or '-'} |")
                text = "### 📋 Your Persistent Watchlist\n\n" + "\n".join(table_lines)
            return text, tool_traces

        # 3. Stock Comparison Intent (e.g., "Compare TCS and Infosys" or "Compare it with TCS")
        if "compare" in lower and ("with" in lower or "and" in lower or "vs" in lower):
            stop_words = {"compare", "between", "and", "with", "vs", "versus", "in", "terms", "of", "the", "stock", "both", "two", "to", "price", "valuation"}
            words = [w.strip().lower() for w in user_query.replace(",", " ").replace("?", "").split() if w.strip().lower() not in stop_words]
            found_symbols = []
            for w in words:
                from config import TICKER_ALIASES
                if w in TICKER_ALIASES:
                    sym = TICKER_ALIASES[w]
                    if sym not in found_symbols:
                        found_symbols.append(sym)
                elif w.upper() in TICKER_ALIASES.values():
                    sym = w.upper()
                    if sym not in found_symbols:
                        found_symbols.append(sym)
                else:
                    ent = self.memory.resolve_entity(w)
                    if ent and ent not in found_symbols:
                        found_symbols.append(ent)
            
            if len(found_symbols) < 2 and self.memory.last_mentioned_symbol:
                if self.memory.last_mentioned_symbol not in found_symbols:
                    found_symbols.insert(0, self.memory.last_mentioned_symbol)

            if len(found_symbols) >= 2:
                s1, s2 = found_symbols[0], found_symbols[1]
                res = self.tools_registry.tool_compare_stocks(s1, s2)
                tool_traces.append({"tool": "tool_compare_stocks", "args": {"symbol1": s1, "symbol2": s2}, "output": res})
                
                c_data = res.get("comparison", {})
                d1 = c_data.get(s1, {})
                d2 = c_data.get(s2, {})
                
                ltp1 = float(d1.get('ltp') or 0.0)
                ltp2 = float(d2.get('ltp') or 0.0)
                pchg1 = float(d1.get('percent_change') or 0.0)
                pchg2 = float(d2.get('percent_change') or 0.0)
                mcap1 = float(d1.get('market_cap_cr') or 0.0)
                mcap2 = float(d2.get('market_cap_cr') or 0.0)
                
                text = (
                    f"### ⚖️ Head-to-Head Comparison: {s1} vs {s2}\n\n"
                    f"| Metric | **{s1}** ({d1.get('name', s1)}) | **{s2}** ({d2.get('name', s2)}) |\n"
                    f"|---|---|---|\n"
                    f"| **Last Traded Price (LTP)** | ₹{ltp1:,.2f} ({pchg1:+.2f}%) | ₹{ltp2:,.2f} ({pchg2:+.2f}%) |\n"
                    f"| **P/E Ratio (Valuation)** | {d1.get('pe_ratio') or 'N/A'} | {d2.get('pe_ratio') or 'N/A'} |\n"
                    f"| **Market Cap (₹ Cr)** | ₹{mcap1:,.0f} Cr | ₹{mcap2:,.0f} Cr |\n"
                    f"| **Dividend Yield** | {d1.get('dividend_yield', 0)}% | {d2.get('dividend_yield', 0)}% |\n"
                    f"| **Return on Equity (ROE)** | {d1.get('roe', 0)}% | {d2.get('roe', 0)}% |\n\n"
                    f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
                )
                self.memory.update_working_context(symbol=s2, user_msg=user_query, assistant_msg=text)
                return text, tool_traces

        # Resolve primary symbol
        symbol = self.memory.resolve_entity(user_query) or self.nse.resolve_symbol(user_query)

        # 4. Add to Watchlist Intent
        if any(w in lower for w in ["add to watchlist", "watchlist add", "track "]):
            res = self.tools_registry.tool_manage_watchlist_and_memory("add_watchlist", symbol=symbol)
            tool_traces.append({"tool": "tool_manage_watchlist_and_memory", "args": {"action": "add_watchlist", "symbol": symbol}, "output": res})
            return f"✅ **{symbol}** has been added to your persistent watchlist in SQLite memory.", tool_traces

        # 5. News & Media Buzz Intent
        if any(w in lower for w in ["news", "headline", "sentiment", "latest news", "buzz"]):
            news_res = self.tools_registry.tool_get_live_stock_news(symbol)
            tool_traces.append({"tool": "tool_get_live_stock_news", "args": {"symbol": symbol}, "output": news_res})
            articles = news_res.get("articles", [])
            lines = [f"- **{a['title']}** ({a['source']} • *{a['published']}*)" for a in articles]
            text = (
                f"### 📰 Real-Time News & Headlines for {symbol}\n\n"
                + "\n".join(lines) + "\n\n"
                f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
            )
            return text, tool_traces

        # 6. Fundamental & Valuation Intent
        if any(w in lower for w in ["fundamental", "pe ratio", "valuation", "market cap", "roe", "dividend", "book value", "balance sheet", "eps"]):
            fund_res = self.tools_registry.tool_get_stock_fundamentals(symbol)
            tool_traces.append({"tool": "tool_get_stock_fundamentals", "args": {"symbol": symbol}, "output": fund_res})
            
            pe = fund_res.get("trailing_pe", "N/A")
            pb = fund_res.get("price_to_book", "N/A")
            mcap = fund_res.get("market_cap_crores", 0)
            roe = fund_res.get("return_on_equity_percent", "N/A")
            div = fund_res.get("dividend_yield_percent", 0.0)
            eps = fund_res.get("trailing_eps", "N/A")
            status_val = fund_res.get("valuation_assessment", "Fairly Valued")
            dist_52w = fund_res.get("distance_from_52w_high_percent")

            text = (
                f"### 💰 Fundamental & Valuation Profile: {fund_res.get('company_name', symbol)} ({symbol})\n\n"
                f"- **Sector & Industry**: {fund_res.get('sector', 'N/A')} ({fund_res.get('industry', 'N/A')})\n"
                f"- **Market Capitalization**: **₹{mcap:,.0f} Crores**\n"
                f"- **Trailing P/E Ratio**: **{pe}** (Valuation Stance: `{status_val}`)\n"
                f"- **Price to Book (P/B)**: **{pb}** | **Book Value/Share**: ₹{fund_res.get('book_value_per_share', 'N/A')}\n"
                f"- **Return on Equity (ROE)**: **{roe}%**\n"
                f"- **Dividend Yield**: **{div}%**\n"
                f"- **Trailing EPS**: ₹{eps}\n"
                f"- **52-Week High Distance**: {f'{dist_52w}%' if dist_52w else 'N/A'}\n\n"
                f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
            )
            return text, tool_traces

        # 7. Default: Comprehensive Live Quote + Technicals + Trade Setup + Knowledge
        quote = self.tools_registry.tool_get_live_quote(symbol)
        tool_traces.append({"tool": "tool_get_live_quote", "args": {"symbol": symbol}, "output": quote})

        if "error" in quote:
            return f"⚠️ {quote['error']}\n\nTip: You can query 100+ stocks like **RELIANCE**, **TCS**, **INFY**, **TATAMOTORS**, **HAL**, **IRFC**, **ZOMATO**, **TRENT**, **HDFCBANK**.", tool_traces

        technicals = self.tools_registry.tool_get_stock_technicals(symbol)
        tool_traces.append({"tool": "tool_get_stock_technicals", "args": {"symbol": symbol}, "output": technicals})

        knowledge = self.tools_registry.tool_search_company_knowledge(query=user_query, symbol=symbol)
        tool_traces.append({"tool": "tool_search_company_knowledge", "args": {"query": user_query, "symbol": symbol}, "output": knowledge})

        ltp = quote.get("last_price", 0)
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)
        direction = "🟢" if pchg >= 0 else "🔴"
        sign = "+" if pchg >= 0 else ""

        rsi = technicals.get("rsi_14", "N/A")
        rsi_cond = technicals.get("rsi_condition", "Neutral")
        bias = technicals.get("overall_bias", "Neutral")
        ema200 = technicals.get("ema_200", "N/A")
        ma_cross = technicals.get("ma_crossover_status", "Neutral")
        pivot = technicals.get("pivot_point", "N/A")
        r1 = technicals.get("resistance_1", "N/A")
        s1 = technicals.get("support_1", "N/A")
        sl = technicals.get("suggested_stop_loss", "N/A")
        tgt = technicals.get("suggested_target", "N/A")

        k_snippet = ""
        if knowledge:
            k_snippet = f"\n\n**Company Business Insights**:\n- {knowledge[0].get('content', '')}"

        text = (
            f"### {direction} {quote.get('company_name', symbol)} ({symbol}) - Live NSE Analysis\n\n"
            f"- **Last Traded Price (LTP)**: ₹{ltp:,.2f} ({sign}{chg:.2f} | {sign}{pchg:.2f}%)\n"
            f"- **Day Range**: Low ₹{quote.get('day_low', 0):,.2f} — High ₹{quote.get('day_high', 0):,.2f}\n"
            f"- **52-Week Range**: Low ₹{quote.get('week_52_low', 0):,.2f} — High ₹{quote.get('week_52_high', 0):,.2f}\n"
            f"- **Data Source**: `{quote.get('source', 'NSE')}`\n\n"
            f"#### 📊 Quantitative & Technical Signals:\n"
            f"- **Overall Stance**: `{bias}` | **MA Trend**: `{ma_cross}`\n"
            f"- **RSI (14-Day)**: `{rsi}` — *{rsi_cond}*\n"
            f"- **200-Day EMA Benchmark**: ₹{ema200} ({'Above' if technicals.get('price_above_ema200') else 'Below'})\n"
            f"- **Pivot Levels**: Pivot ₹{pivot} | Support (S1) **₹{s1}** | Resistance (R1) **₹{r1}**\n"
            f"- **Calculated ATR Setup**: Stop-Loss: **₹{sl}** | Target: **₹{tgt}**"
            f"{k_snippet}\n\n"
            f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
        )

        self.memory.update_working_context(symbol=symbol, user_msg=user_query, assistant_msg=text)
        return text, tool_traces
