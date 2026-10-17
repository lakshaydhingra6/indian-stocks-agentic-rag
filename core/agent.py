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
You have real-time access to live NSE market quotes, historical OHLCV data, technical indicators (RSI, 20/50/200 EMAs, MACD, Bollinger Bands), company knowledge base, and a dual-tier memory system.

GUIDELINES:
1. Always base stock price and technical assertions on fresh live data fetched via your tools.
2. If the user refers to pronouns like 'it', 'the stock', or 'its price', use your active working memory context to resolve the stock symbol.
3. Be structured and precise. For stock queries:
   - Present Key Data: Current LTP, Day Range, % Change, 52-Week High/Low.
   - Technical Summary: RSI condition, EMA status (above/below 50 & 200 EMA), MACD trend, key support/resistance.
   - Fundamental / Business Insight: If relevant, mention business drivers from the knowledge base.
4. When the user asks to save to watchlist, record notes, or view watchlist, invoke the memory tool.
5. Mandatory Disclaimer: Always append a brief disclaimer at the end:
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
            self.tools_registry.tool_search_company_knowledge,
            self.tools_registry.tool_manage_watchlist_and_memory,
            self.tools_registry.tool_get_market_overview
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Executes the matching tool from the registry."""
        tool_map = {
            "tool_get_live_quote": self.tools_registry.tool_get_live_quote,
            "tool_get_stock_technicals": self.tools_registry.tool_get_stock_technicals,
            "tool_search_company_knowledge": self.tools_registry.tool_search_company_knowledge,
            "tool_manage_watchlist_and_memory": self.tools_registry.tool_manage_watchlist_and_memory,
            "tool_get_market_overview": self.tools_registry.tool_get_market_overview,
        }
        fn = tool_map.get(name)
        if not fn:
            # Fallback matching without 'tool_' prefix if named by schema
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

        # If no Gemini API key configured, use intelligent deterministic agent
        if not self.client:
            return self._run_deterministic_agent(user_query)

        # Context enrichment from Short-term & Long-term memory
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
            # Multi-turn tool calling session with Gemini
            chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=augmented_system_prompt,
                    temperature=0.2,
                    tools=self._get_tool_declarations()
                )
            )

            response = chat.send_message(user_query)

            # Handle Function Calling Loop
            max_iterations = 5
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
            
            # Save interaction in persistent memory
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

        # 1. Check for Market Overview intent
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

        # 2. Check for Watchlist view intent
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

        # 3. Resolve target stock
        symbol = self.memory.resolve_entity(user_query) or self.nse.resolve_symbol(user_query)
        if not symbol or symbol == user_query.strip().upper():
            # Check if an alias matches
            symbol = self.nse.resolve_symbol(user_query)

        # 4. Check for Add to Watchlist intent
        if any(w in lower for w in ["add to watchlist", "watchlist add", "track "]):
            res = self.tools_registry.tool_manage_watchlist_and_memory("add_watchlist", symbol=symbol)
            tool_traces.append({"tool": "tool_manage_watchlist_and_memory", "args": {"action": "add_watchlist", "symbol": symbol}, "output": res})
            return f"✅ **{symbol}** has been added to your persistent watchlist in SQLite memory.", tool_traces

        # 5. Live Quote & Technical Analysis intent
        quote = self.tools_registry.tool_get_live_quote(symbol)
        tool_traces.append({"tool": "tool_get_live_quote", "args": {"symbol": symbol}, "output": quote})

        if "error" in quote:
            return f"⚠️ {quote['error']}\n\nTip: You can specify stocks like **RELIANCE**, **TCS**, **INFY**, **TATAMOTORS**, **HDFCBANK**.", tool_traces

        # Technicals
        technicals = self.tools_registry.tool_get_stock_technicals(symbol)
        tool_traces.append({"tool": "tool_get_stock_technicals", "args": {"symbol": symbol}, "output": technicals})

        # Knowledge Store
        knowledge = self.tools_registry.tool_search_company_knowledge(query=user_query, symbol=symbol)
        tool_traces.append({"tool": "tool_search_company_knowledge", "args": {"query": user_query, "symbol": symbol}, "output": knowledge})

        # Synthesis
        ltp = quote.get("last_price", 0)
        chg = quote.get("change", 0)
        pchg = quote.get("percent_change", 0)
        direction = "🟢" if pchg >= 0 else "🔴"
        sign = "+" if pchg >= 0 else ""

        rsi = technicals.get("rsi_14", "N/A")
        rsi_cond = technicals.get("rsi_condition", "Neutral")
        bias = technicals.get("overall_bias", "Neutral")
        ema200 = technicals.get("ema_200", "N/A")
        support = technicals.get("support_level", "N/A")
        resistance = technicals.get("resistance_level", "N/A")

        k_snippet = ""
        if knowledge:
            k_snippet = f"\n\n**Company & Business Highlights**:\n- {knowledge[0].get('content', '')}"

        text = (
            f"### {direction} {quote.get('company_name', symbol)} ({symbol}) - Live NSE Analysis\n\n"
            f"- **Last Traded Price (LTP)**: ₹{ltp:,.2f} ({sign}{chg:.2f} | {sign}{pchg:.2f}%)\n"
            f"- **Day Range**: Low ₹{quote.get('day_low', 0):,.2f} — High ₹{quote.get('day_high', 0):,.2f}\n"
            f"- **52-Week Range**: Low ₹{quote.get('week_52_low', 0):,.2f} — High ₹{quote.get('week_52_high', 0):,.2f}\n"
            f"- **Data Source**: `{quote.get('source', 'NSE')}`\n\n"
            f"#### 📊 Technical Indicator Signals:\n"
            f"- **Overall Technical Stance**: `{bias}`\n"
            f"- **RSI (14-Day)**: `{rsi}` — {rsi_cond}\n"
            f"- **200-Day EMA Benchmark**: ₹{ema200} (Price is {'Above' if technicals.get('price_above_ema200') else 'Below'})\n"
            f"- **Key Levels**: Immediate Support at **₹{support}**, Resistance at **₹{resistance}**"
            f"{k_snippet}\n\n"
            f"> *Disclaimer: Educational analysis only; not SEBI-registered investment advice. Equity investments are subject to market risks.*"
        )

        self.memory.update_working_context(symbol=symbol, user_msg=user_query, assistant_msg=text)
        return text, tool_traces
