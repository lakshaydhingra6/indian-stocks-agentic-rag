from typing import Dict, Any, Optional, List
from core.nse_client import NSEClient
from core.technicals import TechnicalAnalyzer
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore

class StockToolsRegistry:
    """
    Registry of high-level tools executed by the Agentic RAG engine.
    Integrates live NSE data, technical indicator analysis, fundamental valuation ratios,
    live financial news RSS, dual-tier memory, and local semantic knowledge retrieval.
    """

    def __init__(self, nse: NSEClient, memory: DualTierMemoryManager, knowledge: StockKnowledgeStore):
        self.nse = nse
        self.memory = memory
        self.knowledge = knowledge

    def tool_get_live_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the live quote and price details for a given Indian stock symbol on NSE.
        Args:
            symbol: Stock symbol or company name (e.g. 'RELIANCE', 'TCS', 'HDFCBANK', 'Tata Motors', 'HAL').
        """
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        quote = self.nse.get_live_quote(resolved)
        if "error" not in quote:
            self.memory.update_working_context(symbol=resolved)
        return quote

    def tool_get_stock_technicals(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """
        Computes key technical indicators (RSI, 20/50/200 EMAs, Golden/Death cross,
        MACD, Bollinger Bands, Pivot Points, and ATR stop-loss/target).
        Args:
            symbol: Stock symbol or company name (e.g. 'INFY', 'TCS', 'RELIANCE', 'IRFC').
            period: Timeframe for analysis (e.g. '1mo', '3mo', '6mo', '1y').
        """
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        self.memory.update_working_context(symbol=resolved)
        df = self.nse.get_historical_ohlcv(resolved, period=period)
        analysis = TechnicalAnalyzer.calculate_indicators(df)
        analysis["symbol"] = resolved
        return analysis

    def tool_get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches fundamental valuation metrics (P/E ratio, P/B ratio, Market Cap,
        Dividend Yield %, Return on Equity ROE %, Trailing EPS, Debt-to-Equity).
        Args:
            symbol: Stock symbol or company name (e.g. 'HDFCBANK', 'ITC', 'TATAMOTORS').
        """
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        self.memory.update_working_context(symbol=resolved)
        return self.nse.get_stock_fundamentals(resolved)

    def tool_get_live_stock_news(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches real-time financial news headlines and media updates for an Indian stock.
        Args:
            symbol: Target stock symbol or name (e.g. 'ZOMATO', 'RELIANCE', 'ADANIENT').
        """
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        self.memory.update_working_context(symbol=resolved)
        news_items = self.nse.get_live_stock_news(resolved)
        return {
            "symbol": resolved,
            "articles": news_items,
            "count": len(news_items)
        }

    def tool_compare_stocks(self, symbol1: str, symbol2: str) -> Dict[str, Any]:
        """
        Performs a head-to-head comparison of two Indian stocks across price,
        P/E valuation, 52-week performance, and RSI momentum.
        Args:
            symbol1: First stock (e.g. 'TCS')
            symbol2: Second stock (e.g. 'INFY')
        """
        s1 = self.memory.resolve_entity(symbol1) or self.nse.resolve_symbol(symbol1)
        s2 = self.memory.resolve_entity(symbol2) or self.nse.resolve_symbol(symbol2)
        
        q1 = self.nse.get_live_quote(s1)
        q2 = self.nse.get_live_quote(s2)
        f1 = self.nse.get_stock_fundamentals(s1)
        f2 = self.nse.get_stock_fundamentals(s2)
        
        return {
            "comparison": {
                s1: {
                    "name": q1.get("company_name", s1),
                    "ltp": q1.get("last_price"),
                    "percent_change": q1.get("percent_change"),
                    "pe_ratio": f1.get("trailing_pe"),
                    "market_cap_cr": f1.get("market_cap_crores"),
                    "dividend_yield": f1.get("dividend_yield_percent"),
                    "roe": f1.get("return_on_equity_percent")
                },
                s2: {
                    "name": q2.get("company_name", s2),
                    "ltp": q2.get("last_price"),
                    "percent_change": q2.get("percent_change"),
                    "pe_ratio": f2.get("trailing_pe"),
                    "market_cap_cr": f2.get("market_cap_crores"),
                    "dividend_yield": f2.get("dividend_yield_percent"),
                    "roe": f2.get("return_on_equity_percent")
                }
            }
        }

    def tool_search_company_knowledge(self, query: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs semantic vector retrieval over corporate announcements, business profiles,
        earnings notes, and growth drivers for Indian companies.
        Args:
            query: The question or topics to search for (e.g. 'defense order book', 'Blinkit growth').
            symbol: Optional stock symbol filter (e.g. 'HAL', 'ZOMATO').
        """
        if symbol:
            symbol = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        return self.knowledge.search(query=query, symbol=symbol, top_k=3)

    def tool_manage_watchlist_and_memory(self, action: str, symbol: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        """
        Manages the user's persistent watchlist, custom stock research notes, and preferences.
        Args:
            action: One of 'add_watchlist', 'remove_watchlist', 'view_watchlist', 'add_note', 'view_notes'.
            symbol: Target stock symbol (e.g. 'RELIANCE').
            note: Text content for notes (used with 'add_note' or 'add_watchlist').
        """
        action = action.lower()
        if symbol:
            symbol = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)

        if action == "view_watchlist":
            items = self.memory.get_watchlist()
            return {"action": "view_watchlist", "watchlist": items, "count": len(items)}

        if not symbol:
            return {"error": "A stock symbol is required for this action."}

        if action == "add_watchlist":
            quote = self.nse.get_live_quote(symbol)
            comp_name = quote.get("company_name", symbol)
            self.memory.add_to_watchlist(symbol=symbol, company_name=comp_name, notes=note or "")
            return {"status": "success", "message": f"Added {symbol} ({comp_name}) to your persistent watchlist."}

        elif action == "remove_watchlist":
            removed = self.memory.remove_from_watchlist(symbol)
            return {
                "status": "success" if removed else "not_found",
                "message": f"Removed {symbol} from watchlist." if removed else f"{symbol} was not found on your watchlist."
            }

        elif action == "add_note":
            if not note:
                return {"error": "Please provide a note text to save."}
            self.memory.add_stock_note(symbol=symbol, note=note)
            return {"status": "success", "message": f"Saved research note for {symbol}: '{note}'"}

        elif action == "view_notes":
            notes = self.memory.get_stock_notes(symbol)
            return {"symbol": symbol, "notes": notes, "count": len(notes)}

        return {"error": f"Unknown action '{action}'."}

    def tool_get_market_overview(self) -> Dict[str, Any]:
        """
        Returns live Indian market status and current levels for major indices
        (NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT).
        """
        status = self.nse.get_market_status()
        indices = self.nse.get_major_indices()
        return {
            "market_status": status,
            "indices": indices
        }
