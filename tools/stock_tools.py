from typing import Dict, Any, Optional, List
from core.nse_client import NSEClient
from core.technicals import TechnicalAnalyzer
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore

class StockToolsRegistry:
    """
    Registry of high-level tools executed by the Agentic RAG engine.
    Integrates live NSE data, technical indicator analysis, dual-tier memory,
    and local semantic knowledge retrieval.
    """

    def __init__(self, nse: NSEClient, memory: DualTierMemoryManager, knowledge: StockKnowledgeStore):
        self.nse = nse
        self.memory = memory
        self.knowledge = knowledge

    def tool_get_live_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the live quote and price details for a given Indian stock symbol on NSE.
        Args:
            symbol: Stock symbol or company name (e.g. 'RELIANCE', 'TCS', 'HDFCBANK', 'Tata Motors').
        """
        # Entity resolution if pronoun or alias passed
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        quote = self.nse.get_live_quote(resolved)
        if "error" not in quote:
            self.memory.update_working_context(symbol=resolved)
        return quote

    def tool_get_stock_technicals(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """
        Computes key technical indicators (RSI, 20/50/200 EMAs, MACD, Bollinger Bands,
        Support/Resistance, and overall bias) from historical candles.
        Args:
            symbol: Stock symbol or company name (e.g. 'INFY', 'TCS', 'RELIANCE').
            period: Timeframe for analysis (e.g. '1mo', '3mo', '6mo', '1y').
        """
        resolved = self.memory.resolve_entity(symbol) or self.nse.resolve_symbol(symbol)
        self.memory.update_working_context(symbol=resolved)
        df = self.nse.get_historical_ohlcv(resolved, period=period)
        analysis = TechnicalAnalyzer.calculate_indicators(df)
        analysis["symbol"] = resolved
        return analysis

    def tool_search_company_knowledge(self, query: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs semantic vector retrieval over corporate announcements, business profile,
        earnings notes, and growth drivers for Indian companies.
        Args:
            query: The question or topics to search for (e.g. 'EV growth capex', 'cloud margin').
            symbol: Optional stock symbol filter (e.g. 'TATAMOTORS').
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
