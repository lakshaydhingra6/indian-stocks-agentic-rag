import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DB_PATH, TICKER_ALIASES

class DualTierMemoryManager:
    """
    Manages both Short-Term Working Memory (in-memory context, entity resolution)
    and Long-Term Persistent Memory (SQLite for watchlist, portfolio preferences, notes).
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # Working Memory (Short-term context)
        self.last_mentioned_symbol: Optional[str] = None
        self.recent_symbols: List[str] = []
        self.conversation_buffer: List[Dict[str, str]] = []

        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """Initializes tables for long-term memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Watchlist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    target_price REAL
                )
            """)

            # 2. Stock Notes & Insights
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 3. User Preferences (e.g., risk appetite, investment horizon)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    pref_key TEXT PRIMARY KEY,
                    pref_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Interaction History across sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT,
                    agent_response TEXT,
                    mentioned_symbols TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # --- Short-Term Working Memory Operations ---

    def update_working_context(self, symbol: Optional[str] = None, user_msg: str = "", assistant_msg: str = ""):
        """Updates short-term in-session context."""
        if symbol:
            symbol = symbol.upper()
            self.last_mentioned_symbol = symbol
            if symbol not in self.recent_symbols:
                self.recent_symbols.append(symbol)
                if len(self.recent_symbols) > 5:
                    self.recent_symbols.pop(0)

        if user_msg or assistant_msg:
            self.conversation_buffer.append({
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": time.strftime("%H:%M:%S")
            })
            if len(self.conversation_buffer) > 10:
                self.conversation_buffer.pop(0)

    def resolve_entity(self, text: str) -> Optional[str]:
        """
        Resolves pronouns like 'it', 'this company', 'the stock'
        or stock names to the active stock symbol.
        """
        lower = text.lower()
        # Direct ticker aliases check
        for alias, sym in TICKER_ALIASES.items():
            if alias in lower:
                self.update_working_context(symbol=sym)
                return sym

        # Contextual pronouns
        pronoun_triggers = ["it", "this stock", "this company", "that stock", "the stock", "same stock", "its"]
        words = lower.split()
        if any(p in words or p in lower for p in pronoun_triggers):
            if self.last_mentioned_symbol:
                return self.last_mentioned_symbol

        return None

    def get_working_context_summary(self) -> str:
        """Returns a concise summary of the active working memory for prompt conditioning."""
        recent_stocks_str = ", ".join(self.recent_symbols) if self.recent_symbols else "None"
        last_str = self.last_mentioned_symbol or "None"
        return f"Active Stock in Context: {last_str} | Recent Stocks Discussed: {recent_stocks_str}"

    # --- Long-Term Persistent Memory Operations ---

    def add_to_watchlist(self, symbol: str, company_name: str = "", notes: str = "", target_price: Optional[float] = None) -> bool:
        """Adds or updates a stock in the persistent watchlist."""
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO watchlist (symbol, company_name, notes, target_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name = COALESCE(NULLIF(excluded.company_name, ''), watchlist.company_name),
                    notes = COALESCE(NULLIF(excluded.notes, ''), watchlist.notes),
                    target_price = COALESCE(excluded.target_price, watchlist.target_price)
            """, (symbol, company_name or symbol, notes, target_price))
            conn.commit()
            return True

    def remove_from_watchlist(self, symbol: str) -> bool:
        """Removes a stock from the watchlist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
            conn.commit()
            return cursor.rowcount > 0

    def get_watchlist(self) -> List[Dict[str, Any]]:
        """Retrieves all stocks from the user's persistent watchlist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, company_name, added_at, notes, target_price FROM watchlist ORDER BY added_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_stock_note(self, symbol: str, note: str) -> bool:
        """Saves a permanent research note/insight for a stock."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO stock_notes (symbol, note) VALUES (?, ?)", (symbol.upper(), note))
            conn.commit()
            return True

    def get_stock_notes(self, symbol: str) -> List[Dict[str, Any]]:
        """Retrieves stored research notes for a stock."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, note, created_at FROM stock_notes WHERE symbol = ? ORDER BY created_at DESC", (symbol.upper(),))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def set_user_preference(self, key: str, value: str):
        """Stores a persistent user preference (e.g. risk_appetite, horizon)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_preferences (pref_key, pref_value)
                VALUES (?, ?)
                ON CONFLICT(pref_key) DO UPDATE SET
                    pref_value = excluded.pref_value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value))
            conn.commit()

    def get_user_preferences(self) -> Dict[str, str]:
        """Gets all persistent user preferences."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pref_key, pref_value FROM user_preferences")
            rows = cursor.fetchall()
            return {row["pref_key"]: row["pref_value"] for row in rows}

    def record_interaction(self, query: str, response: str, symbols: List[str]):
        """Persists an interaction in history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interaction_history (user_query, agent_response, mentioned_symbols)
                VALUES (?, ?, ?)
            """, (query, response, json.dumps(symbols)))
            conn.commit()

    def get_recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves recent user interactions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_query, agent_response, mentioned_symbols, created_at
                FROM interaction_history ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
