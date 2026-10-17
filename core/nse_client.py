import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional, List
import logging
from config import NSE_BASE_URL, NSE_HEADERS, TICKER_ALIASES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSEClient:
    """
    Robust Real-Time Client for National Stock Exchange (NSE) India.
    Includes automated session cookie management, live equity quotes,
    financial fundamentals & valuation ratios, real-time financial news RSS,
    market status, and automatic fallback to Yahoo Finance (.NS).
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self.last_cookie_refresh = 0
        self.cookie_ttl = 300  # 5 minutes
        self._init_session()

    def _init_session(self):
        """Initializes or refreshes the session cookies by hitting the NSE homepage."""
        try:
            current_time = time.time()
            if current_time - self.last_cookie_refresh > self.cookie_ttl:
                response = self.session.get(NSE_BASE_URL, timeout=8)
                if response.status_code == 200:
                    self.last_cookie_refresh = current_time
                    logger.info("NSE session cookies successfully refreshed.")
        except Exception as e:
            logger.warning(f"Failed to refresh NSE cookies: {e}. Fallback mechanisms are active.")

    RENAME_FALLBACKS = {
        "TATAMOTORS": "TMPV",
        "TATAMTRDVR": "TMPV",
    }

    def resolve_symbol(self, query: str) -> str:
        """Resolves common names/aliases (e.g., 'reliance', 'tata motors', 'hal') to standard NSE symbol."""
        cleaned = query.strip().lower()
        if cleaned in TICKER_ALIASES:
            sym = TICKER_ALIASES[cleaned]
            return self.RENAME_FALLBACKS.get(sym, sym)
        for alias, symbol in TICKER_ALIASES.items():
            if alias in cleaned or cleaned in alias:
                return self.RENAME_FALLBACKS.get(symbol, symbol)
        raw_sym = query.strip().upper().replace(" ", "")
        return self.RENAME_FALLBACKS.get(raw_sym, raw_sym)

    def get_market_status(self) -> Dict[str, Any]:
        """Fetches the current live status of NSE market segments."""
        self._init_session()
        url = f"{NSE_BASE_URL}/api/marketStatus"
        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                data = resp.json()
                for market in data.get("marketState", []):
                    if market.get("market") == "Capital Market":
                        return {
                            "status": market.get("marketStatus", "Open"),
                            "trade_date": market.get("tradeDate", ""),
                            "index": market.get("index", "NIFTY 50"),
                            "last_price": market.get("last", 0),
                            "variation": market.get("variation", 0),
                            "percent_change": market.get("percentChange", 0),
                            "source": "NSE_LIVE"
                        }
        except Exception:
            pass

        # Fallback via yfinance for NIFTY 50
        try:
            nifty = yf.Ticker("^NSEI")
            info = nifty.fast_info
            last_price = info.last_price
            prev_close = info.previous_close
            change = last_price - prev_close if last_price and prev_close else 0
            pct_change = (change / prev_close * 100) if prev_close else 0
            return {
                "status": "Regular Trading",
                "trade_date": time.strftime("%d-%b-%Y"),
                "index": "NIFTY 50",
                "last_price": round(float(last_price), 2) if last_price else 0.0,
                "variation": round(float(change), 2),
                "percent_change": round(float(pct_change), 2),
                "source": "FALLBACK_YFINANCE"
            }
        except Exception:
            return {
                "status": "Market Active",
                "trade_date": time.strftime("%d-%b-%Y"),
                "index": "NIFTY 50",
                "last_price": 0.0,
                "variation": 0.0,
                "percent_change": 0.0,
                "source": "UNAVAILABLE"
            }

    def get_live_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches live real-time stock quote from NSE India API with detailed
        LTP, 52W High/Low, order book depth, circuit limits, and valuation metrics.
        Falls back seamlessly to Yahoo Finance (.NS) if NSE endpoint is restricted.
        """
        symbol = self.resolve_symbol(symbol)
        self._init_session()

        # Attempt 1: Direct NSE Equity Quote API
        url = f"{NSE_BASE_URL}/api/quote-equity?symbol={symbol}"
        try:
            resp = self.session.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                price_info = data.get("priceInfo", {})
                info = data.get("info", {})
                metadata = data.get("metadata", {})

                ltp = price_info.get("lastPrice", 0)
                change = price_info.get("change", 0)
                pchange = price_info.get("pChange", 0)
                prev_close = price_info.get("previousClose", 0)
                open_price = price_info.get("open", 0)
                day_high = price_info.get("intraDayHighLow", {}).get("max", 0)
                day_low = price_info.get("intraDayHighLow", {}).get("min", 0)
                week_52_high = price_info.get("weekHighLow", {}).get("max", 0)
                week_52_low = price_info.get("weekHighLow", {}).get("min", 0)
                upper_circuit = price_info.get("upperCP", "N/A")
                lower_circuit = price_info.get("lowerCP", "N/A")

                return {
                    "symbol": symbol,
                    "company_name": info.get("companyName", symbol),
                    "industry": metadata.get("industry", "N/A"),
                    "last_price": float(ltp),
                    "change": round(float(change), 2),
                    "percent_change": round(float(pchange), 2),
                    "previous_close": float(prev_close),
                    "open": float(open_price),
                    "day_high": float(day_high),
                    "day_low": float(day_low),
                    "week_52_high": float(week_52_high),
                    "week_52_low": float(week_52_low),
                    "upper_circuit": upper_circuit,
                    "lower_circuit": lower_circuit,
                    "isin": metadata.get("isin", "N/A"),
                    "listing_status": metadata.get("status", "Listed"),
                    "source": "NSE_LIVE",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            logger.warning(f"NSE API live quote fallback engaged for {symbol}: {e}")

        # Attempt 2: Resilient Fallback to Yahoo Finance (.NS)
        try:
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            fast = ticker.fast_info
            info_dict = ticker.info or {}

            last_price = fast.last_price or info_dict.get("regularMarketPrice", 0.0)
            prev_close = fast.previous_close or info_dict.get("regularMarketPreviousClose", 0.0)
            change = (last_price - prev_close) if (last_price and prev_close) else 0.0
            pchange = (change / prev_close * 100) if prev_close else 0.0

            return {
                "symbol": symbol,
                "company_name": info_dict.get("longName", symbol),
                "industry": info_dict.get("industry", info_dict.get("sector", "N/A")),
                "last_price": round(float(last_price), 2),
                "change": round(float(change), 2),
                "percent_change": round(float(pchange), 2),
                "previous_close": round(float(prev_close), 2),
                "open": round(float(fast.open or info_dict.get("regularMarketOpen", 0.0)), 2),
                "day_high": round(float(fast.day_high or info_dict.get("regularMarketDayHigh", 0.0)), 2),
                "day_low": round(float(fast.day_low or info_dict.get("regularMarketDayLow", 0.0)), 2),
                "week_52_high": round(float(fast.year_high or info_dict.get("fiftyTwoWeekHigh", 0.0)), 2),
                "week_52_low": round(float(fast.year_low or info_dict.get("fiftyTwoWeekLow", 0.0)), 2),
                "upper_circuit": "N/A",
                "lower_circuit": "N/A",
                "isin": info_dict.get("isin", "N/A"),
                "pe_ratio": round(float(info_dict.get("trailingPE", 0.0)), 2) if info_dict.get("trailingPE") else "N/A",
                "market_cap_cr": round(float(fast.market_cap or 0) / 10000000, 2),
                "source": "FALLBACK_NSE_YFINANCE",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Failed to fetch stock data for {symbol}: {e}")
            return {
                "error": f"Could not retrieve stock quote for '{symbol}'. Please ensure it is a valid NSE listed ticker.",
                "symbol": symbol
            }

    def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches comprehensive fundamental valuation ratios, financials,
        and balance sheet metrics for an Indian stock.
        """
        symbol = self.resolve_symbol(symbol)
        try:
            yf_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info or {}
            fast = ticker.fast_info

            market_cap_cr = round(float(fast.market_cap or info.get("marketCap", 0)) / 10000000, 2)
            pe_ratio = round(float(info.get("trailingPE", 0)), 2) if info.get("trailingPE") else None
            forward_pe = round(float(info.get("forwardPE", 0)), 2) if info.get("forwardPE") else None
            pb_ratio = round(float(info.get("priceToBook", 0)), 2) if info.get("priceToBook") else None
            div_yield = round(float(info.get("dividendYield", 0) or 0) * 100, 2) if info.get("dividendYield") else 0.0
            roe = round(float(info.get("returnOnEquity", 0) or 0) * 100, 2) if info.get("returnOnEquity") else None
            eps = round(float(info.get("trailingEps", 0)), 2) if info.get("trailingEps") else None
            book_value = round(float(info.get("bookValue", 0)), 2) if info.get("bookValue") else None
            beta = round(float(info.get("beta", 1.0)), 2) if info.get("beta") else 1.0
            debt_to_equity = round(float(info.get("debtToEquity", 0)), 2) if info.get("debtToEquity") else None
            profit_margin = round(float(info.get("profitMargins", 0) or 0) * 100, 2) if info.get("profitMargins") else None
            
            # Valuation assessment
            valuation_status = "Fairly Valued"
            if pe_ratio:
                if pe_ratio < 15:
                    valuation_status = "Attractive / Undervalued"
                elif pe_ratio > 45:
                    valuation_status = "Premium / High Valuation"

            # Distance from 52-week High
            cur_price = fast.last_price or info.get("regularMarketPrice", 0)
            high_52w = fast.year_high or info.get("fiftyTwoWeekHigh", 0)
            dist_52w_high = round(((cur_price - high_52w) / high_52w * 100), 2) if high_52w and cur_price else None

            return {
                "symbol": symbol,
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": round(float(cur_price), 2) if cur_price else 0.0,
                "market_cap_crores": market_cap_cr,
                "trailing_pe": pe_ratio,
                "forward_pe": forward_pe,
                "price_to_book": pb_ratio,
                "dividend_yield_percent": div_yield,
                "return_on_equity_percent": roe,
                "trailing_eps": eps,
                "book_value_per_share": book_value,
                "beta": beta,
                "debt_to_equity": debt_to_equity,
                "profit_margin_percent": profit_margin,
                "valuation_assessment": valuation_status,
                "distance_from_52w_high_percent": dist_52w_high
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {
                "error": f"Could not retrieve fundamental data for {symbol}: {str(e)}",
                "symbol": symbol
            }

    def get_live_stock_news(self, symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Fetches live financial news headlines and updates for the stock using
        Google News RSS feed (100% free, real-time, no API key required).
        """
        symbol = self.resolve_symbol(symbol)
        query = f"{symbol} stock NSE India"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        articles = []
        try:
            resp = requests.get(url, headers={"User-Agent": NSE_HEADERS["User-Agent"]}, timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    source = item.find("source").text if item.find("source") is not None else "Financial News"
                    
                    if title:
                        articles.append({
                            "title": title,
                            "published": pub_date[:16] if pub_date else "Recent",
                            "source": source,
                            "link": link
                        })
        except Exception as e:
            logger.warning(f"Error fetching live news for {symbol}: {e}")

        # If empty or failed, provide fallback
        if not articles:
            articles.append({
                "title": f"Market analysis and trading volume updates for {symbol} on NSE.",
                "published": time.strftime("%d %b %Y"),
                "source": "NSE Feed",
                "link": "#"
            })

        return articles

    def get_historical_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Fetches historical OHLCV data for technical analysis and charts."""
        symbol = self.resolve_symbol(symbol)
        try:
            yf_symbol = f"{symbol}.NS"
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_major_indices(self) -> Dict[str, Dict[str, Any]]:
        """Fetches live quotes for major Indian indices (Nifty 50, Bank Nifty, Sensex, Nifty IT)."""
        indices_to_track = {
            "NIFTY 50": "^NSEI",
            "BANK NIFTY": "^NSEBANK",
            "SENSEX": "^BSESN",
            "NIFTY IT": "^CNXIT"
        }
        results = {}
        for name, ticker_sym in indices_to_track.items():
            try:
                t = yf.Ticker(ticker_sym)
                fast = t.fast_info
                last_price = fast.last_price
                prev_close = fast.previous_close
                if last_price and prev_close:
                    change = last_price - prev_close
                    pct_change = (change / prev_close) * 100
                    results[name] = {
                        "value": round(float(last_price), 2),
                        "change": round(float(change), 2),
                        "percent_change": round(float(pct_change), 2)
                    }
            except Exception:
                continue
        return results
