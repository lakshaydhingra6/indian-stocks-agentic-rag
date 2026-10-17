import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database
DB_PATH = DATA_DIR / "memory.db"

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# NSE Scraper / Request Headers
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive"
}

# Major NSE Indices
POPULAR_INDICES = {
    "NIFTY 50": "NIFTY 50",
    "NIFTY BANK": "NIFTY BANK",
    "NIFTY IT": "NIFTY IT",
    "NIFTY AUTO": "NIFTY AUTO",
    "NIFTY PHARMA": "NIFTY PHARMA"
}

# Popular Stock Symbol Mapping (Aliases to official NSE symbols)
TICKER_ALIASES = {
    "reliance": "RELIANCE",
    "ril": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc bank": "HDFCBANK",
    "hdfc": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "state bank of india": "SBIN",
    "sbi": "SBIN",
    "tata motors": "TATAMOTORS",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "itc": "ITC",
    "l&t": "LT",
    "larson and toubro": "LT",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "maruti": "MARUTI",
    "maruti suzuki": "MARUTI",
    "bajaj finance": "BAJFINANCE",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH",
    "adani enterprises": "ADANIENT",
    "adani ports": "ADANIPORTS",
    "tata steel": "TATASTEEL",
    "sun pharma": "SUNPHARMA",
    "kotak": "KOTAKBANK",
    "kotak mahindra": "KOTAKBANK",
    "zomato": "ZOMATO",
    "paytm": "PAYTM",
    "jio financial": "JIOFIN"
}
