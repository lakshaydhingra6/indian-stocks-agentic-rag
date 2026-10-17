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
    "NIFTY PHARMA": "NIFTY PHARMA",
    "NIFTY FMCG": "NIFTY FMCG",
    "NIFTY METAL": "NIFTY METAL"
}

# Comprehensive Indian Stock Symbol Mapping (100+ Top Equities)
TICKER_ALIASES = {
    # NIFTY 50 / Heavyweights
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
    "larsen and toubro": "LT",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "maruti": "MARUTI",
    "maruti suzuki": "MARUTI",
    "bajaj finance": "BAJFINANCE",
    "bajfinance": "BAJFINANCE",
    "bajaj finserv": "BAJAJFINSV",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH",
    "tech mahindra": "TECHM",
    "ltimindtree": "LTIM",
    "adani enterprises": "ADANIENT",
    "adani ports": "ADANIPORTS",
    "adani power": "ADANIPOWER",
    "adani green": "ADANIGREEN",
    "tata steel": "TATASTEEL",
    "jsw steel": "JSWSTEEL",
    "sun pharma": "SUNPHARMA",
    "cipla": "CIPLA",
    "dr reddy": "DRREDDY",
    "kotak": "KOTAKBANK",
    "kotak mahindra": "KOTAKBANK",
    "axis bank": "AXISBANK",
    "indusind bank": "INDUSINDBK",
    "titan": "TITAN",
    "asian paints": "ASIANPAINT",
    "ultratech cement": "ULTRACEMCO",
    "ntpc": "NTPC",
    "power grid": "POWERGRID",
    "coal india": "COALINDIA",
    "ongc": "ONGC",
    "bpcl": "BPCL",
    "eicher motors": "EICHERMOT",
    "hero motocorp": "HEROMOTOCO",
    "bajaj auto": "BAJAJ-AUTO",
    "mahindra and mahindra": "M&M",
    "m&m": "M&M",
    "nestle": "NESTLEIND",
    "britannia": "BRITANNIA",
    "tata consumer": "TATACONSUM",

    # Defense & Aerospace
    "hal": "HAL",
    "hindustan aeronautics": "HAL",
    "bel": "BEL",
    "bharat electronics": "BEL",
    "bdl": "BDL",
    "bharat dynamics": "BDL",
    "mazagon dock": "MAZDOCK",
    "cochin shipyard": "COCHINSHIP",

    # Railways & Public Sector (PSU)
    "irfc": "IRFC",
    "indian railway finance": "IRFC",
    "rvnl": "RVNL",
    "rail vikas nigam": "RVNL",
    "irctc": "IRCTC",
    "ircon": "IRCON",
    "bhel": "BHEL",
    "sail": "SAIL",
    "canara bank": "CANBK",
    "bank of baroda": "BANKBARODA",
    "pnb": "PNB",
    "punjab national bank": "PNB",
    "union bank": "UNIONBANK",
    "rec": "RECLTD",
    "pfc": "PFC",
    "power finance": "PFC",

    # Retail, FMCG & Consumer Tech
    "trent": "TRENT",
    "zomato": "ZOMATO",
    "swiggy": "SWIGGY",
    "nykaa": "NYKAA",
    "paytm": "PAYTM",
    "one97": "PAYTM",
    "jio financial": "JIOFIN",
    "dmart": "DMART",
    "avenue supermarts": "DMART",
    "varun beverages": "VBL",
    "vbl": "VBL",
    "cdsl": "CDSL",
    "bse": "BSE",
    "angel one": "ANGELONE",
    "tata power": "TATAPOWER",
    "suzlon": "SUZLON",
    "vedanta": "VEDL"
}
