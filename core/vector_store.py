import json
import math
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DATA_DIR

class StockKnowledgeStore:
    """
    Lightweight, local semantic vector knowledge store for Indian stock news,
    quarterly earnings summaries, business descriptions, and corporate announcements.
    Operates 100% locally with zero cloud dependencies or costs.
    """

    def __init__(self, storage_path: Path = DATA_DIR / "knowledge_store.json"):
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self._load_or_seed()

    def _load_or_seed(self):
        """Loads existing knowledge base or seeds with comprehensive Indian stocks insights."""
        # Pre-seed rich knowledge for top Indian companies across all sectors
        default_docs = [
            {
                "symbol": "RELIANCE",
                "title": "Reliance Industries Business Overview & Green Energy Capex",
                "content": "Reliance Industries Limited (RIL) is India's largest conglomerate with core operations in Oil-to-Chemicals (O2C), Telecommunications (Jio Infocomm), and Retail (Reliance Retail). RIL is aggressively investing in New Energy (gigafactories in Jamnagar for solar PV, battery storage, and green hydrogen). Jio leads the 5G rollout with over 470M subscribers, while Retail continues omnichannel expansion.",
                "category": "Company Profile & Capex"
            },
            {
                "symbol": "TCS",
                "title": "Tata Consultancy Services (TCS) AI & Cloud Pipeline",
                "content": "Tata Consultancy Services is India's largest IT services exporter. TCS focuses on Enterprise AI transformations, generative AI pipelines, and cloud migrations. Key verticals include BFSI (30%+ of revenue), Retail & CPG, and Communications. TCS maintains industry-leading operating margins (around 24-26%) and high return on equity.",
                "category": "Earnings & Business"
            },
            {
                "symbol": "INFY",
                "title": "Infosys AI-First Platform & Digital Revenue",
                "content": "Infosys is the second-largest Indian IT provider, pioneering its 'Topaz' generative AI suite. Infosys derives over 60% of revenue from digital transformations, financial services, and energy. It regularly conducts buybacks and distributes 85%+ of free cash flow via dividends.",
                "category": "Earnings & Business"
            },
            {
                "symbol": "HDFCBANK",
                "title": "HDFC Bank Post-Merger Strategy & Branch Expansion",
                "content": "HDFC Bank is India's largest private sector bank following its historic merger with parent HDFC Limited. The bank is prioritizing deposit mobilization to optimize its Credit-Deposit (CD) ratio, expanding rural/semi-urban branch presence, and leveraging digital lending.",
                "category": "Banking & Financials"
            },
            {
                "symbol": "ICICIBANK",
                "title": "ICICI Bank Digital Growth & Return Ratios",
                "content": "ICICI Bank is known for robust asset quality, industry-low gross NPAs (<2.2%), and superior return on assets (RoA > 2.2%). Key growth drivers include the 'iMobile Pay' app and high-yielding retail, SME, and business banking segments.",
                "category": "Banking & Financials"
            },
            {
                "symbol": "SBIN",
                "title": "State Bank of India (SBI) Credit Growth & YONO Dominance",
                "content": "State Bank of India is the nation's premier public sector lender, accounting for over one-fifth of India's banking assets. Strong retail home loans, robust corporate credit underwriting, and digital customer onboarding via the YONO mobile platform drive record net interest margins (NIM).",
                "category": "PSU Banking"
            },
            {
                "symbol": "TMPV",
                "title": "Tata Motors (Passenger Vehicles & EV) Leadership & JLR",
                "content": "Tata Motors Passenger Vehicles (TMPV) dominates the Indian Electric Vehicle (EV) passenger car segment with Nexon EV, Punch EV, and Curvv EV, commanding >65% market share. Its luxury subsidiary Jaguar Land Rover (JLR) delivers strong free cash flows driven by Defender and Range Rover order books. Following the 2026 demerger, passenger vehicles trade under TMPV and commercial vehicles under TMCV.",
                "category": "Automobile & EV"
            },
            {
                "symbol": "ITC",
                "title": "ITC Hotels Demerger & FMCG Scaling",
                "content": "ITC is an Indian conglomerate spanning Cigarettes, Paperboards, Agribusiness, FMCG, and Hotels. The company completed the demerger of ITC Hotels to unlock shareholder value. FMCG brands (Aashirvaad, Sunfeast, Bingo) are driving non-cigarette revenue growth with expanding operating margins.",
                "category": "FMCG & Diversified"
            },
            {
                "symbol": "BHARTIARTL",
                "title": "Bharti Airtel ARPU Growth & 5G Rollout",
                "content": "Bharti Airtel is a premier telecommunications provider in India and Africa. Airtel boasts the highest ARPU (Average Revenue Per User) in the Indian telecom sector (> ₹205), expanding home broadband (Xstream Fiber) and Airtel Business enterprise data solutions.",
                "category": "Telecom"
            },
            {
                "symbol": "HAL",
                "title": "Hindustan Aeronautics Limited (HAL) Defense Order Book",
                "content": "HAL is India's sole aerospace manufacturing defense PSU, producing Tejas Light Combat Aircraft (LCA Mk1A), Prachand Light Combat Helicopters (LCH), and Sukhoi Su-30 upgrades. HAL holds a robust multi-year order book of over ₹90,000 Crores driven by India's indigenization and Make-in-India defense policies.",
                "category": "Defense & Aerospace"
            },
            {
                "symbol": "BEL",
                "title": "Bharat Electronics Limited (BEL) Defense Electronics",
                "content": "Bharat Electronics is India's leading defense electronics PSU, supplying advanced radars, missile guidance systems, naval sonars, electronic warfare suites, and civilian EVMs. BEL maintains debt-free balance sheet, high Return on Capital Employed (ROCE > 25%), and strong order inflow from armed forces.",
                "category": "Defense & Electronics"
            },
            {
                "symbol": "IRFC",
                "title": "Indian Railway Finance Corporation (IRFC) Lending Pipeline",
                "content": "IRFC is the dedicated market borrowing arm of the Indian Railways under the Ministry of Railways. It finances rolling stock assets (locomotives, coaches) and railway infrastructure projects. It boasts 100% sovereign-backed exposure with virtually zero Non-Performing Assets (NPAs).",
                "category": "Railways & PSU"
            },
            {
                "symbol": "RVNL",
                "title": "Rail Vikas Nigam Limited (RVNL) Metro & EPC Expansion",
                "content": "RVNL executes turnkey railway infrastructure projects including new line construction, doubling, gauge conversion, railway electrification, and metro projects. It has expanded internationally into Central Asia and the Middle East.",
                "category": "Railways & Infra"
            },
            {
                "symbol": "ZOMATO",
                "title": "Zomato Food Delivery & Blinkit Quick Commerce Hypergrowth",
                "content": "Zomato is India's leading consumer tech platform. While food delivery provides stable cash flows with expanding contribution margins, its quick commerce arm 'Blinkit' is experiencing 100%+ YoY gross order value growth, rapidly adding dark stores across Tier 1 and Tier 2 Indian cities.",
                "category": "Consumer Tech"
            },
            {
                "symbol": "TRENT",
                "title": "Trent Ltd (Tata Retail) Zudio Scaling & Retail Dominance",
                "content": "Trent, the retail arm of the Tata Group, is experiencing explosive retail growth led by its value-fashion brand 'Zudio'. Zudio's rapid weekly inventory refresh, aggressive store rollout, and high sales per square foot have made Trent one of the highest compounders on the Indian bourses.",
                "category": "Retail & Fashion"
            },
            {
                "symbol": "LT",
                "title": "Larsen & Toubro (L&T) Mega Infrastructure Order Inflows",
                "content": "L&T is India's premier engineering, procurement, and construction (EPC) conglomerate. Its core order book exceeds ₹4.5 lakh crore spanning railways, renewable energy, green hydrogen plants, hydrocarbons in the Middle East, and high-speed rail corridors.",
                "category": "Infrastructure"
            },
            {
                "symbol": "SUNPHARMA",
                "title": "Sun Pharmaceutical Global Specialty Pipeline & Formulations",
                "content": "Sun Pharma is India's largest pharmaceutical company and the 4th largest global specialty generic firm. It leads the domestic formulations market with strong brand equity and continues expanding its high-margin global specialty dermatology and ophthalmology portfolio (Ilumya, Cequa, Winlevi).",
                "category": "Pharma & Healthcare"
            },
            {
                "symbol": "BAJFINANCE",
                "title": "Bajaj Finance Digital Omnichannel & Asset Franchise",
                "content": "Bajaj Finance is India's largest non-banking financial company (NBFC). Known for its proprietary customer credit database, consumer durable zero-cost EMI financing, and the Bajaj Finserv mobile super-app, it maintains industry-leading AUM growth with strong asset quality.",
                "category": "Financial Services & NBFC"
            }
        ]

        # Merge or initialize
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    existing_symbols = {d.get("symbol") for d in existing}
                    for d in default_docs:
                        if d.get("symbol") not in existing_symbols:
                            existing.append(d)
                    self.documents = existing
                    self._save()
                    return
            except Exception:
                pass

        self.documents = default_docs
        self._save()

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
        except Exception:
            pass

    def add_document(self, symbol: str, title: str, content: str, category: str = "Corporate News"):
        """Adds a new document to the knowledge store."""
        doc = {
            "symbol": symbol.upper(),
            "title": title,
            "content": content,
            "category": category
        }
        self.documents.append(doc)
        self._save()

    def _tokenize(self, text: str) -> set:
        """Tokenization for semantic and keyword matching."""
        words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower())
        return set(words)

    def search(self, query: str, symbol: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant knowledge pieces using BM25-style term scoring.
        Optionally filters by stock symbol.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            query_tokens = set(query.lower().split())

        scores = []
        for doc in self.documents:
            sym_match = False
            if symbol:
                if doc.get("symbol", "").upper() == symbol.upper():
                    sym_match = True
                else:
                    continue

            doc_text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('category', '')}"
            doc_tokens = self._tokenize(doc_text)

            intersection = query_tokens.intersection(doc_tokens)
            score = len(intersection) / (math.sqrt(len(query_tokens) * len(doc_tokens)) + 1e-5)
            if sym_match:
                score += 1.5

            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scores[:top_k] if item[0] > 0.05 or sym_match]

        if not results and symbol:
            results = [d for d in self.documents if d.get("symbol", "").upper() == symbol.upper()]

        return results
