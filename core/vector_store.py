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
        """Loads existing knowledge base or seeds with top Indian stocks insights."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                    return
            except Exception:
                pass
        
        # Pre-seed rich knowledge for top Indian companies
        self.documents = [
            {
                "symbol": "RELIANCE",
                "title": "Reliance Industries Business Overview & Capex",
                "content": "Reliance Industries Limited (RIL) is India's largest conglomerate with core operations in Oil-to-Chemicals (O2C), Telecommunications (Jio Infocomm), and Retail (Reliance Retail). RIL is aggressively investing in New Energy (gigafactories in Jamnagar for solar PV, battery storage, and green hydrogen). Jio leads the 5G rollout with over 470M subscribers, while Retail continues omnichannel expansion.",
                "category": "Company Profile"
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
                "symbol": "TATAMOTORS",
                "title": "Tata Motors EV Leadership & JLR Turnaround",
                "content": "Tata Motors dominates the Indian Electric Vehicle (EV) passenger car segment with Nexon EV, Punch EV, and Curvv EV, commanding >65% market share. Its luxury subsidiary Jaguar Land Rover (JLR) delivers strong free cash flows driven by Defender and Range Rover order books. The company approved demerging into Commercial Vehicles and Passenger Vehicles businesses.",
                "category": "Automobile"
            },
            {
                "symbol": "ICICIBANK",
                "title": "ICICI Bank Digital Growth & Return Ratios",
                "content": "ICICI Bank is known for robust asset quality, industry-low gross NPAs, and superior return on assets (RoA > 2.2%). Key growth drivers include the 'iMobile Pay' app and high-yielding retail, SME, and business banking segments.",
                "category": "Banking & Financials"
            },
            {
                "symbol": "ITC",
                "title": "ITC Hotels Demerger & FMCG Scaling",
                "content": "ITC is an Indian conglomerate spanning Cigarettes, Paperboards, Agribusiness, FMCG, and Hotels. The company completed the demerger of ITC Hotels to unlock shareholder value. FMCG brands (Aashirvaad, Sunfeast, Bingo) are driving non-cigarette revenue growth.",
                "category": "FMCG"
            },
            {
                "symbol": "BHARTIARTL",
                "title": "Bharti Airtel ARPU Growth & 5G Rollout",
                "content": "Bharti Airtel is a premier telecommunications provider in India and Africa. Airtel boasts the highest ARPU (Average Revenue Per User) in the Indian telecom sector (> ₹200), expanding home broadband (Xstream Fiber) and Airtel Business enterprise solutions.",
                "category": "Telecom"
            }
        ]
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
        """Simple word tokenization for local semantic/keyword matching."""
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
            # If symbol is specified, boost matches for that symbol
            sym_match = False
            if symbol:
                if doc.get("symbol", "").upper() == symbol.upper():
                    sym_match = True
                else:
                    continue  # Skip if target symbol is specified and doesn't match

            doc_text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('category', '')}"
            doc_tokens = self._tokenize(doc_text)

            # Jaccard / Overlap score
            intersection = query_tokens.intersection(doc_tokens)
            score = len(intersection) / (math.sqrt(len(query_tokens) * len(doc_tokens)) + 1e-5)
            if sym_match:
                score += 1.5  # Heavy boost for exact ticker match

            scores.append((score, doc))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scores[:top_k] if item[0] > 0.05 or sym_match]
        
        # If no specific match found, return fallback top results for the symbol if provided
        if not results and symbol:
            results = [d for d in self.documents if d.get("symbol", "").upper() == symbol.upper()]

        return results
