import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DATA_DIR

class IPOManager:
    """
    Manages Comprehensive Upcoming & Recent Indian IPO Intelligence.
    Tracks issue dates, price bands, lot sizes, issue sizes (Fresh vs OFS),
    GMP (Grey Market Premium), and detailed 3-year financials
    (Turnover/Revenue, Net Profit PAT, EBITDA margins, and peer valuation).
    """

    def __init__(self, storage_path: Path = DATA_DIR / "ipo_data.json"):
        self.storage_path = storage_path
        self.ipos: List[Dict[str, Any]] = []
        self._load_or_seed()

    def _load_or_seed(self):
        """Loads IPO dataset or seeds with real-world Indian IPO data."""
        default_ipos = [
            {
                "id": "hyundai",
                "company_name": "Hyundai Motor India Limited",
                "symbol": "HYUNDAI",
                "sector": "Automobile (Passenger Vehicles)",
                "status": "Upcoming / Active",
                "issue_dates": "15 Oct 2026 - 17 Oct 2026",
                "listing_date": "22 Oct 2026",
                "price_band": "₹1,865 - ₹1,960",
                "lot_size": 7,
                "min_investment": 13720,
                "issue_size_crores": 27870,
                "fresh_issue_crores": 0,
                "ofs_crores": 27870,
                "gmp_premium": "₹65 (3.3%)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹47,378 Cr",
                        "FY23": "₹60,307 Cr",
                        "FY24": "₹69,829 Cr"
                    },
                    "net_profit_pat": {
                        "FY22": "₹2,867 Cr",
                        "FY23": "₹4,709 Cr",
                        "FY24": "₹6,060 Cr"
                    },
                    "ebitda_margin_fy24": "13.1%",
                    "ronw_roe_fy24": "28.6%",
                    "pe_at_upper_band": "26.3x (vs Maruti Suzuki ~29.1x)",
                    "debt_to_equity": "0.18"
                },
                "key_strengths": "India's 2nd largest passenger vehicle manufacturer (14.6% market share). Strong SUV lineup (Creta, Venue, Exter) and EV roadmap (Creta EV). Export leader from India to 80+ countries.",
                "objectives_of_issue": "100% Offer for Sale (OFS) by parent Hyundai Motor Company Korea; enhance brand visibility and provide liquidity to shareholders."
            },
            {
                "id": "swiggy",
                "company_name": "Swiggy Limited",
                "symbol": "SWIGGY",
                "sector": "Consumer Tech & Quick Commerce",
                "status": "Upcoming",
                "issue_dates": "06 Nov 2026 - 08 Nov 2026",
                "listing_date": "13 Nov 2026",
                "price_band": "₹371 - ₹390",
                "lot_size": 38,
                "min_investment": 14820,
                "issue_size_crores": 11327,
                "fresh_issue_crores": 4499,
                "ofs_crores": 6828,
                "gmp_premium": "₹25 (6.4%)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹5,705 Cr",
                        "FY23": "₹8,265 Cr",
                        "FY24": "₹11,247 Cr"
                    },
                    "net_profit_pat": {
                        "FY22": "-₹3,628 Cr (Loss)",
                        "FY23": "-₹4,179 Cr (Loss)",
                        "FY24": "-₹2,350 Cr (Loss reduced 44%)"
                    },
                    "ebitda_margin_fy24": "-18.2% (Improving)",
                    "gross_order_value_fy24": "₹35,000+ Cr",
                    "pe_at_upper_band": "Negative (Valued at ~₹87,000 Cr Market Cap vs Zomato ~₹2.3 Lakh Cr)",
                    "debt_to_equity": "0.05"
                },
                "key_strengths": "Duopoly in Indian food delivery (alongside Zomato). High-growth quick commerce platform 'Instamart' with expanding dark store network across 40+ Indian cities. High user stickiness via Swiggy One membership.",
                "objectives_of_issue": "Investment in Instamart dark store expansion (₹1,178 Cr), technology and cloud infrastructure (₹703 Cr), brand marketing (₹1,115 Cr), and inorganic acquisitions."
            },
            {
                "id": "ntpc_green",
                "company_name": "NTPC Green Energy Limited (NGEL)",
                "symbol": "NTPCGREEN",
                "sector": "Renewable Energy & Green Power",
                "status": "Upcoming",
                "issue_dates": "19 Nov 2026 - 22 Nov 2026",
                "listing_date": "27 Nov 2026",
                "price_band": "₹102 - ₹108",
                "lot_size": 138,
                "min_investment": 14904,
                "issue_size_crores": 10000,
                "fresh_issue_crores": 10000,
                "ofs_crores": 0,
                "gmp_premium": "₹14 (13.0%)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹910 Cr",
                        "FY23": "₹1,262 Cr",
                        "FY24": "₹1,962 Cr"
                    },
                    "net_profit_pat": {
                        "FY22": "₹94 Cr",
                        "FY23": "₹171 Cr",
                        "FY24": "₹344 Cr (+101% YoY)"
                    },
                    "operational_capacity": "3.2 GW (Target 60 GW by 2032)",
                    "ebitda_margin_fy24": "87.4% (Utility Scale)",
                    "ronw_roe_fy24": "8.9%",
                    "pe_at_upper_band": "52.4x (Peer Adani Green trades > 150x)",
                    "debt_to_equity": "1.85 (CapEx Intensive)"
                },
                "key_strengths": "Renewable energy arm of Maharatna PSU NTPC. Highest credit rating (AAA). 100% sovereign-backed long-term Power Purchase Agreements (PPAs) with SECI and state DISCOMs.",
                "objectives_of_issue": "100% Fresh Issue: ₹7,500 Cr for repayment of debt taken for green projects, remainder for solar and green hydrogen gigafactory capex."
            },
            {
                "id": "waaree",
                "company_name": "Waaree Energies Limited",
                "symbol": "WAAREE",
                "sector": "Solar Energy & Equipment",
                "status": "Active / Bidding",
                "issue_dates": "21 Oct 2026 - 23 Oct 2026",
                "listing_date": "28 Oct 2026",
                "price_band": "₹1,427 - ₹1,503",
                "lot_size": 9,
                "min_investment": 13527,
                "issue_size_crores": 4321,
                "fresh_issue_crores": 3600,
                "ofs_crores": 721,
                "gmp_premium": "₹1,420 (94.5% Listing Gain Expected)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹2,854 Cr",
                        "FY23": "₹6,750 Cr",
                        "FY24": "₹11,398 Cr (+68.8% YoY)"
                    },
                    "net_profit_pat": {
                        "FY22": "₹79 Cr",
                        "FY23": "₹500 Cr",
                        "FY24": "₹1,274 Cr (+154.8% YoY)"
                    },
                    "ebitda_margin_fy24": "13.8%",
                    "ronw_roe_fy24": "30.3%",
                    "pe_at_upper_band": "34.2x (vs Premier Energies ~78x)",
                    "debt_to_equity": "0.12 (Virtually Debt Free)"
                },
                "key_strengths": "India's largest manufacturer and exporter of Solar PV Modules (12 GW installed capacity). Massive order book of >20 GW. Major beneficiary of US tariffs on Chinese solar cells.",
                "objectives_of_issue": "₹2,775 Cr for setting up a 6 GW Ingot-Wafer, Solar Cell, and Module manufacturing facility in Odisha, India."
            },
            {
                "id": "afcons",
                "company_name": "Afcons Infrastructure Limited (Shapoorji Pallonji)",
                "symbol": "AFCONS",
                "sector": "Infrastructure & Engineering",
                "status": "Upcoming",
                "issue_dates": "25 Oct 2026 - 29 Oct 2026",
                "listing_date": "04 Nov 2026",
                "price_band": "₹440 - ₹463",
                "lot_size": 32,
                "min_investment": 14816,
                "issue_size_crores": 5430,
                "fresh_issue_crores": 1250,
                "ofs_crores": 4180,
                "gmp_premium": "₹55 (11.9%)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹11,019 Cr",
                        "FY23": "₹12,637 Cr",
                        "FY24": "₹13,267 Cr"
                    },
                    "net_profit_pat": {
                        "FY22": "₹357 Cr",
                        "FY23": "₹410 Cr",
                        "FY24": "₹449 Cr"
                    },
                    "order_book_fy24": "₹31,650 Cr",
                    "ebitda_margin_fy24": "11.2%",
                    "ronw_roe_fy24": "13.6%",
                    "pe_at_upper_band": "35.8x (vs L&T ~37.4x)",
                    "debt_to_equity": "0.78"
                },
                "key_strengths": "Flagship infrastructure engineering arm of the 150-year-old Shapoorji Pallonji Group. Track record of executing complex mega-projects: Mumbai Trans Harbour Link (Atal Setu), Chenab Railway Bridge (world's highest), and international marine ports.",
                "objectives_of_issue": "Funding capital expenditure for construction equipment (₹320 Cr), working capital requirements (₹500 Cr), repayment of borrowings (₹500 Cr), and promoter debt reduction via OFS."
            },
            {
                "id": "hero_fincorp",
                "company_name": "Hero Fincorp Limited",
                "symbol": "HEROFIN",
                "sector": "Financial Services (NBFC)",
                "status": "Upcoming",
                "issue_dates": "TBA (Q4 2026)",
                "listing_date": "TBA",
                "price_band": "₹820 - ₹850 (Tentative)",
                "lot_size": 17,
                "min_investment": 14450,
                "issue_size_crores": 3668,
                "fresh_issue_crores": 2100,
                "ofs_crores": 1568,
                "gmp_premium": "₹110 (12.9%)",
                "listing_exchanges": "NSE, BSE",
                "financials": {
                    "turnover_revenue": {
                        "FY22": "₹3,450 Cr",
                        "FY23": "₹4,120 Cr",
                        "FY24": "₹4,892 Cr (+18.7%)"
                    },
                    "net_profit_pat": {
                        "FY22": "-₹194 Cr (Loss)",
                        "FY23": "₹457 Cr",
                        "FY24": "₹637 Cr (+39.4%)"
                    },
                    "aum_assets_under_mgmt": "₹51,800 Cr",
                    "gross_npa": "4.1%",
                    "net_npa": "2.2%",
                    "ronw_roe_fy24": "13.5%",
                    "debt_to_equity": "5.2x"
                },
                "key_strengths": "Captive retail two-wheeler financing for Hero MotoCorp (world's largest 2-wheeler maker). Strong growth in SME and affordable home loans with presence across 4,000+ touchpoints in India.",
                "objectives_of_issue": "Augment Tier-1 capital base to meet future capital adequacy requirements arising from loan asset growth."
            }
        ]

        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.ipos = json.load(f)
                    return
            except Exception:
                pass

        self.ipos = default_ipos
        self._save()

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.ipos, f, indent=2)
        except Exception:
            pass

    def get_upcoming_ipos(self, status: str = "all") -> List[Dict[str, Any]]:
        """Returns list of upcoming or active IPOs."""
        if status == "all":
            return self.ipos
        return [i for i in self.ipos if status.lower() in i.get("status", "").lower()]

    def get_ipo_details(self, query: str) -> Optional[Dict[str, Any]]:
        """Finds detailed company financial, turnover, and valuation data for an IPO."""
        q = query.strip().lower()
        for ipo in self.ipos:
            if (q in ipo["company_name"].lower() or 
                q in ipo.get("symbol", "").lower() or 
                q in ipo.get("id", "").lower()):
                return ipo
        return None

    def search_ipos(self, keyword: str) -> List[Dict[str, Any]]:
        """Searches across IPO companies, sectors, and strengths."""
        kw = keyword.lower()
        return [
            i for i in self.ipos
            if kw in i["company_name"].lower()
            or kw in i.get("sector", "").lower()
            or kw in i.get("key_strengths", "").lower()
        ]
