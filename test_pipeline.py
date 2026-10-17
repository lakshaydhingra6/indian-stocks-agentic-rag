import sys
import logging
from core.nse_client import NSEClient
from core.technicals import TechnicalAnalyzer
from core.memory_manager import DualTierMemoryManager
from core.vector_store import StockKnowledgeStore
from core.agent import IndianStockRAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPipeline")

def run_tests():
    logger.info("=====================================================")
    logger.info("STARTING COMPREHENSIVE AGENTIC RAG SYSTEM TEST SUITE")
    logger.info("=====================================================")

    # 1. Test NSE Client (Live Data & Fundamentals & News)
    logger.info("\n[1/6] Testing NSE Client Live Data, Valuation & News...")
    nse = NSEClient()
    
    status = nse.get_market_status()
    logger.info(f"Market Status: {status.get('status')} (Source: {status.get('source')})")

    quote = nse.get_live_quote("RELIANCE")
    logger.info(f"Quote RELIANCE: LTP = ₹{quote.get('last_price')} | Change = {quote.get('change')} ({quote.get('percent_change')}%)")
    assert quote.get("last_price", 0) > 0, "Failed to fetch price for RELIANCE"

    fundamentals = nse.get_stock_fundamentals("INFY")
    logger.info(f"Fundamentals INFY: P/E = {fundamentals.get('trailing_pe')} | ROE = {fundamentals.get('return_on_equity_percent')}% | Market Cap = ₹{fundamentals.get('market_cap_crores')} Cr")
    assert fundamentals.get("symbol") == "INFY", "Failed to fetch fundamentals for INFY"

    news = nse.get_live_stock_news("ZOMATO", limit=2)
    logger.info(f"Live News ZOMATO: Fetched {len(news)} articles. Top headline: '{news[0].get('title')}'")
    assert len(news) > 0, "Failed to fetch live news for ZOMATO"

    # 2. Test Historical Candles & Advanced Technicals
    logger.info("\n[2/6] Testing Advanced Technical Indicators & Trade Setups...")
    df = nse.get_historical_ohlcv("TCS", period="3mo")
    indicators = TechnicalAnalyzer.calculate_indicators(df)
    logger.info(f"TCS RSI (14): {indicators.get('rsi_14')} ({indicators.get('rsi_condition')})")
    logger.info(f"TCS Pivot Point: ₹{indicators.get('pivot_point')} | S1: ₹{indicators.get('support_1')} | R1: ₹{indicators.get('resistance_1')}")
    logger.info(f"TCS ATR (14): ₹{indicators.get('atr_14')} | Stop Loss: ₹{indicators.get('suggested_stop_loss')} | Target: ₹{indicators.get('suggested_target')}")
    assert "pivot_point" in indicators and "atr_14" in indicators, "Advanced technicals calculation failed"

    # 3. Test Dual-Tier Memory System
    logger.info("\n[3/6] Testing Dual-Tier Memory System...")
    memory = DualTierMemoryManager()
    memory.update_working_context(symbol="HAL", user_msg="What is HAL price?", assistant_msg="LTP is around 4500")
    resolved = memory.resolve_entity("What is its P/E ratio?")
    logger.info(f"Pronoun 'its' resolved to context symbol: '{resolved}'")
    assert resolved == "HAL", f"Expected HAL, got {resolved}"

    memory.add_to_watchlist("HAL", company_name="Hindustan Aeronautics Ltd", notes="Defense capex play")
    watchlist = memory.get_watchlist()
    symbols = [w["symbol"] for w in watchlist]
    logger.info(f"Watchlist contains: {symbols}")
    assert "HAL" in symbols, "Persistent watchlist failed"

    # 4. Test Semantic Knowledge Store (Defense, Railways, Tech)
    logger.info("\n[4/6] Testing Expanded Knowledge Store...")
    k_store = StockKnowledgeStore()
    res_defense = k_store.search("Tejas fighter aircraft radar order book", symbol="HAL")
    logger.info(f"Defense Search matched: {res_defense[0].get('title') if res_defense else 'None'}")
    assert len(res_defense) > 0, "Failed to match defense knowledge"

    res_zomato = k_store.search("quick commerce Blinkit dark stores", symbol="ZOMATO")
    logger.info(f"Zomato Search matched: {res_zomato[0].get('title') if res_zomato else 'None'}")
    assert len(res_zomato) > 0, "Failed to match Zomato knowledge"

    # 5. Test Head-to-Head Comparison Tool
    logger.info("\n[5/6] Testing Stock Comparison Tool...")
    agent = IndianStockRAGAgent()
    comp_res, comp_traces = agent.run("Compare TCS and Infosys")
    logger.info(f"Comparison response snippet:\n{comp_res[:200]}...\n")
    assert len(comp_traces) > 0, "Comparison tool failed to execute"

    # 7. Test Upcoming IPOs & Financials Intelligence
    logger.info("\n[7/7] Testing Upcoming IPOs & Primary Market Intelligence...")
    from core.ipo_manager import IPOManager
    ipo_mgr = IPOManager()
    all_ipos = ipo_mgr.get_upcoming_ipos()
    logger.info(f"Loaded {len(all_ipos)} Upcoming IPOs. First: {all_ipos[0].get('company_name')}")
    assert len(all_ipos) >= 5, "Expected at least 5 IPOs in database"

    swiggy_ipo = ipo_mgr.get_ipo_details("swiggy")
    assert swiggy_ipo is not None, "Failed to get Swiggy IPO details"
    rev = swiggy_ipo["financials"]["turnover_revenue"]["FY24"]
    logger.info(f"Swiggy IPO FY24 Turnover: {rev} | Issue Size: ₹{swiggy_ipo.get('issue_size_crores')} Cr")

    ipo_list_res, ipo_list_traces = agent.run("What upcoming IPOs are coming to NSE?")
    logger.info(f"IPO List Agent Response Snippet:\n{ipo_list_res[:200]}...\n")
    assert len(ipo_list_traces) > 0, "Agent failed to execute tool_get_upcoming_ipos"

    ipo_det_res, ipo_det_traces = agent.run("What is Hyundai IPO turnover and profit?")
    logger.info(f"Hyundai IPO Agent Response Snippet:\n{ipo_det_res[:200]}...\n")
    assert len(ipo_det_traces) > 0, "Agent failed to execute tool_get_ipo_details"

    logger.info("=====================================================")
    logger.info("✅ ALL 7 COMPREHENSIVE TEST STAGES PASSED SUCCESSFULLY!")
    logger.info("=====================================================")

if __name__ == "__main__":
    run_tests()
