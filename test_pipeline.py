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
    logger.info("=========================================")
    logger.info("STARTING AGENTIC RAG SYSTEM TEST PIPELINE")
    logger.info("=========================================")

    # 1. Test NSE Client (Live Data)
    logger.info("\n[1/5] Testing NSE Client & Live Data Fetch...")
    nse = NSEClient()
    
    status = nse.get_market_status()
    logger.info(f"Market Status: {status.get('status')} (Source: {status.get('source')})")
    assert status is not None, "Failed to get market status"

    quote = nse.get_live_quote("RELIANCE")
    logger.info(f"Live Quote for RELIANCE: LTP = ₹{quote.get('last_price')} | Change = {quote.get('change')} ({quote.get('percent_change')}%) [Source: {quote.get('source')}]")
    assert quote.get("last_price", 0) > 0, "Failed to fetch valid price for RELIANCE"

    # 2. Test Historical OHLCV & Technical Analysis
    logger.info("\n[2/5] Testing Historical Candles & Technical Indicators...")
    df = nse.get_historical_ohlcv("RELIANCE", period="3mo")
    assert df is not None and not df.empty, "Failed to fetch historical candles"
    logger.info(f"Fetched {len(df)} candles for RELIANCE.")

    indicators = TechnicalAnalyzer.calculate_indicators(df)
    logger.info(f"RSI (14): {indicators.get('rsi_14')} ({indicators.get('rsi_condition')})")
    logger.info(f"Overall Bias: {indicators.get('overall_bias')}")
    logger.info(f"EMA 200: {indicators.get('ema_200')}")
    assert "rsi_14" in indicators, "Technical indicators calculation failed"

    # 3. Test Dual-Tier Memory System (Short-Term + Long-Term SQLite)
    logger.info("\n[3/5] Testing Dual-Tier Memory System...")
    memory = DualTierMemoryManager()

    # Short-term working context & pronoun resolution
    memory.update_working_context(symbol="INFY", user_msg="What is Infosys price?", assistant_msg="LTP is around 1800")
    resolved = memory.resolve_entity("What is its 200 EMA?")
    logger.info(f"Pronoun 'its' resolved to active context stock: '{resolved}'")
    assert resolved == "INFY", f"Expected INFY, got {resolved}"

    # Long-term persistent SQLite memory
    memory.add_to_watchlist("TATAMOTORS", company_name="Tata Motors Ltd", notes="Tracking EV growth")
    watchlist = memory.get_watchlist()
    symbols_in_wl = [w["symbol"] for w in watchlist]
    logger.info(f"Persistent Watchlist items: {symbols_in_wl}")
    assert "TATAMOTORS" in symbols_in_wl, "Watchlist persistence failed"

    # 4. Test Local Vector Knowledge Store
    logger.info("\n[4/5] Testing Semantic Knowledge Store...")
    k_store = StockKnowledgeStore()
    search_results = k_store.search("EV passenger car electric vehicle", symbol="TATAMOTORS")
    logger.info(f"Search results for EV query: {len(search_results)} found.")
    assert len(search_results) > 0, "Semantic search returned empty results"
    logger.info(f"Matched title: {search_results[0].get('title')}")

    # 5. Test Full Agent Execution
    logger.info("\n[5/5] Testing Autonomous Agentic Execution Loop...")
    agent = IndianStockRAGAgent()
    
    # Query 1: Direct live stock analysis
    response_1, traces_1 = agent.run("What is the current price and technical trend of TCS?")
    logger.info(f"Query 1 Tool Traces Count: {len(traces_1)}")
    logger.info(f"Query 1 Response Preview:\n{response_1[:250]}...\n")
    assert len(traces_1) > 0, "Agent did not execute any tools for Query 1"

    # Query 2: Contextual query (pronoun memory)
    response_2, traces_2 = agent.run("How does it compare to Infosys?")
    logger.info(f"Query 2 Tool Traces Count: {len(traces_2)}")
    logger.info(f"Query 2 Response Preview:\n{response_2[:250]}...\n")

    logger.info("=========================================")
    logger.info("✅ ALL 5 PIPELINE TESTS PASSED SUCCESSFULLY!")
    logger.info("=========================================")

if __name__ == "__main__":
    run_tests()
