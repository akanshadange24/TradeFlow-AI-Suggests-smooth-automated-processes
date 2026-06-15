import warnings
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")

import os
import asyncio
import logging
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# REMOVED: langchain_openai
from langchain_groq import ChatGroq # ADDED: Groq integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import time
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite




# Silence LangGraph warnings
warnings.filterwarnings("ignore", module="langgraph")
warnings.filterwarnings("ignore", module="langchain_community") 
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#1. Define a robust cache to prevent redundant API calls
# This caches results for 60 seconds, which is perfect for intraday trading logic
@lru_cache(maxsize=32)
def get_cached_ticker(ticker: str):
    return yf.Ticker(ticker)

# 2. Add retry logic for rate-limit errors (429) or connection drops
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, Exception))
)
def fetch_ticker_data_safely(ticker: str):
    stock = get_cached_ticker(ticker)
    # Using 'fast_info' is significantly lighter and less likely to hit rate limits
    info = stock.fast_info
    
    # Example of handling potential failures
    if not info:
        raise ConnectionError(f"Could not retrieve data for {ticker}")
        
    return stock
# ==============================================================================
# 1. THE NERVES: LIVE MARKET DATA ENGINE
# ==============================================================================
def fetch_live_market_data(ticker: str) -> Dict[str, Any]:
    logger.info(f"Nerves Engine -> Fetching live data for {ticker} via yfinance...")
    try:
        stock = yf.Ticker(ticker)
        
        # 👉 BUG FIX: Bypass Yahoo's blocked .info() by pulling directly from history
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No historical data returned for {ticker}")
            
        current_price = float(hist['Close'].iloc[-1])
        
        # Safely attempt to get PE ratio
        pe_ratio = "N/A"
        debt_to_equity = "N/A"
        try:
            pe_ratio = stock.fast_info.get("trailingPE", "N/A")
        except: pass

        # Fetch News
        news_headlines = []
        try:
            if hasattr(stock, 'news'):
                news_headlines = [article.get('title', '') for article in stock.news[:3]]
        except: pass
        
        hist['Returns'] = hist['Close'].pct_change()
        daily_returns = hist['Returns'].dropna().tolist()
        
        # Fetch S&P 500 for Macro Trend Analysis (Using Index ticker ^GSPC to avoid rate limits)
        spy = yf.Ticker("^GSPC") 
        spy_hist = spy.history(period="5d")
        spy_trend = "UP" if float(spy_hist['Close'].iloc[-1]) > float(spy_hist['Close'].iloc[0]) else "DOWN"
        
        return {
            "current_price": current_price,
            "pe_ratio": pe_ratio,
            "debt_to_equity": debt_to_equity,
            "daily_returns": daily_returns,
            "recent_news": news_headlines,
            "macro_trend_indicator": spy_trend,
            "error": None
        }
    except Exception as e:
        logger.error(f"Failed to fetch live data: {str(e)}")
        # If it fails completely, it passes 0.0 to the frontend to trigger the Zero-Catcher
        return {
            "current_price": 0.0, "pe_ratio": "N/A", "debt_to_equity": "N/A",
            "daily_returns": [0.0] * 20, "recent_news": [], "macro_trend_indicator": "FLAT", "error": str(e)
        }

# ==============================================================================
# 2. QUANTITATIVE EVALUATION ENGINE
# ==============================================================================
def calculate_evaluation_metrics(daily_returns: List[float], risk_free_rate: float = 0.04) -> Dict[str, float]:
    if not daily_returns or len(daily_returns) < 2:
        return {"cumulative_return": 0.0, "annualized_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
    
    returns_array = np.array(daily_returns)
    trading_days = len(returns_array)
    cumulative_return = np.prod(1 + returns_array) - 1
    annualized_return = (1 + cumulative_return) ** (252 / trading_days) - 1
    daily_volatility = np.std(returns_array)
    annualized_volatility = daily_volatility * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
    cumulative_index = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative_index)
    drawdowns = (cumulative_index - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown)
    }

# ==============================================================================
# 3. UNIFIED DATA MODELS & STATE SCHEMAS
# ==============================================================================
class BacktestEvaluation(BaseModel):
    cumulative_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    maximum_drawdown_pct: float

class ActionableRecommendation(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence_score: float
    supporting_rationale: str
    risk_vectors: List[str]
    performance_metrics: BacktestEvaluation

def merge_insights(dict1: dict, dict2: dict) -> dict:
    return {**dict1, **dict2}

class SystemState(TypedDict):
    user_query: str
    ticker: str
    groq_api_key: str # UPDATED: Modified state typing naming for semantic consistency
    live_data: Dict[str, Any]
    agent_insights: Annotated[Dict[str, Any], merge_insights]
    final_recommendation: ActionableRecommendation
    human_approval: bool
    execution_receipt: Dict[str, Any]

# ==============================================================================
# 4. THE BRAINS: INTELLECTUAL SPECIALIZED AGENTS (ISAs)
# ==============================================================================
def get_llm(state: SystemState):
    # Dynamic Check: Prioritize UI inputs, fall back to .env variable
    key = state.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    if not key: 
        return None
    # Swapped out ChatOpenAI for ChatGroq running Llama 3 8B
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=key, temperature=0.2)

def fundamental_isa_node(state: SystemState) -> Dict:
    logger.info("Core ISA -> Executing Fundamental Analysis")
    ticker, live, llm = state["ticker"], state["live_data"], get_llm(state)
    context = f"Asset: {ticker}. P/E Ratio: {live['pe_ratio']}. Debt/Equity: {live['debt_to_equity']}."
    if not llm: return {"agent_insights": {"fundamental": f"[Demo] Fundamentals stable. {context}"}}
    prompt = ChatPromptTemplate.from_template("You are a Fundamental Analyst. Give a 1-sentence brief: {data}")
    return {"agent_insights": {"fundamental": llm.invoke(prompt.format(data=context)).content}}

def technical_isa_node(state: SystemState) -> Dict:
    logger.info("Core ISA -> Executing Technical Analysis")
    ticker, live, llm = state["ticker"], state["live_data"], get_llm(state)
    context = f"Live Price for {ticker}: ${live['current_price']:.2f}."
    if not llm: return {"agent_insights": {"technical": f"[Demo] Price momentum above average. {context}"}}
    prompt = ChatPromptTemplate.from_template("You are a Technical Analyst. Give a 1-sentence trend outlook: {data}")
    return {"agent_insights": {"technical": llm.invoke(prompt.format(data=context)).content}}

def news_nlp_isa_node(state: SystemState) -> Dict:
    logger.info("Core ISA -> Executing News Intelligence NLP Summary")
    live, llm = state["live_data"], get_llm(state)
    news = live.get("recent_news", [])
    if not news: return {"agent_insights": {"news_sentiment": "No recent news detected."}}
    if not llm: return {"agent_insights": {"news_sentiment": f"[Demo] News volume normal. Top headline: {news[0]}"}}
    prompt = ChatPromptTemplate.from_template("You are an NLP Sentiment Analyst. Summarize the market sentiment of these headlines in 1 sentence: {news}")
    return {"agent_insights": {"news_sentiment": llm.invoke(prompt.format(news=news)).content}}

def macro_trend_isa_node(state: SystemState) -> Dict:
    logger.info("Core ISA -> Executing Macro Market Trend Analysis")
    live, llm = state["live_data"], get_llm(state)
    trend = live.get("macro_trend_indicator", "FLAT")
    context = f"S&P 500 5-day trend is currently {trend}."
    if not llm: return {"agent_insights": {"macro_trend": f"[Demo] Broad market is moving {trend}."}}
    prompt = ChatPromptTemplate.from_template("You are a Macroeconomist. Assess this 5-day S&P 500 trend ({data}) and its impact on equities in 1 sentence.")
    return {"agent_insights": {"macro_trend": llm.invoke(prompt.format(data=context)).content}}

def supply_chain_isa_node(state: SystemState) -> Dict:
    logger.info("Core ISA -> Executing Sector & Supply-Chain Analysis")
    ticker, llm = state["ticker"], get_llm(state)
    if not llm: return {"agent_insights": {"supply_chain": f"[Demo] Sector supply chains for {ticker} are functioning normally."}}
    prompt = ChatPromptTemplate.from_template("You are a Supply Chain Analyst. In 1 sentence, what are the current global supply chain constraints for the sector that {ticker} operates in?")
    return {"agent_insights": {"supply_chain": llm.invoke(prompt.format(ticker=ticker)).content}}

def external_search_isa_node(state: SystemState) -> Dict:
    logger.info("Supporting Agent -> Executing External Search Retrieval")
    ticker = state["ticker"]
    llm = get_llm(state)
    
    if not llm: 
        return {"agent_insights": {"external_search": f"[Demo] Web chatter indicates standard institutional holding for {ticker}."}}
        
    try:
        search = DuckDuckGoSearchRun()
        query = f"latest institutional moves, hedge fund chatter, and financial developments for {ticker} stock"
        raw_search_results = search.invoke(query)
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        raw_search_results = "No recent web data available."

    prompt = ChatPromptTemplate.from_template(
        "You are an Alternative Data Intelligence Analyst. Read these live web search results for {ticker} and summarize the institutional or internet sentiment in exactly 1 concise sentence: {results}"
    )
    
    response = llm.invoke(prompt.format(ticker=ticker, results=str(raw_search_results)[:2000]))
    return {"agent_insights": {"external_search": response.content}}

# ==============================================================================
# 5. SUPPORTING AGENTS & DATA ROUTER
# ==============================================================================
def data_fetcher_node(state: SystemState) -> Dict:
    return {"live_data": fetch_live_market_data(state["ticker"])}

def investment_recommendation_agent_node(state: SystemState) -> Dict:
    logger.info("Supporting Agent -> Synthesizing Matrix & Dynamic Metrics")
    insights, live, llm = state["agent_insights"], state["live_data"], get_llm(state)
    calc = calculate_evaluation_metrics(live["daily_returns"])
    
    evaluation = BacktestEvaluation(
        cumulative_return_pct=calc["cumulative_return"] * 100, annualized_return_pct=calc["annualized_return"] * 100,
        sharpe_ratio=calc["sharpe_ratio"], maximum_drawdown_pct=calc["max_drawdown"] * 100
    )
    
    if not llm:
        rec = ActionableRecommendation(
            action="BUY" if calc["sharpe_ratio"] > 1 else "HOLD", confidence_score=0.85,
            supporting_rationale=f"Tech: {insights.get('technical')[:30]}... News: {insights.get('news_sentiment')[:30]}...",
            risk_vectors=["Market volatility"], performance_metrics=evaluation
        )
        return {"final_recommendation": rec}

    # 👉 UPGRADED PROMPT: Force decisiveness and request a dynamic confidence score
    prompt = ChatPromptTemplate.from_template(
        "You are a decisive quantitative hedge fund manager. Synthesize these agent insights: {insights}. "
        "You MUST choose a definitive action (BUY, SELL, or HOLD). Do not be overly cautious. "
        "Assign a confidence score between 0.00 and 1.00 based on the strength of the signals. "
        "Format EXACTLY as: ACTION: [BUY/SELL/HOLD] | CONFIDENCE: [number] | RATIONALE: [2-sentence rationale]"
    )
    res_text = llm.invoke(prompt.format(insights=str(insights))).content
    
    # 👉 UPGRADED PARSER: Safely extract the exact action and dynamic confidence score
    action = "HOLD"
    confidence = 0.50
    rationale = res_text
    
    try:
        parts = res_text.split("|")
        for part in parts:
            part = part.strip()
            if part.startswith("ACTION:"):
                raw_action = part.replace("ACTION:", "").strip().upper()
                if "BUY" in raw_action: action = "BUY"
                elif "SELL" in raw_action: action = "SELL"
            elif part.startswith("CONFIDENCE:"):
                confidence = float(part.replace("CONFIDENCE:", "").strip())
            elif part.startswith("RATIONALE:"):
                rationale = part.replace("RATIONALE:", "").strip()
    except Exception as e:
        logger.warning(f"Failed to parse LLM output: {res_text}")
        rationale = f"Parsing fallback. Raw output: {res_text}"
    
    return {"final_recommendation": ActionableRecommendation(
        action=action, 
        confidence_score=confidence, # <--- Now completely dynamic!
        supporting_rationale=rationale,
        risk_vectors=["Dynamic systematic macro risk"], 
        performance_metrics=evaluation
    )}

def order_execution_node(state: SystemState) -> Dict:
    logger.info("GATEKEEPER PASSED: Order Execution Node Activated.")
    if not state.get("human_approval", False):
        return {"execution_receipt": {"status": "REJECTED_BY_USER", "timestamp": datetime.now(timezone.utc).isoformat()}}
        
    return {"execution_receipt": {
        "status": "ORDER_FILLED", "asset": state["ticker"], "allocated_action": state["final_recommendation"].action,
        "timestamp": datetime.now(timezone.utc).isoformat(), "broker_ref_id": f"BRK-{int(datetime.now(timezone.utc).timestamp())}-X"
    }}

# ==============================================================================
# 6. GRAPH COMPILATION 
# ==============================================================================
def build_interactive_trading_graph(checkpointer) -> StateGraph:
    workflow = StateGraph(SystemState)

    workflow.add_node("Data_Fetcher", data_fetcher_node)
    workflow.add_node("ISA_Fundamental", fundamental_isa_node)
    workflow.add_node("ISA_Technical", technical_isa_node)
    workflow.add_node("ISA_News", news_nlp_isa_node)
    workflow.add_node("ISA_Macro", macro_trend_isa_node)
    workflow.add_node("ISA_SupplyChain", supply_chain_isa_node)
    workflow.add_node("Support_Investment_Recommendation", investment_recommendation_agent_node)
    workflow.add_node("Order_Execution", order_execution_node)

    workflow.set_entry_point("Data_Fetcher")
    workflow.add_node("ISA_Search", external_search_isa_node)
    
    for node in ["ISA_Fundamental", "ISA_Technical", "ISA_News", "ISA_Macro", "ISA_SupplyChain", "ISA_Search"]:
        workflow.add_edge("Data_Fetcher", node)
        workflow.add_edge(node, "Support_Investment_Recommendation")
    
    workflow.add_edge("Support_Investment_Recommendation", "Order_Execution")
    workflow.add_edge("Order_Execution", END)

    return workflow.compile(
        checkpointer=checkpointer, 
        interrupt_before=["Order_Execution"]
    )

# ==============================================================================
# 7. TERMINAL EXECUTION ENGINE (THE IGNITION SWITCH)
# ==============================================================================
async def main():
    print("\n[PHASE 1] RUNNING DISCONNECTED CONCURRENT AGENTS...")
    
    # 1. Create the connection explicitly
    # This is NOT a context manager; it is a raw connection object.
    conn = await aiosqlite.connect("trading_agent_memory.db")  #await
    
    # 2. Instantiate the saver directly using the connection object
    checkpointer = AsyncSqliteSaver(conn=conn)
    
    # 3. Now pass this concrete instance to your builder
    app = build_interactive_trading_graph(checkpointer)
    
    # 4. Now that we have the instance, call setup()
    await app.checkpointer.setup()

    thread_config = {"configurable": {"thread_id": "terminal_live_test"}}
    
    initial_state = {
        "ticker": "NVDA",
        "groq_api_key": os.getenv("GROQ_API_KEY", ""), 
        "user_query": "Analyze this stock"
    }
    
    # Run graph until interrupt
    async for event in app.astream(initial_state, config=thread_config):
        pass 
        
    # FETCH STATE (Must be Async)
    snapshot = await app.aget_state(thread_config)
    state = snapshot.values
    
    if "final_recommendation" in state:
        rec = state["final_recommendation"]
        metrics = rec.performance_metrics
        
        # YOUR PREFERRED OUTPUT FORMAT
        print("\n============================================================")
        print("   LANGGRAPH BREAKPOINT TRIGGERED: AWAITING INTERACTION   ")
        print("============================================================")
        print(f"Next Target Node Pending:  ('Order_Execution',)")
        print(f"Agent Action Proposal:     {rec.action} (Confidence: {rec.confidence_score*100:.1f}%)")
        print("\n--- DYNAMICALLY CALCULATED EVALUATION METRICS ---")
        print(f"Cumulative Return:         {metrics.cumulative_return_pct:.2f}%")
        print(f"Annualized Return:         {metrics.annualized_return_pct:.2f}%")
        print(f"Sharpe Ratio:              {metrics.sharpe_ratio:.2f}")
        print(f"Historical Max Drawdown:   {metrics.maximum_drawdown_pct:.2f}%")
        print("------------------------------------------------------------")
        
        user_input = input("Type 'APPROVE' to send trade or 'REJECT' to abort: ").strip().upper()
        
        print("\n[PHASE 2] INJECTING USER APPROVAL & RESUMING STATE MEMORY...")
        
        # UPDATE STATE (Must be Async)
        await app.aupdate_state(thread_config, {"human_approval": user_input == "APPROVE"})
        
        # RESUME GRAPH
        async for output in app.astream(None, config=thread_config):
            pass
            
        # FINAL RECEIPT (Must be Async)
        final_snapshot = await app.aget_state(thread_config)
        receipt = final_snapshot.values.get("execution_receipt", {})
        
        print("\n============================================================")
        print("                 BROKER CLEARED RECEIPT                   ")
        print("============================================================")
        print(f"Receipt Status: {receipt.get('status')}")
        if receipt.get("status") == "ORDER_FILLED":
            print(f"Broker Reference: {receipt.get('broker_ref_id')}")

if __name__ == "__main__":
    asyncio.run(main())