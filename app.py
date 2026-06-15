
import streamlit as st
import os
import asyncio
import aiosqlite
import nest_asyncio
import time
import warnings
import yfinance as yf  # 👉 ADDED: For the live top ticker tape
from datetime import datetime, timezone
from dotenv import load_dotenv
from Trading_agents import build_interactive_trading_graph, get_llm
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
import urllib.request
import urllib.parse
import json

# ==============================================================================
# SYSTEM INITIALIZATION & WARNING SUPPRESSION
# ==============================================================================
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langgraph.checkpoint.base")

nest_asyncio.apply()
load_dotenv()

st.set_page_config(page_title="AI Multi-Agent Trading Terminal", page_icon="🤖", layout="wide")

# ==============================================================================
# LIVE MACRO TICKER TAPE (GOLD, CURRENCY, INDEXES)
# ==============================================================================
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_exchange_rate():
    """Fetches the live USD to INR exchange rate."""
    try:
        return yf.Ticker("INR=X").fast_info.last_price
    except Exception:
        return 83.50  # Safe fallback if Yahoo Finance rate-limits us

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 mins so it doesn't slow down the app
def get_market_ticker():
    # Key global and Indian macroeconomic indicators
    symbols = {
        "NIFTY 50": "^NSEI", 
        "SENSEX": "^BSESN", 
        "USD/INR": "INR=X", 
        "GOLD": "GC=F", 
        "SILVER": "SI=F",
        "CRUDE OIL": "CL=F"
    }
    results = []
    for name, sym in symbols.items():
        try:
            stock = yf.Ticker(sym)
            price = stock.fast_info.last_price
            prev = stock.fast_info.previous_close
            change = ((price - prev) / prev) * 100
            
            # Formatting logic: Currencies and global commodities vs Indian Indexes
            currency_symbol = "₹" if name in ["NIFTY 50", "SENSEX", "USD/INR"] else "$"
            arrow = "▲" if change >= 0 else "▼"
            color = "#00FF00" if change >= 0 else "#FF4136" # Bright Green or Red
            
            results.append(
                f"<span style='color: #E0E0E0; font-weight: 600; font-family: monospace;'>{name}:</span> "
                f"<span style='color: {color}; font-weight: bold; font-family: monospace;'>{currency_symbol}{price:.2f} ({arrow}{abs(change):.2f}%)</span>"
            )
        except Exception:
            continue
            
    # Join with a distinct separator
    return "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(results)

# Inject the scrolling ticker tape at the very top of the UI
ticker_html = get_market_ticker()
if ticker_html:
    st.markdown(
        f"""
        <div style="background-color: #1E1E1E; padding: 8px 0px; border-bottom: 2px solid #333; margin-bottom: 15px;">
            <marquee behavior="scroll" direction="left" scrollamount="5" style="font-size: 16px;">
                {ticker_html}
            </marquee>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.title("🤖 Live Multi-Agent Trading Terminal")
st.caption("Automated Live Quantitative Analysis with Interconnected Specialized Agents (NSE, BSE & Global Benchmarks)")

# ==============================================================================
# YAHOO FINANCE LIVE TICKER SEARCH
# ==============================================================================
def get_real_ticker(company_name):
    """Searches Yahoo Finance's live database for the exact official ticker."""
    try:
        # Format the name safely for a URL (e.g., "State Bank" -> "State%20Bank")
        safe_name = urllib.parse.quote(company_name.strip())
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={safe_name}&quotesCount=1"
        
        # Disguise the request as a normal web browser to avoid getting blocked
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            # If Yahoo Finance finds a match, extract the exact symbol
            if 'quotes' in data and len(data['quotes']) > 0:
                return data['quotes'][0]['symbol'].upper()
    except Exception:
        pass
        
    # Fallback to whatever the user typed if the search somehow fails
    return company_name.upper().strip()

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "thread_config" not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": f"streamlit_live_session_{int(time.time())}"}}
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# TIME MACHINE STATE INITIALIZATION
if "state_history" not in st.session_state:
    st.session_state.state_history = []
if "history_index" not in st.session_state:
    st.session_state.history_index = 0

if "chat_history" not in st.session_state:
    if os.environ.get("GROQ_API_KEY"):
        start_msg = "✅ **System Ready.** API Key securely loaded. Enter an Indian stock, Global stock, or Index name (e.g., Nifty 50, Reliance, Apple) in the sidebar to begin."
    else:
        start_msg = "Welcome to the Terminal. Provide a Groq API key and asset name in the sidebar, then run the analysis."
    st.session_state.chat_history = [{"role": "assistant", "content": start_msg}]

# ==============================================================================
# SIDEBAR CONFIGURATIONS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Live Configurations")
    
    groq_key = st.text_input(
        "Groq API Key", 
        type="password", 
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Leave blank if you set this in your .env file!"
    )
    user_input_ticker = st.text_input("Target Stock, Index, or Company Name", value="NIFTY 50").strip()
    
    # Supported Exchanges Cheat-sheet UI
    with st.expander("💡 Supported Ticker Examples"):
        st.markdown("""
        **Indian Markets (NSE/BSE):**
        * `Nifty 50` ➔ `^NSEI`
        * `Sensex` ➔ `^BSESN`
        * `Reliance` ➔ `RELIANCE.NS`
        * `TCS` ➔ `TCS.NS`
        * `SBI (BSE)` ➔ `SBIN.BO`
        
        **Global Benchmarks & Equities:**
        * `S&P 500` ➔ `^GSPC`
        * `Nasdaq 100` ➔ `^NDX`
        * `Dow Jones` ➔ `^DJI`
        * `Apple` ➔ `AAPL`
        """)
        
    st.markdown("---")
    trigger_analysis = st.button("▶️ Run Multi-Agent Analysis", use_container_width=True)

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
    st.session_state["groq_api_key"] = groq_key

# ==============================================================================
# AGENT RUNTIME EXECUTION
# ==============================================================================
if trigger_analysis:
    active_key = groq_key or os.environ.get("GROQ_API_KEY")
    
    # 👉 THE NEW BULLETPROOF LOOKUP
    with st.spinner(f"🔍 Querying Yahoo Finance Database for '{user_input_ticker}'..."):
        resolved_ticker = get_real_ticker(user_input_ticker)
        
    if resolved_ticker != user_input_ticker.upper():
        st.toast(f"🎯 Database Match: Linked '{user_input_ticker}' to official ticker '{resolved_ticker}'")

    with st.spinner(f"Activating Specialized Agents for {resolved_ticker}..."):
        # 🚨 I DELETED THE AI GUESSER BLOCK THAT WAS CAUSING THE PROBLEM HERE 🚨
        
        st.session_state.thread_config = {"configurable": {"thread_id": f"streamlit_live_session_{int(time.time())}"}}
        st.session_state.graph_state = None
        st.session_state.state_history = []
        st.session_state.history_index = 0
        
        st.session_state.chat_history = [{"role": "assistant", "content": f"Initializing analysis for {resolved_ticker} across connected exchanges..."}]
        
        async def execute_phase_1():
            async with aiosqlite.connect("trading_agent_memory.db") as conn:
                saver = AsyncSqliteSaver(conn=conn)
                await saver.setup()
                
                local_app = build_interactive_trading_graph(saver)
                
                initial_state = {
                    "ticker": resolved_ticker,
                    "groq_api_key": active_key,
                    "user_query": f"Analyze {resolved_ticker}"
                }
                
                async def stream_logic():
                    async for _ in local_app.astream(initial_state, config=st.session_state.thread_config):
                        pass
                    return await local_app.aget_state(st.session_state.thread_config)
                
                task = asyncio.create_task(stream_logic())
                return await task

        final_snapshot = asyncio.run(execute_phase_1())
        
        st.session_state.state_history.append(final_snapshot.values)
        st.session_state.history_index = len(st.session_state.state_history) - 1
        st.session_state.graph_state = st.session_state.state_history[st.session_state.history_index]
        st.session_state.analysis_run = True
        
        rec_action = st.session_state.graph_state["final_recommendation"].action
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"🚨 **Analysis Complete for {resolved_ticker}**!\n\nProposed Action: **{rec_action}**.\nI have evaluated metrics from the relevant exchange ecosystem. Ask me anything!"
        })

# ==============================================================================
# SMART DEMO DETECTION
# ==============================================================================
is_demo_state = False
if st.session_state.graph_state:
    fund_insight = st.session_state.graph_state.get("agent_insights", {}).get("fundamental", "")
    if "[Demo]" in fund_insight:
        is_demo_state = True

# ==============================================================================
# SPLIT SCREEN UI ARCHITECTURE 
# ==============================================================================
terminal_col, chat_col = st.columns([2.2, 1.2], gap="large")

# ------------------------------------------------------------------------------
# LEFT SCREEN: MAIN TERMINAL
# ------------------------------------------------------------------------------
with terminal_col:
    st.subheader("📊 Execution Dashboard")
    
    if st.session_state.state_history and len(st.session_state.state_history) > 1:
        st.markdown("### ⏪ State Time Machine")
        h_col1, h_col2, h_col3 = st.columns([1, 2, 1])
        
        with h_col1:
            if st.button("⬅️ Back", disabled=st.session_state.history_index == 0, use_container_width=True):
                st.session_state.history_index -= 1
                st.session_state.graph_state = st.session_state.state_history[st.session_state.history_index]
                st.rerun()
        with h_col2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; color: gray;'><b>Viewing Phase {st.session_state.history_index + 1} of {len(st.session_state.state_history)}</b></div>", unsafe_allow_html=True)
        with h_col3:
            if st.button("Forward ➡️", disabled=st.session_state.history_index == len(st.session_state.state_history) - 1, use_container_width=True):
                st.session_state.history_index += 1
                st.session_state.graph_state = st.session_state.state_history[st.session_state.history_index]
                st.rerun()
        st.markdown("---")
    
    if st.session_state.analysis_run and st.session_state.graph_state:
        state = st.session_state.graph_state
        rec = state.get("final_recommendation")
        metrics = rec.performance_metrics
        live = state.get("live_data", {})
        
        display_ticker = state.get("ticker", user_input_ticker.upper())
        
        if is_demo_state and (groq_key or os.environ.get("GROQ_API_KEY")):
            st.info("🔑 **Live API Key Detected!** You are viewing cached Demo data. Click **▶️ Run Multi-Agent Analysis** to generate live AI insights.")
            
        col1, col2, col3, col4 = st.columns(4)
        
        # --- NEW: Dual Currency Logic ---
        usd_inr_rate = get_exchange_rate()
        current_price = live.get('current_price', 0.0)
        
        # 👉 ZERO CATCHER UI ALERT (Will hide zeros if YFinance returns nothing)
        if current_price == 0.0:
            st.error(f"🚨 **MARKET DATA UNAVAILABLE:** Yahoo Finance returned empty data for **`{display_ticker}`**.")
            st.warning("If this is an Indian stock, it may not be in the database. Please look up the exact Yahoo Finance symbol and type it directly into the search bar.")
        else:
            is_usd = ("^" in display_ticker and display_ticker not in ["^NSEI", "^BSESN"]) or ("." not in display_ticker and "^" not in display_ticker)
            
            if is_usd:
                price_usd = current_price
                price_inr = current_price * usd_inr_rate
                main_price = f"${price_usd:,.2f}"
                sub_price = f"₹{price_inr:,.2f} (INR)"
            else:
                price_inr = current_price
                price_usd = current_price / usd_inr_rate
                main_price = f"₹{price_inr:,.2f}"
                sub_price = f"${price_usd:,.2f} (USD)"
                
            # Using delta and delta_color="off" creates a clean, gray subtitle!
            col1.metric("Live Value", main_price, delta=sub_price, delta_color="off")
            col2.metric("Cumulative Return", f"{metrics.cumulative_return_pct:.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            col4.metric("Max Drawdown", f"{metrics.maximum_drawdown_pct:.2f}%")
            st.markdown("---")
            
            if not state.get("execution_receipt"):
                st.write(f"**Calculated Action Proposal:** `{rec.action}` | **Confidence:** `{rec.confidence_score*100:.1f}%`")
                st.info(f"**Model Strategy Summary:** {rec.supporting_rationale}")
                
                exec_col1, exec_col2 = st.columns(2)
                
                with exec_col1:
                    if st.button("✅ APPROVE & ROUTE ORDER", use_container_width=True, type="primary"):
                        async def run_approval():
                            async with aiosqlite.connect("trading_agent_memory.db") as conn:
                                saver = AsyncSqliteSaver(conn=conn)
                                await saver.setup()
                                local_app = build_interactive_trading_graph(saver)
                                await local_app.aupdate_state(st.session_state.thread_config, {"human_approval": True})
                                
                                async def stream_logic():
                                    async for _ in local_app.astream(None, config=st.session_state.thread_config): pass
                                    return await local_app.aget_state(st.session_state.thread_config)
                                return await asyncio.create_task(stream_logic())
                        
                        final_snapshot = asyncio.run(run_approval())
                        st.session_state.state_history.append(final_snapshot.values)
                        st.session_state.history_index = len(st.session_state.state_history) - 1
                        st.session_state.graph_state = st.session_state.state_history[st.session_state.history_index]
                        st.rerun()
                        
                with exec_col2:
                    if st.button("❌ REJECT & ABORT ORDER", use_container_width=True):
                        async def run_rejection():
                            async with aiosqlite.connect("trading_agent_memory.db") as conn:
                                saver = AsyncSqliteSaver(conn=conn)
                                await saver.setup()
                                local_app = build_interactive_trading_graph(saver)
                                await local_app.aupdate_state(st.session_state.thread_config, {"human_approval": False})
                                
                                async def stream_logic():
                                    async for _ in local_app.astream(None, config=st.session_state.thread_config): pass
                                    return await local_app.aget_state(st.session_state.thread_config)
                                return await asyncio.create_task(stream_logic())
                        
                        final_snapshot = asyncio.run(run_rejection())
                        st.session_state.state_history.append(final_snapshot.values)
                        st.session_state.history_index = len(st.session_state.state_history) - 1
                        st.session_state.graph_state = st.session_state.state_history[st.session_state.history_index]
                        st.rerun()
                        
            else:
                receipt = state.get("execution_receipt", {})
                if receipt.get("status") == "ORDER_FILLED":
                    st.success(f"✅ **EXCHANGE CLEARED COMPLETED**")
                    st.json(receipt)
                else:
                    st.error("🚨 **ORDER CANCELLED / REJECTED BY USER**")
                    st.json(receipt)
    else:
        st.info("👈 Enter an asset or index and click 'Run Multi-Agent Analysis' in the sidebar to begin.")

# ------------------------------------------------------------------------------
# RIGHT SCREEN: AI CHAT COMPANION
# ------------------------------------------------------------------------------
with chat_col:
    st.subheader("💬 Agent Chat Companion")
    
    chat_container = st.container(height=350, border=True)
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if prompt := st.chat_input("Ask about the analysis..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

# ------------------------------------------------------------------------------
# INVISIBLE CHATBOT PROCESSING
# ------------------------------------------------------------------------------
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    user_query = st.session_state.chat_history[-1]["content"]
    active_key = groq_key or os.environ.get("GROQ_API_KEY")
    
    llm = get_llm(st.session_state.graph_state if st.session_state.graph_state else {"groq_api_key": active_key})
    
    if not llm:
        response_text = "⚠️ Chatbot requires a valid Groq API key."
    else:
        context = ""
        if st.session_state.graph_state:
            state_insights = st.session_state.graph_state.get("agent_insights", {})
            context = f"""
            You have access to the following insights generated by your multi-agent trading system for ticker {st.session_state.graph_state.get('ticker')}:
            - Fundamental Analysis: {state_insights.get('fundamental', 'N/A')}
            - Technical Analysis: {state_insights.get('technical', 'N/A')}
            - News NLP Sentiment: {state_insights.get('news_sentiment', 'N/A')}
            - Macroeconomic Trend: {state_insights.get('macro_trend', 'N/A')}
            - Sector Supply Chain: {state_insights.get('supply_chain', 'N/A')}
            - External Search: {state_insights.get('external_search', 'N/A')}
            """
        
        system_prompt = f"You are the Master Interface Chatbot for a Multi-Agent Trading Terminal. {context} Answer the user professionally."
        
        try:
            res = llm.invoke([HumanMessage(content=f"{system_prompt}\n\nUser Question: {user_query}")])
            response_text = res.content
        except Exception as e:
            response_text = f"Error: {str(e)}"
            
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    st.rerun()