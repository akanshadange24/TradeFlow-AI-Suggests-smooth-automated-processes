# TradeFlow-AI: Multi-Agent Trading Terminal

### 🔗 Project Links

* **[Live Trading Terminal](https://tradeflow-ai-suggests-smooth-automated-processes-tufczhjjlkrsn.streamlit.app/)**
* **[View Execution Trace](https://smith.langchain.com/public/4a6997ab-610f-4494-83a3-fdcfed5d3390/r)**

A modular, event-driven trading framework built using LangGraph...

# Autonomous Agentic Trading System

An intelligent, stateful, multi-agent framework designed to perform quantitative market analysis and provide actionable trading recommendations with a human-in-the-loop validation layer.

## 🏗️ System Architecture

This system utilizes a **State Machine (LangGraph)** architecture, where specialized agents (ISAs) function as modular nodes.

* **Data Fetcher**: Ingests real-time market data and historical returns.
* **Intellectual Specialized Agents (ISAs)**:
* **Fundamental**: Analyzes P/E ratios and debt/equity.
* **Technical**: Provides momentum and trend outlooks.
* **News NLP**: Performs sentiment analysis on market headlines.
* **Macro**: Assesses S&P 500 trends and economic impact.
* **Supply Chain**: Identifies sector-specific constraints.
* **External Search**: Integrates institutional/hedge fund chatter via DuckDuckGo.


* **Synthesizer Node**: Aggregates signals using a decisive LLM policy to output a Buy/Sell/Hold recommendation with a dynamic confidence score.
* **Human-in-the-Loop (HITL)**: A deterministic breakpoint ensuring no trade is executed without manual approval.

## 🚀 Key Features

* **Stateful Memory**: Uses `AsyncSqliteSaver` to maintain context throughout the agent's decision-making process.
* **Asynchronous Execution**: Built on `asyncio` and `aiosqlite` for non-blocking I/O, allowing concurrent agent analysis.
* **Production-Ready Patterns**: Implements `tenacity` for exponential backoff retries, ensuring resilience against API rate limits.
* **Dynamic Risk Evaluation**: Calculates Sharpe Ratio, Annualized Return, and Maximum Drawdown in real-time.

## 🛠️ Technology Stack

* **Orchestration**: [LangGraph](https://www.langchain.com/langgraph)
* **LLM Engine**: [Llama 3.1 8B](https://groq.com) via Groq API
* **Data Sources**: `yfinance`, DuckDuckGo Search
* **Database**: `aiosqlite`
* **Frontend/Deployment**: Streamlit

## ⚙️ How to Deploy

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ai-trading-agent.git

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Set environment variables:**
Add your `GROQ_API_KEY` to your deployment settings.
4. **Deploy:** Connect your GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/).

## 📈 Future Roadmap

* [ ] Migrate from SQLite to PostgreSQL for distributed state persistence.
* [ ] Add support for multi-asset portfolio rebalancing.

---

