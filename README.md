# FinVest Pro - Systematic Multi-Asset Investment Platform

## FINS3645 FinTech Project 2026

A comprehensive investment platform offering systematically managed funds with news-sentiment analytics.

## Features

### Fund Offerings
- **Equity Funds** - Diversified portfolios of major tech stocks
- **Crypto Funds** - Bitcoin and Ethereum portfolios
- **Combined Funds** - Multi-asset portfolios spanning equities and crypto

### Optimization Methods
- Maximum Sharpe Ratio (Mean-Variance Tangency)
- Minimum Variance Portfolio
- Risk Parity
- Equal Weight (Baseline)

### Analytics
- Out-of-sample backtesting with rolling windows
- Real-time news sentiment analysis
- Growth of $1 visualizations
- Drawdown analysis
- Risk metrics (Sharpe, Calmar, Volatility)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd FINS3645_FinTech_Project

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

Or run directly with Streamlit:

```bash
streamlit run app/main.py
```

## Project Structure

```
FINS3645_FinTech_Project/
├── app/
│   └── main.py              # Streamlit application
├── src/
│   ├── __init__.py
│   ├── data_collection.py   # Market data & news fetching
│   ├── portfolio_optimization.py  # Portfolio optimization methods
│   ├── backtesting.py       # Out-of-sample backtesting
│   ├── sentiment_analysis.py # News sentiment analysis
│   └── factsheet.py         # Fund fact sheet generation
├── data/                    # Data storage
├── requirements.txt         # Python dependencies
├── run.py                   # Application runner
└── README.md               # This file
```

## Usage

1. **Dashboard** - Overview of all funds and their performance
2. **Fund Comparison** - Detailed comparison of different funds and strategies
3. **Sentiment Analytics** - Real-time market sentiment from news
4. **Invest** - Select a fund and make an investment

## Innovation Features

- **Custom Financial Lexicon** - Domain-specific sentiment analysis
- **Ensemble Sentiment** - Multiple NLP methods combined
- **Risk Parity Optimization** - Equal risk contribution portfolios
- **Interactive Visualizations** - Plotly-based charts and dashboards

## License

Academic use only - FINS3645 FinTech Project 2026
