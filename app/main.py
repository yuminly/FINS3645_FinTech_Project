import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import MarketDataCollector, NewsCollector
from src.portfolio_optimization import PortfolioOptimizer
from src.backtesting import Backtester
from src.sentiment_analysis import SentimentAnalyzer
from src.factsheet import FactSheetGenerator

st.set_page_config(
    page_title="FinVest Pro - Systematic Multi-Asset Investment Platform",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache market data."""
    collector = MarketDataCollector()
    
    equity_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V']
    crypto_symbols = ['BTC', 'ETH']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    equity_data = collector.fetch_equity_data(equity_tickers, start_date, end_date)
    crypto_data = collector.fetch_crypto_data(crypto_symbols, start_date, end_date)
    
    return equity_data, crypto_data, collector.get_risk_free_rate()


@st.cache_data
def optimize_portfolios(equity_returns, crypto_returns, combined_returns, rf):
    """Optimize all portfolios."""
    optimizer = PortfolioOptimizer(rf)
    
    results = {}
    
    equity_methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
    for method in equity_methods:
        weights, metrics = getattr(optimizer, f'optimize_{method}')(equity_returns)
        results[f'equity_{method}'] = {'weights': weights, 'metrics': metrics}
    
    for method in equity_methods:
        weights, metrics = getattr(optimizer, f'optimize_{method}')(crypto_returns)
        results[f'crypto_{method}'] = {'weights': weights, 'metrics': metrics}
    
    for method in equity_methods:
        weights, metrics = getattr(optimizer, f'optimize_{method}')(combined_returns)
        results[f'combined_{method}'] = {'weights': weights, 'metrics': metrics}
    
    return results


@st.cache_data
def run_backtests(equity_returns, crypto_returns, combined_returns, rf):
    """Run backtests for all strategies."""
    backtester = Backtester(rf)
    
    methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
    
    equity_backtests = backtester.compare_strategies(equity_returns, methods)
    crypto_backtests = backtester.compare_strategies(crypto_returns, methods)
    combined_backtests = backtester.compare_strategies(combined_returns, methods)
    
    return equity_backtests, crypto_backtests, combined_backtests


@st.cache_data
def analyze_sentiment():
    """Analyze news sentiment."""
    collector = NewsCollector()
    analyzer = SentimentAnalyzer()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    news = collector.fetch_news(
        "stock market OR cryptocurrency OR economy",
        start_date,
        end_date
    )
    
    sentiment_df = analyzer.analyze_news_batch(news)
    sentiment_index = analyzer.build_sentiment_index(sentiment_df)
    sentiment_summary = analyzer.get_sentiment_summary(sentiment_df)
    
    return sentiment_df, sentiment_index, sentiment_summary


def main():
    st.sidebar.title("FinVest Pro")
    st.sidebar.markdown("**Systematic Multi-Asset Investment Platform**")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Fund Comparison", "Sentiment Analytics", "Invest"]
    )
    
    equity_data, crypto_data, rf = load_data()
    
    equity_returns = np.log(equity_data / equity_data.shift(1)).dropna()
    crypto_returns = np.log(crypto_data / crypto_data.shift(1)).dropna()
    
    combined_prices = pd.concat([equity_data, crypto_data], axis=1)
    combined_returns = np.log(combined_prices / combined_prices.shift(1)).dropna()
    
    portfolio_results = optimize_portfolios(
        equity_returns, crypto_returns, combined_returns, rf
    )
    
    equity_backtests, crypto_backtests, combined_backtests = run_backtests(
        equity_returns, crypto_returns, combined_returns, rf
    )
    
    if page == "Dashboard":
        render_dashboard(
            portfolio_results, equity_backtests, 
            crypto_backtests, combined_backtests
        )
    elif page == "Fund Comparison":
        render_fund_comparison(
            portfolio_results, equity_backtests,
            crypto_backtests, combined_backtests
        )
    elif page == "Sentiment Analytics":
        render_sentiment_analytics()
    elif page == "Invest":
        render_invest(portfolio_results)


def render_dashboard(portfolio_results, equity_bt, crypto_bt, combined_bt):
    st.title("📈 FinVest Pro Dashboard")
    
    st.markdown("""
    Welcome to FinVest Pro, your systematic multi-asset investment platform.
    Browse our fund offerings and invest in professionally managed portfolios.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Equity Fund (Max Sharpe)",
            f"{portfolio_results['equity_max_sharpe']['metrics']['annual_return']:.2%}",
            f"Sharpe: {portfolio_results['equity_max_sharpe']['metrics']['sharpe_ratio']:.2f}"
        )
    
    with col2:
        st.metric(
            "Crypto Fund (Max Sharpe)",
            f"{portfolio_results['crypto_max_sharpe']['metrics']['annual_return']:.2%}",
            f"Sharpe: {portfolio_results['crypto_max_sharpe']['metrics']['sharpe_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            "Combined Fund (Max Sharpe)",
            f"{portfolio_results['combined_max_sharpe']['metrics']['annual_return']:.2%}",
            f"Sharpe: {portfolio_results['combined_max_sharpe']['metrics']['sharpe_ratio']:.2f}"
        )
    
    st.subheader("Fund Performance Overview")
    
    fig = go.Figure()
    
    funds = {
        'Equity Fund': equity_bt['max_sharpe'],
        'Crypto Fund': crypto_bt['max_sharpe'],
        'Combined Fund': combined_bt['max_sharpe']
    }
    
    for name, data in funds.items():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cumulative_return'],
            name=name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Growth of $1 - All Funds",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_fund_comparison(portfolio_results, equity_bt, crypto_bt, combined_bt):
    st.title("📊 Fund Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fund_type = st.selectbox(
            "Select Fund Type",
            ["Equity", "Crypto", "Combined"]
        )
    
    with col2:
        strategy = st.selectbox(
            "Select Strategy",
            ["Maximum Sharpe", "Minimum Variance", "Risk Parity", "Equal Weight"]
        )
    
    strategy_map = {
        "Maximum Sharpe": "max_sharpe",
        "Minimum Variance": "min_variance",
        "Risk Parity": "risk_parity",
        "Equal Weight": "equal_weight"
    }
    
    fund_key = f"{fund_type.lower()}_{strategy_map[strategy]}"
    backtest_key = strategy_map[strategy]
    
    metrics = portfolio_results[fund_key]['metrics']
    weights = portfolio_results[fund_key]['weights']
    
    if fund_type == "Equity":
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V']
        backtest_data = equity_bt[backtest_key]
    elif fund_type == "Crypto":
        assets = ['BTC-USD', 'ETH-USD']
        backtest_data = crypto_bt[backtest_key]
    else:
        assets = list(equity_data.columns) + list(crypto_data.columns)
        backtest_data = combined_bt[backtest_key]
    
    st.subheader(f"{fund_type} Fund - {strategy}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Return", f"{metrics['annual_return']:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{metrics['annual_volatility']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=metrics['cumulative_returns'].index,
            y=metrics['cumulative_returns'].values,
            name='Fund',
            line=dict(width=2)
        ))
        fig_growth.update_layout(
            title="Growth of $1",
            template="plotly_white"
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with col2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=metrics['drawdowns'].index,
            y=metrics['drawdowns'].values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1)
        ))
        fig_dd.update_layout(
            title="Drawdowns",
            template="plotly_white"
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    
    st.subheader("Current Holdings")
    
    holdings_df = pd.DataFrame({
        'Asset': assets,
        'Weight': weights
    })
    holdings_df = holdings_df[holdings_df['Weight'] > 0.01]
    
    fig_holdings = go.Figure(data=[go.Pie(
        labels=holdings_df['Asset'],
        values=holdings_df['Weight'],
        hole=0.3
    )])
    fig_holdings.update_layout(title="Portfolio Allocation")
    st.plotly_chart(fig_holdings, use_container_width=True)


def render_sentiment_analytics():
    st.title("📰 News Sentiment Analytics")
    
    sentiment_df, sentiment_index, sentiment_summary = analyze_sentiment()
    
    st.subheader("Market Sentiment Index")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sentiment_index.index,
        y=sentiment_index['sentiment_score'],
        name='Daily Sentiment',
        line=dict(color='blue', width=1),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=sentiment_index.index,
        y=sentiment_index['sentiment_ma7'],
        name='7-Day MA',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sentiment_index.index,
        y=sentiment_index['sentiment_ma30'],
        name='30-Day MA',
        line=dict(color='green', width=2)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Market Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Sentiment", f"{sentiment_summary['mean_sentiment']:.3f}")
    with col2:
        st.metric("Positive", f"{sentiment_summary['pct_positive']:.1%}")
    with col3:
        st.metric("Negative", f"{sentiment_summary['pct_negative']:.1%}")
    with col4:
        st.metric("Neutral", f"{sentiment_summary['pct_neutral']:.1%}")
    
    st.subheader("Recent News Headlines")
    
    recent_news = sentiment_df.head(20)
    
    for _, row in recent_news.iterrows():
        sentiment_color = "green" if row['sentiment_score'] > 0.3 else "red" if row['sentiment_score'] < -0.3 else "gray"
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {sentiment_color}; margin-bottom: 10px;">
            <strong>{row['headline']}</strong><br>
            <small>Sentiment: {row['sentiment_label']} ({row['sentiment_score']:.3f})</small>
        </div>
        """, unsafe_allow_html=True)


def render_invest(portfolio_results):
    st.title("💰 Invest")
    
    st.markdown("""
    Select a fund and allocate your investment. FinVest Pro manages your portfolio
    systematically using quantitative optimization methods.
    """)
    
    investment_amount = st.number_input(
        "Investment Amount ($)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=100
    )
    
    st.subheader("Select Fund")
    
    fund_options = {
        "Equity Fund - Maximum Sharpe": "equity_max_sharpe",
        "Equity Fund - Minimum Variance": "equity_min_variance",
        "Equity Fund - Risk Parity": "equity_risk_parity",
        "Crypto Fund - Maximum Sharpe": "crypto_max_sharpe",
        "Crypto Fund - Minimum Variance": "crypto_min_variance",
        "Crypto Fund - Risk Parity": "crypto_risk_parity",
        "Combined Fund - Maximum Sharpe": "combined_max_sharpe",
        "Combined Fund - Minimum Variance": "combined_min_variance",
        "Combined Fund - Risk Parity": "combined_risk_parity"
    }
    
    selected_fund = st.selectbox("Choose a fund", list(fund_options.keys()))
    
    fund_key = fund_options[selected_fund]
    metrics = portfolio_results[fund_key]['metrics']
    weights = portfolio_results[fund_key]['weights']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Annual Return", f"{metrics['annual_return']:.2%}")
    with col2:
        st.metric("Risk (Volatility)", f"{metrics['annual_volatility']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    
    st.subheader("Portfolio Allocation")
    
    if "equity" in fund_key:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V']
    elif "crypto" in fund_key:
        assets = ['BTC', 'ETH']
    else:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'BTC', 'ETH']
    
    allocations = {}
    for i, asset in enumerate(assets):
        if weights[i] > 0.01:
            allocations[asset] = investment_amount * weights[i]
    
    alloc_df = pd.DataFrame([
        {'Asset': k, 'Allocation ($)': f"${v:,.2f}", 'Weight': f"{v/investment_amount:.1%}"}
        for k, v in allocations.items()
    ])
    
    st.dataframe(alloc_df, use_container_width=True)
    
    st.subheader("Investment Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Investment Amount:** ${investment_amount:,.2f}
        
        **Expected Annual Return:** ${investment_amount * metrics['annual_return']:,.2f}
        
        **Risk Level:** {metrics['annual_volatility']:.2%} volatility
        """)
    
    with col2:
        st.markdown(f"""
        **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}
        
        **Max Drawdown:** {metrics['max_drawdown']:.2%}
        
        **Management Fee:** 1.5% p.a.
        """)
    
    if st.button("Confirm Investment", type="primary"):
        st.success(f"""
        🎉 Investment Confirmed!
        
        You have invested ${investment_amount:,.2f} in **{selected_fund}**.
        
        Your portfolio will be managed systematically and rebalanced periodically.
        """)
        
        st.balloons()


if __name__ == "__main__":
    main()
