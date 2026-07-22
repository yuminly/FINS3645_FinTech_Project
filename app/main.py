import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.portfolio_optimization import PortfolioOptimizer
from src.backtesting import Backtester
from src.sentiment_analysis import SentimentAnalyzer

st.set_page_config(
    page_title="FinVest Pro - Systematic Multi-Asset Investment Platform",
    page_icon="📈",
    layout="wide"
)

# Asset universe matching Part A
EQUITY_TICKERS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'KO']
EQUITY_SECTORS = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'JPM': 'Financials',
    'XOM': 'Energy',
    'KO': 'Consumer'
}
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE']
RF_ANNUAL = 0.04


@st.cache_data
def generate_data():
    """
    Generate synthetic market data matching Part A exactly.
    Uses simple returns (pct_change), not log returns.
    """
    np.random.seed(42)

    dates = pd.bdate_range(start='2022-01-03', end='2024-12-31', freq='B')
    n_days = len(dates)

    stocks = {
        'AAPL': {'base_price': 170, 'drift': 0.12, 'vol': 0.25},
        'MSFT': {'base_price': 330, 'drift': 0.15, 'vol': 0.22},
        'JPM':  {'base_price': 140, 'drift': 0.08, 'vol': 0.28},
        'XOM':  {'base_price': 90, 'drift': 0.10, 'vol': 0.32},
        'KO':   {'base_price': 60, 'drift': 0.06, 'vol': 0.18}
    }

    cryptos = {
        'BTC': {'base_price': 42000, 'drift': 0.25, 'vol': 0.65},
        'ETH': {'base_price': 2200, 'drift': 0.30, 'vol': 0.75},
        'SOL': {'base_price': 100, 'drift': 0.40, 'vol': 0.90},
        'ADA': {'base_price': 0.50, 'drift': 0.20, 'vol': 0.85},
        'DOGE': {'base_price': 0.10, 'drift': 0.15, 'vol': 1.00}
    }

    stock_corr = np.array([
        [1.00, 0.75, 0.40, 0.20, 0.30],
        [0.75, 1.00, 0.45, 0.25, 0.35],
        [0.40, 0.45, 1.00, 0.50, 0.40],
        [0.20, 0.25, 0.50, 1.00, 0.35],
        [0.30, 0.35, 0.40, 0.35, 1.00]
    ])

    crypto_corr = np.array([
        [1.00, 0.85, 0.70, 0.65, 0.55],
        [0.85, 1.00, 0.75, 0.70, 0.60],
        [0.70, 0.75, 1.00, 0.65, 0.55],
        [0.65, 0.70, 0.65, 1.00, 0.50],
        [0.55, 0.60, 0.55, 0.50, 1.00]
    ])

    def gen_corr_returns(n, corr, drifts, vols):
        L = np.linalg.cholesky(corr)
        indep = np.random.randn(n, len(drifts))
        return drifts + indep @ L.T * vols

    stock_drifts = np.array([s['drift'] / 252 for s in stocks.values()])
    stock_vols = np.array([s['vol'] / np.sqrt(252) for s in stocks.values()])
    stock_rets = gen_corr_returns(n_days, stock_corr, stock_drifts, stock_vols)

    crypto_drifts = np.array([c['drift'] / 252 for c in cryptos.values()])
    crypto_vols = np.array([c['vol'] / np.sqrt(252) for c in cryptos.values()])
    crypto_rets = gen_corr_returns(n_days, crypto_corr, crypto_drifts, crypto_vols)

    def rets_to_prices(rets, bases, tickers):
        p = pd.DataFrame(index=dates)
        for i, t in enumerate(tickers):
            p[t] = bases[i] * np.exp(np.cumsum(rets[:, i]))
        return p

    stock_prices = rets_to_prices(stock_rets, [s['base_price'] for s in stocks.values()], list(stocks.keys()))
    crypto_prices = rets_to_prices(crypto_rets, [c['base_price'] for c in cryptos.values()], list(cryptos.keys()))

    # AAPL 4:1 split on 2023-06-15
    split_date = pd.Timestamp('2023-06-15')
    stock_prices.loc[stock_prices.index < split_date, 'AAPL'] *= 4.0

    # SIMPLE returns (pct_change), matching Part A
    stock_returns = stock_prices.pct_change(fill_method=None).dropna()
    crypto_returns = crypto_prices.pct_change(fill_method=None).dropna()
    combined_returns = pd.concat([stock_returns, crypto_returns], axis=1).dropna()

    return stock_prices, crypto_prices, stock_returns, crypto_returns, combined_returns


@st.cache_data
def optimize_all_portfolios(stock_returns, crypto_returns, combined_returns):
    """Optimise all portfolios matching Part A."""
    optimizer = PortfolioOptimizer(risk_free_rate=RF_ANNUAL)
    results = {}

    for name, rets in [('equity', stock_returns), ('crypto', crypto_returns), ('combined', combined_returns)]:
        for method in ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']:
            fn = getattr(optimizer, f'optimize_{method}')
            w, m = fn(rets)
            results[f'{name}_{method}'] = {'weights': w, 'metrics': m}

    return results


@st.cache_data
def run_all_backtests(stock_returns, crypto_returns, combined_returns):
    """Run OOS backtests matching Part A."""
    bt = Backtester(risk_free_rate=RF_ANNUAL)
    methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
    results = {}
    for name, rets in [('equity', stock_returns), ('crypto', crypto_returns), ('combined', combined_returns)]:
        for method in methods:
            results[f'{name}_{method}'] = bt.rolling_backtest(rets, lookback_days=252, rebalance_freq=21, method=method)
    return results


@st.cache_data
def generate_sentiment_data():
    """Generate synthetic sentiment data matching Part A (5 sectors)."""
    np.random.seed(123)
    dates = pd.bdate_range(start='2022-01-03', end='2024-12-31', freq='B')
    n_days = len(dates)
    sectors = ['Technology', 'Financials', 'Energy', 'Consumer', 'Technology']
    tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'KO']

    records = []
    for _ in range(5000):
        si = np.random.randint(0, len(sectors))
        di = np.random.randint(0, n_days)
        s = np.clip(np.random.normal(0.05, 0.3), -1, 1)
        records.append({
            'date': dates[di],
            'sector': sectors[si],
            'ticker': tickers[si],
            'sentiment_score': s,
            'sentiment_label': 'Positive' if s > 0.1 else ('Negative' if s < -0.1 else 'Neutral'),
            'confidence': abs(s)
        })

    return pd.DataFrame(records)


def main():
    st.sidebar.title("FinVest Pro")
    st.sidebar.markdown("**Systematic Multi-Asset Investment Platform**")

    page = st.sidebar.radio("Navigation", ["Dashboard", "Fund Comparison", "Sentiment Analytics", "Invest"])

    stock_prices, crypto_prices, stock_returns, crypto_returns, combined_returns = generate_data()
    portfolio_results = optimize_all_portfolios(stock_returns, crypto_returns, combined_returns)
    backtest_results = run_all_backtests(stock_returns, crypto_returns, combined_returns)

    if page == "Dashboard":
        render_dashboard(portfolio_results, backtest_results)
    elif page == "Fund Comparison":
        render_fund_comparison(portfolio_results, backtest_results)
    elif page == "Sentiment Analytics":
        render_sentiment_analytics()
    elif page == "Invest":
        render_invest(portfolio_results)


def render_dashboard(portfolio_results, bt):
    st.title("📈 FinVest Pro Dashboard")

    st.markdown("""
    Welcome to FinVest Pro, your systematic multi-asset investment platform.
    Browse our fund offerings and invest in professionally managed portfolios.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        m = portfolio_results['equity_max_sharpe']['metrics']
        st.metric("Equity Fund (Max Sharpe)", f"{m['annual_return']:.2%}", f"Sharpe: {m['sharpe_ratio']:.2f}")

    with col2:
        m = portfolio_results['crypto_max_sharpe']['metrics']
        st.metric("Crypto Fund (Max Sharpe)", f"{m['annual_return']:.2%}", f"Sharpe: {m['sharpe_ratio']:.2f}")

    with col3:
        m = portfolio_results['combined_max_sharpe']['metrics']
        st.metric("Combined Fund (Max Sharpe)", f"{m['annual_return']:.2%}", f"Sharpe: {m['sharpe_ratio']:.2f}")

    st.subheader("Out-of-Sample Fund Performance")

    fig = go.Figure()
    funds = {
        'Equity Fund': bt['equity_max_sharpe'],
        'Crypto Fund': bt['crypto_max_sharpe'],
        'Combined Fund': bt['combined_max_sharpe']
    }

    for name, data in funds.items():
        fig.add_trace(go.Scatter(x=data.index, y=data['cumulative_return'], name=name, line=dict(width=2)))

    fig.update_layout(title="Growth of $1 — All Funds (OOS)", xaxis_title="Date", yaxis_title="Value ($)",
                      hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy Comparison (Combined Fund)")
    fig2 = go.Figure()
    for method, label in [('max_sharpe', 'Max Sharpe'), ('min_variance', 'Min Variance'),
                          ('risk_parity', 'Risk Parity'), ('equal_weight', 'Equal Weight')]:
        data = bt[f'combined_{method}']
        fig2.add_trace(go.Scatter(x=data.index, y=data['cumulative_return'], name=label, line=dict(width=1.5)))

    fig2.update_layout(title="Combined Fund — All Strategies (OOS)", xaxis_title="Date", yaxis_title="Value ($)",
                       hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)


def render_fund_comparison(portfolio_results, bt):
    st.title("📊 Fund Comparison")

    col1, col2 = st.columns(2)
    with col1:
        fund_type = st.selectbox("Select Fund Type", ["Equity", "Crypto", "Combined"])
    with col2:
        strategy = st.selectbox("Select Strategy", ["Maximum Sharpe", "Minimum Variance", "Risk Parity", "Equal Weight"])

    strategy_map = {"Maximum Sharpe": "max_sharpe", "Minimum Variance": "min_variance",
                    "Risk Parity": "risk_parity", "Equal Weight": "equal_weight"}

    fund_key = f"{fund_type.lower()}_{strategy_map[strategy]}"
    metrics = portfolio_results[fund_key]['metrics']
    weights = portfolio_results[fund_key]['weights']

    if fund_type == "Equity":
        assets = EQUITY_TICKERS
    elif fund_type == "Crypto":
        assets = CRYPTO_SYMBOLS
    else:
        assets = EQUITY_TICKERS + CRYPTO_SYMBOLS

    st.subheader(f"{fund_type} Fund — {strategy}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Return", f"{metrics['annual_return']:.2%}")
    c2.metric("Annual Volatility", f"{metrics['annual_volatility']:.2%}")
    c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=metrics['cumulative_returns'].index, y=metrics['cumulative_returns'].values,
                                   name='Fund', line=dict(width=2)))
        fig_g.update_layout(title="Growth of $1 (In-Sample)", template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

    with c2:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=metrics['drawdowns'].index, y=metrics['drawdowns'].values,
                                   fill='tozeroy', name='Drawdown', line=dict(color='red', width=1)))
        fig_d.update_layout(title="Drawdowns (In-Sample)", template="plotly_white")
        st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Out-of-Sample Backtest")
    bt_data = bt[fund_key]
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_data.index, y=bt_data['cumulative_return'], name='OOS Return', line=dict(width=2)))
    fig_bt.update_layout(title="Out-of-Sample Growth of $1", xaxis_title="Date", yaxis_title="Value ($)",
                         template="plotly_white")
    st.plotly_chart(fig_bt, use_container_width=True)

    st.subheader("Current Holdings")
    holdings = pd.DataFrame({'Asset': assets, 'Weight': weights})
    holdings = holdings[holdings['Weight'] > 0.01]

    fig_pie = go.Figure(data=[go.Pie(labels=holdings['Asset'], values=holdings['Weight'], hole=0.3)])
    fig_pie.update_layout(title="Portfolio Allocation")
    st.plotly_chart(fig_pie, use_container_width=True)


def render_sentiment_analytics():
    st.title("📰 News Sentiment Analytics")

    sentiment_df = generate_sentiment_data()

    st.subheader("Sector Sentiment Overview")

    fig = go.Figure()
    sector_colors = {'Technology': '#08519c', 'Financials': '#cb181d', 'Energy': '#d94701', 'Consumer': '#238b45'}

    for sector in sentiment_df['sector'].unique():
        sd = sentiment_df[sentiment_df['sector'] == sector]
        daily = sd.groupby('date')['sentiment_score'].mean()
        ma7 = daily.rolling(7).mean()
        fig.add_trace(go.Scatter(x=ma7.index, y=ma7.values, name=sector,
                                 line=dict(color=sector_colors.get(sector, '#636363'), width=1.5)))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="7-Day MA Sentiment by Sector", xaxis_title="Date", yaxis_title="Sentiment Score",
                      template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Sentiment", f"{sentiment_df['sentiment_score'].mean():.3f}")
    c2.metric("Positive", f"{(sentiment_df['sentiment_label'] == 'Positive').mean():.1%}")
    c3.metric("Negative", f"{(sentiment_df['sentiment_label'] == 'Negative').mean():.1%}")
    c4.metric("Neutral", f"{(sentiment_df['sentiment_label'] == 'Neutral').mean():.1%}")

    st.subheader("Sector Sentiment Correlation")
    pivot = sentiment_df.pivot_table(values='sentiment_score', index='date', columns='sector').rolling(21).mean()
    corr = pivot.corr()

    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                          colorscale='RdBu_r', zmin=-1, zmax=1, text=corr.round(2).values,
                                          texttemplate='%{text}', textfont=dict(size=11)))
    fig_corr.update_layout(title="Cross-Sector Sentiment Correlation (60-day rolling)", template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Recent Headlines")
    recent = sentiment_df.sort_values('date', ascending=False).head(20)
    for _, row in recent.iterrows():
        color = "green" if row['sentiment_score'] > 0.1 else ("red" if row['sentiment_score'] < -0.1 else "gray")
        st.markdown(f"""
        <div style="padding: 8px; border-left: 4px solid {color}; margin-bottom: 8px;">
            <strong>{row['ticker']}</strong> ({row['sector']}) — {row['date'].strftime('%Y-%m-%d')}<br>
            Sentiment: {row['sentiment_label']} ({row['sentiment_score']:.3f})
        </div>
        """, unsafe_allow_html=True)


def render_invest(portfolio_results):
    st.title("💰 Invest")

    st.markdown("""
    Select a fund and allocate your investment. FinVest Pro manages your portfolio
    systematically using quantitative optimisation methods.
    """)

    investment_amount = st.number_input("Investment Amount ($)", min_value=100, max_value=1000000, value=10000, step=100)

    fund_options = {}
    for ft in ['Equity', 'Crypto', 'Combined']:
        for strat in ['Maximum Sharpe', 'Minimum Variance', 'Risk Parity']:
            key = f"{ft} Fund — {strat}"
            fund_options[key] = f"{ft.lower()}_{strat.lower().replace(' ', '_')}"

    selected = st.selectbox("Select Fund", list(fund_options.keys()))
    fund_key = fund_options[selected]
    metrics = portfolio_results[fund_key]['metrics']
    weights = portfolio_results[fund_key]['weights']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Annual Return", f"{metrics['annual_return']:.2%}")
    c2.metric("Risk (Volatility)", f"{metrics['annual_volatility']:.2%}")
    c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    if 'equity' in fund_key:
        assets = EQUITY_TICKERS
    elif 'crypto' in fund_key:
        assets = CRYPTO_SYMBOLS
    else:
        assets = EQUITY_TICKERS + CRYPTO_SYMBOLS

    allocations = {a: investment_amount * w for a, w in zip(assets, weights) if w > 0.01}

    alloc_df = pd.DataFrame([{'Asset': k, 'Allocation ($)': f"${v:,.2f}", 'Weight': f"{v/investment_amount:.1%}"}
                             for k, v in allocations.items()])
    st.dataframe(alloc_df, use_container_width=True)

    st.subheader("Investment Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Investment Amount:** ${investment_amount:,.2f}
        **Expected Annual Return:** ${investment_amount * metrics['annual_return']:,.2f}
        **Risk Level:** {metrics['annual_volatility']:.2%} volatility
        """)
    with c2:
        st.markdown(f"""
        **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}
        **Max Drawdown:** {metrics['max_drawdown']:.2%}
        **Management Fee:** 1.5% p.a.
        """)

    if st.button("Confirm Investment", type="primary"):
        st.success(f"🎉 Investment Confirmed! You have invested ${investment_amount:,.2f} in **{selected}**.")
        st.balloons()


if __name__ == "__main__":
    main()
