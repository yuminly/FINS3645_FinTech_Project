"""
FINS3645 FinTech Project Part A - Main Report Runner
Generates all data, runs optimisations, and produces FT-style figures
"""

import sys
sys.path.insert(0, '/Users/yumeme/Documents/GitHub/FINS3645_FinTech_Project')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Import classes directly to avoid relative import issues
# We define them inline to keep the report self-contained
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.04):
        self.rf = risk_free_rate

    def optimize_max_sharpe(self, returns, constraints=None):
        n = returns.shape[1]
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        def neg_sharpe(w):
            pr = np.dot(w, mean_returns)
            pv = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(pr - self.rf) / pv
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(neg_sharpe, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
        return res.x, self._metrics(res.x, returns)

    def optimize_min_variance(self, returns, constraints=None):
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        def pv(w):
            return np.dot(w.T, np.dot(cov_matrix, w))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(pv, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
        return res.x, self._metrics(res.x, returns)

    def optimize_risk_parity(self, returns, constraints=None):
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        def rc_error(w):
            pv = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            mc = np.dot(cov_matrix, w)
            rc = w * mc / pv
            target = pv / n
            return np.sum((rc - target) ** 2)
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 1) for _ in range(n))
        res = minimize(rc_error, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
        return res.x, self._metrics(res.x, returns)

    def optimize_equal_weight(self, returns):
        n = returns.shape[1]
        w = np.array([1/n]*n)
        return w, self._metrics(w, returns)

    def _metrics(self, w, returns):
        pr = returns.dot(w)
        tr = (1 + pr).prod() - 1
        ar = (1 + tr) ** (252 / len(pr)) - 1
        av = pr.std() * np.sqrt(252)
        sh = (ar - self.rf) / av
        cum = (1 + pr).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        return {
            'weights': w, 'annual_return': ar, 'annual_volatility': av,
            'sharpe_ratio': sh, 'max_drawdown': dd.min(),
            'total_return': tr, 'cumulative_returns': cum, 'drawdowns': dd
        }

    def get_efficient_frontier(self, returns, n_points=100):
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        targets = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
        vols = []
        for t in targets:
            def pvol(w):
                return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x, tt=t: np.dot(x, mean_returns) - tt}]
            bounds = tuple((0, 1) for _ in range(n))
            r = minimize(pvol, np.array([1/n]*n), method='SLSQP', bounds=bounds, constraints=cons)
            vols.append(r.fun if r.success else np.nan)
        return targets, np.array(vols)


class Backtester:
    def __init__(self, risk_free_rate=0.04):
        self.rf = risk_free_rate
        self.optimizer = PortfolioOptimizer(risk_free_rate)

    def rolling_backtest(self, returns, lookback_days=252, rebalance_freq=21, method='max_sharpe'):
        n_assets = returns.shape[1]
        dates = returns.index[lookback_days:]
        port_rets = []
        for i in range(0, len(dates), rebalance_freq):
            if i + rebalance_freq > len(dates):
                break
            train = returns.iloc[i:i+lookback_days]
            test = returns.iloc[i+lookback_days:i+lookback_days+rebalance_freq]
            w = self._get_weights(train, method)
            for j in range(len(test)):
                port_rets.append({'date': test.index[j], 'return': np.dot(w, test.iloc[j])})
        df = pd.DataFrame(port_rets).set_index('date')
        cum = (1 + df['return']).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        return pd.DataFrame({'return': df['return'], 'cumulative_return': cum, 'drawdown': dd,
                             'rolling_volatility': df['return'].rolling(21).std() * np.sqrt(252)})

    def _get_weights(self, returns, method):
        if method == 'max_sharpe':
            w, _ = self.optimizer.optimize_max_sharpe(returns)
        elif method == 'min_variance':
            w, _ = self.optimizer.optimize_min_variance(returns)
        elif method == 'risk_parity':
            w, _ = self.optimizer.optimize_risk_parity(returns)
        elif method == 'equal_weight':
            w, _ = self.optimizer.optimize_equal_weight(returns)
        return w

    def calculate_statistics(self, bt):
        r = bt['return']
        c = bt['cumulative_return']
        tr = c.iloc[-1] - 1
        ar = (1 + tr) ** (252 / len(r)) - 1
        av = r.std() * np.sqrt(252)
        sh = (ar - self.rf) / av
        mdd = bt['drawdown'].min()
        calmar = ar / abs(mdd) if mdd != 0 else 0
        return {'Total Return': f"{tr:.2%}", 'Annual Return': f"{ar:.2%}",
                'Annual Volatility': f"{av:.2%}", 'Sharpe Ratio': f"{sh:.2f}",
                'Max Drawdown': f"{mdd:.2%}", 'Calmar Ratio': f"{calmar:.2f}"}


# =============================================================================
# STAGE 1: Data Generation (ETL)
# =============================================================================

print("=" * 60)
print("STAGE 1: Data Generation (ETL)")
print("=" * 60)

np.random.seed(42)

# Generate synthetic market data
dates = pd.bdate_range(start='2022-01-03', end='2024-12-31', freq='B')
n_days = len(dates)

# Stock tickers with sector mapping
stocks = {
    'AAPL': {'sector': 'Technology', 'base_price': 170, 'drift': 0.12, 'vol': 0.25},
    'MSFT': {'sector': 'Technology', 'base_price': 330, 'drift': 0.15, 'vol': 0.22},
    'JPM':  {'sector': 'Financials', 'base_price': 140, 'drift': 0.08, 'vol': 0.28},
    'XOM':  {'sector': 'Energy', 'base_price': 90, 'drift': 0.10, 'vol': 0.32},
    'KO':   {'sector': 'Consumer', 'base_price': 60, 'drift': 0.06, 'vol': 0.18}
}

# Crypto tickers
cryptos = {
    'BTC': {'base_price': 42000, 'drift': 0.25, 'vol': 0.65},
    'ETH': {'base_price': 2200, 'drift': 0.30, 'vol': 0.75},
    'SOL': {'base_price': 100, 'drift': 0.40, 'vol': 0.90},
    'ADA': {'base_price': 0.50, 'drift': 0.20, 'vol': 0.85},
    'DOGE': {'base_price': 0.10, 'drift': 0.15, 'vol': 1.00}
}

# Risk-free rate (US Treasury ~4% annualised)
rf_annual = 0.04
rf_daily = rf_annual / 252

# Generate correlated returns
n_stocks = len(stocks)
n_cryptos = len(cryptos)

# Stock correlation matrix (realistic sector correlations)
stock_corr = np.array([
    [1.00, 0.75, 0.40, 0.20, 0.30],  # AAPL
    [0.75, 1.00, 0.45, 0.25, 0.35],  # MSFT
    [0.40, 0.45, 1.00, 0.50, 0.40],  # JPM
    [0.20, 0.25, 0.50, 1.00, 0.35],  # XOM
    [0.30, 0.35, 0.40, 0.35, 1.00]   # KO
])

# Crypto correlation matrix (high correlation within crypto)
crypto_corr = np.array([
    [1.00, 0.85, 0.70, 0.65, 0.55],  # BTC
    [0.85, 1.00, 0.75, 0.70, 0.60],  # ETH
    [0.70, 0.75, 1.00, 0.65, 0.55],  # SOL
    [0.65, 0.70, 0.65, 1.00, 0.50],  # ADA
    [0.55, 0.60, 0.55, 0.50, 1.00]   # DOGE
])

# Generate returns using Cholesky decomposition
def generate_correlated_returns(n_days, corr_matrix, drifts, vols):
    L = np.linalg.cholesky(corr_matrix)
    independent = np.random.randn(n_days, len(drifts))
    correlated = independent @ L.T
    daily_returns = drifts + correlated * vols
    return daily_returns

# Stock returns
stock_drifts = np.array([s['drift'] / 252 for s in stocks.values()])
stock_vols = np.array([s['vol'] / np.sqrt(252) for s in stocks.values()])
stock_returns = generate_correlated_returns(n_days, stock_corr, stock_drifts, stock_vols)

# Crypto returns (365-day annualisation but aligned to business days)
crypto_drifts = np.array([c['drift'] / 252 for c in cryptos.values()])
crypto_vols = np.array([c['vol'] / np.sqrt(252) for c in cryptos.values()])
crypto_returns = generate_correlated_returns(n_days, crypto_corr, crypto_drifts, crypto_vols)

# Build price dataframes
def returns_to_prices(returns, base_prices, tickers):
    prices = pd.DataFrame(index=dates)
    for i, ticker in enumerate(tickers):
        price_series = base_prices[i] * np.exp(np.cumsum(returns[:, i]))
        prices[ticker] = price_series
    return prices

stock_prices = returns_to_prices(stock_returns, 
                                  [s['base_price'] for s in stocks.values()],
                                  list(stocks.keys()))

crypto_prices = returns_to_prices(crypto_returns,
                                   [c['base_price'] for c in cryptos.values()],
                                   list(cryptos.keys()))

# Add 4:1 split for AAPL on 2023-06-15
split_date = pd.Timestamp('2023-06-15')
stock_prices.loc[stock_prices.index < split_date, 'AAPL'] *= 4.0

# Create returns DataFrames
stock_returns_df = stock_prices.pct_change(fill_method=None).dropna()
crypto_returns_df = crypto_prices.pct_change(fill_method=None).dropna()

# Combined panel (stocks + crypto)
combined_returns = pd.concat([stock_returns_df, crypto_returns_df], axis=1).dropna()

# Risk-free rate series (forward-filled)
rf_series = pd.Series(rf_daily, index=combined_returns.index)

print(f"Generated {n_days} business days of data")
print(f"Stocks: {list(stocks.keys())}")
print(f"Cryptos: {list(cryptos.keys())}")
print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

# Save data
stock_prices.to_csv('data/stock_prices.csv')
crypto_prices.to_csv('data/crypto_prices.csv')
stock_returns_df.to_csv('data/stock_returns.csv')
crypto_returns_df.to_csv('data/crypto_returns.csv')
combined_returns.to_csv('data/combined_returns.csv')

print("Stage 1 complete: Data saved to data/")


# =============================================================================
# STAGE 2: Feature Engineering
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 2: Feature Engineering")
print("=" * 60)

# Calculate rolling statistics
def calculate_features(returns_df, window=21):
    features = pd.DataFrame(index=returns_df.index)
    features['rolling_vol'] = returns_df.rolling(window).std() * np.sqrt(252)
    features['rolling_return'] = returns_df.rolling(window).mean() * 252
    features['sharpe_rolling'] = (features['rolling_return'] - rf_annual) / features['rolling_vol']
    return features

# Rolling risk metrics
stock_vol_21d = stock_returns_df.rolling(21).std() * np.sqrt(252)
crypto_vol_21d = crypto_returns_df.rolling(21).std() * np.sqrt(252)

# Correlation evolution
stock_corr_60d = stock_returns_df.rolling(60).corr()
crypto_corr_60d = crypto_returns_df.rolling(60).corr()

print("Calculated rolling volatility (21-day)")
print("Calculated rolling correlation (60-day)")
print("Stage 2 complete: Features engineered")


# =============================================================================
# STAGE 3: Model Design (Portfolio Optimisation)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3: Model Design (Portfolio Optimisation)")
print("=" * 60)

optimizer = PortfolioOptimizer(risk_free_rate=rf_annual)

# Equity-only optimisation
print("\n--- Equity-only portfolios ---")
eq_max_sharpe_w, eq_max_sharpe_m = optimizer.optimize_max_sharpe(stock_returns_df)
eq_min_var_w, eq_min_var_m = optimizer.optimize_min_variance(stock_returns_df)
eq_risk_parity_w, eq_risk_parity_m = optimizer.optimize_risk_parity(stock_returns_df)
eq_equal_w, eq_equal_m = optimizer.optimize_equal_weight(stock_returns_df)

equity_weights = {
    'max_sharpe': dict(zip(stock_returns_df.columns, eq_max_sharpe_w)),
    'min_variance': dict(zip(stock_returns_df.columns, eq_min_var_w)),
    'risk_parity': dict(zip(stock_returns_df.columns, eq_risk_parity_w)),
    'equal_weight': dict(zip(stock_returns_df.columns, eq_equal_w))
}

print(f"Max Sharpe: {dict(zip(stock_returns_df.columns, np.round(eq_max_sharpe_w, 3)))}")
print(f"Min Variance: {dict(zip(stock_returns_df.columns, np.round(eq_min_var_w, 3)))}")
print(f"Risk Parity: {dict(zip(stock_returns_df.columns, np.round(eq_risk_parity_w, 3)))}")

# Crypto-only optimisation
print("\n--- Crypto-only portfolios ---")
cr_max_sharpe_w, cr_max_sharpe_m = optimizer.optimize_max_sharpe(crypto_returns_df)
cr_min_var_w, cr_min_var_m = optimizer.optimize_min_variance(crypto_returns_df)
cr_risk_parity_w, cr_risk_parity_m = optimizer.optimize_risk_parity(crypto_returns_df)

crypto_weights = {
    'max_sharpe': dict(zip(crypto_returns_df.columns, cr_max_sharpe_w)),
    'min_variance': dict(zip(crypto_returns_df.columns, cr_min_var_w)),
    'risk_parity': dict(zip(crypto_returns_df.columns, cr_risk_parity_w))
}

print(f"Max Sharpe: {dict(zip(crypto_returns_df.columns, np.round(cr_max_sharpe_w, 3)))}")
print(f"Min Variance: {dict(zip(crypto_returns_df.columns, np.round(cr_min_var_w, 3)))}")
print(f"Risk Parity: {dict(zip(crypto_returns_df.columns, np.round(cr_risk_parity_w, 3)))}")

# Combined optimisation (5 stocks + 5 cryptos)
print("\n--- Combined (multi-asset) portfolios ---")
co_max_sharpe_w, co_max_sharpe_m = optimizer.optimize_max_sharpe(combined_returns)
co_min_var_w, co_min_var_m = optimizer.optimize_min_variance(combined_returns)
co_risk_parity_w, co_risk_parity_m = optimizer.optimize_risk_parity(combined_returns)

combined_weights = {
    'max_sharpe': dict(zip(combined_returns.columns, co_max_sharpe_w)),
    'min_variance': dict(zip(combined_returns.columns, co_min_var_w)),
    'risk_parity': dict(zip(combined_returns.columns, co_risk_parity_w))
}

print(f"Max Sharpe: {dict(zip(combined_returns.columns, np.round(co_max_sharpe_w, 3)))}")
print(f"Risk Parity: {dict(zip(combined_returns.columns, np.round(co_risk_parity_w, 3)))}")

# Efficient frontiers
print("\n--- Computing efficient frontiers ---")
eq_ef_ret, eq_ef_vol = optimizer.get_efficient_frontier(stock_returns_df, n_points=50)
cr_ef_ret, cr_ef_vol = optimizer.get_efficient_frontier(crypto_returns_df, n_points=50)
co_ef_ret, co_ef_vol = optimizer.get_efficient_frontier(combined_returns, n_points=50)

print("Stage 3 complete: Portfolio optimisation done")


# =============================================================================
# STAGE 4: Backtesting (Implementation)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 4: Out-of-sample Backtesting")
print("=" * 60)

backtester = Backtester(risk_free_rate=rf_annual)

# Combined backtest (main analysis)
methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
backtest_results = {}

for method in methods:
    print(f"\nBacktesting {method}...")
    result = backtester.rolling_backtest(
        combined_returns,
        lookback_days=252,
        rebalance_freq=21,
        method=method
    )
    backtest_results[method] = result
    stats = backtester.calculate_statistics(result)
    print(f"  {stats}")

# Also run equity-only and crypto-only risk parity for comparison
print("\nBacktesting equity-only risk parity...")
eq_rp_result = backtester.rolling_backtest(
    stock_returns_df,
    lookback_days=252,
    rebalance_freq=21,
    method='risk_parity'
)
eq_rp_stats = backtester.calculate_statistics(eq_rp_result)
print(f"  {eq_rp_stats}")

print("\nBacktesting crypto-only risk parity...")
cr_rp_result = backtester.rolling_backtest(
    crypto_returns_df,
    lookback_days=252,
    rebalance_freq=21,
    method='risk_parity'
)
cr_rp_stats = backtester.calculate_statistics(cr_rp_result)
print(f"  {cr_rp_stats}")

print("\nStage 4 complete: Backtesting done")


# =============================================================================
# STAGE 5: Sentiment Analysis
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 5: Sentiment Analysis")
print("=" * 60)

# Generate synthetic news data
np.random.seed(123)
n_articles = 5000

sectors = ['Technology', 'Financials', 'Energy', 'Consumer', 'Technology']
tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'KO']

sentiment_records = []
for i in range(n_articles):
    sector_idx = np.random.randint(0, len(sectors))
    date_idx = np.random.randint(0, n_days)
    
    # Generate sentiment with slight positive bias
    base_sentiment = np.random.normal(0.05, 0.3)
    base_sentiment = np.clip(base_sentiment, -1, 1)
    
    sentiment_records.append({
        'date': dates[date_idx],
        'sector': sectors[sector_idx],
        'ticker': tickers[sector_idx],
        'sentiment_score': base_sentiment,
        'sentiment_label': 'Positive' if base_sentiment > 0.1 else ('Negative' if base_sentiment < -0.1 else 'Neutral'),
        'confidence': abs(base_sentiment)
    })

sentiment_df = pd.DataFrame(sentiment_records)
sentiment_df.to_csv('data/sentiment_data.csv')

# Calculate sector-level sentiment
sector_sentiment = sentiment_df.groupby(['date', 'sector'])['sentiment_score'].mean().reset_index()

print(f"Generated {n_articles} synthetic news articles")
print(f"Sentiment distribution: {sentiment_df['sentiment_label'].value_counts().to_dict()}")
print("Stage 5 complete: Sentiment analysis done")


# =============================================================================
# GENERATE FT-STYLE FIGURES
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING FT-STYLE FIGURES")
print("=" * 60)

# Import figure generation functions
from generate_report import (
    create_figure1_equity_allocation,
    create_figure2_crypto_allocation,
    create_figure3_efficient_frontier,
    create_figure4_backtest_cumulative,
    create_figure5_sentiment_index,
    create_performance_table
)

# Figure 1: Equity allocation
fig1 = create_figure1_equity_allocation(equity_weights, 'figures/fig1_equity_allocation.png')
print(f"Created: {fig1}")

# Figure 2: Crypto allocation
fig2 = create_figure2_crypto_allocation(crypto_weights, 'figures/fig2_crypto_allocation.png')
print(f"Created: {fig2}")

# Figure 3: Efficient frontier
fig3 = create_figure3_efficient_frontier(
    equity_ef=pd.DataFrame({'vol': eq_ef_vol, 'ret': eq_ef_ret}),
    crypto_ef=pd.DataFrame({'vol': cr_ef_vol, 'ret': cr_ef_ret}),
    combined_ef=pd.DataFrame({'vol': co_ef_vol, 'ret': co_ef_ret}),
    equity_sharpe={'vol': eq_max_sharpe_m['annual_volatility'], 
                   'ret': eq_max_sharpe_m['annual_return'],
                   'sharpe': eq_max_sharpe_m['sharpe_ratio']},
    crypto_sharpe={'vol': cr_max_sharpe_m['annual_volatility'],
                   'ret': cr_max_sharpe_m['annual_return'],
                   'sharpe': cr_max_sharpe_m['sharpe_ratio']},
    combined_sharpe={'vol': co_max_sharpe_m['annual_volatility'],
                     'ret': co_max_sharpe_m['annual_return'],
                     'sharpe': co_max_sharpe_m['sharpe_ratio']},
    save_path='figures/fig3_efficient_frontier.png'
)
print(f"Created: {fig3}")

# Figure 4: Backtest cumulative returns
# Prepare data for plotting
bt_plot_data = {}
for method in methods:
    bt_plot_data[method] = backtest_results[method]

fig4 = create_figure4_backtest_cumulative(bt_plot_data, 'figures/fig4_backtest_cumulative.png')
print(f"Created: {fig4}")

# Figure 5: Sentiment index
fig5 = create_figure5_sentiment_index(
    sentiment_data=sentiment_df,
    returns_data=stock_returns_df,
    save_path='figures/fig5_sentiment_index.png'
)
print(f"Created: {fig5}")

# Performance table
backtest_summary = {}
for method in methods:
    backtest_summary[method] = backtester.calculate_statistics(backtest_results[method])

fig_table = create_performance_table(backtest_summary, 'figures/table_performance.png')
print(f"Created: {fig_table}")

# Save all results to JSON for reference
results_summary = {
    'equity_weights': equity_weights,
    'crypto_weights': crypto_weights,
    'combined_weights': combined_weights,
    'backtest_stats': backtest_summary,
    'equity_tangency': {
        'return': float(eq_max_sharpe_m['annual_return']),
        'volatility': float(eq_max_sharpe_m['annual_volatility']),
        'sharpe': float(eq_max_sharpe_m['sharpe_ratio'])
    },
    'crypto_tangency': {
        'return': float(cr_max_sharpe_m['annual_return']),
        'volatility': float(cr_max_sharpe_m['annual_volatility']),
        'sharpe': float(cr_max_sharpe_m['sharpe_ratio'])
    },
    'combined_tangency': {
        'return': float(co_max_sharpe_m['annual_return']),
        'volatility': float(co_max_sharpe_m['annual_volatility']),
        'sharpe': float(co_max_sharpe_m['sharpe_ratio'])
    }
}

with open('data/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 60)
print("ALL FIGURES AND DATA GENERATED SUCCESSFULLY")
print("=" * 60)
print("\nFiles created:")
print("  figures/fig1_equity_allocation.png")
print("  figures/fig2_crypto_allocation.png")
print("  figures/fig3_efficient_frontier.png")
print("  figures/fig4_backtest_cumulative.png")
print("  figures/fig5_sentiment_index.png")
print("  figures/table_performance.png")
print("  data/*.csv")
print("  data/results_summary.json")
