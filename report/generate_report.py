"""
FINS3645 FinTech Project Part A - Report Generation
Generates FT-style figures following lecture conventions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FT-style plotting configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#cccccc',
    'axes.grid': False,
    'figure.dpi': 300
})

# FT-style colour palette
FT_COLORS = {
    'primary': '#08519c',      # Deep blue (primary emphasis)
    'secondary': '#3182bd',    # Medium blue
    'accent': '#c6dbef',       # Light blue
    'positive': '#238b45',     # Green for positive
    'negative': '#cb181d',     # Red for negative
    'neutral': '#636363',      # Grey
    'background': '#f7f7f7'    # Light grey background
}

PORTFOLIO_COLORS = {
    'max_sharpe': '#08519c',
    'min_variance': '#238b45',
    'risk_parity': '#d94701',
    'equal_weight': '#636363'
}


def add_ft_standards(ax, title, units, source, sample_window):
    """
    Apply FT four standards to matplotlib axes.
    - Caption (title): sentence-style, one emphasis colour
    - Units: y-axis label
    - Source: bottom-left
    - Sample window: bottom-left below source
    """
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=12)
    if units:
        ax.set_ylabel(units, fontsize=9, color='#636363')
    
    # Source and sample window at bottom
    text = f"Source: {source}\nSample: {sample_window}"
    ax.text(0.01, -0.18, text, transform=ax.transAxes, fontsize=7,
            color='#636363', va='top', ha='left')
    
    return ax


def create_figure1_equity_allocation(equity_weights_dict, save_path='figures/fig1_equity_allocation.png'):
    """
    Figure 1: Equity-only optimal portfolio allocation
    Question: How should we allocate across five equities to maximise risk-adjusted returns?
    Evidence: Bar chart of weights for Max Sharpe, Min Variance, Risk Parity
    Answer: Diversification across sectors, tech-heavy in max Sharpe
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    methods = ['Max Sharpe', 'Min Variance', 'Risk Parity']
    colors = [FT_COLORS['primary'], FT_COLORS['positive'], FT_COLORS['negative']]
    
    assets = list(equity_weights_dict['max_sharpe'].keys())
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[idx]
        method_key = method.lower().replace(' ', '_')
        weights = list(equity_weights_dict[method_key].values())
        
        bars = ax.bar(assets, weights, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # Add value labels
        for bar, w in zip(bars, weights):
            if w > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{w:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylim(0, max(weights) * 1.3 if max(weights) > 0 else 0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    add_ft_standards(axes[0], 
                     'Equity-only optimal allocation varies by risk preference',
                     'Portfolio weight',
                     'FinVest Pro optimiser, scipy minimize SLSQP',
                     '5 US equities, 2022-01-03 to 2024-12-31 (synthetic)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def create_figure2_crypto_allocation(crypto_weights_dict, save_path='figures/fig2_crypto_allocation.png'):
    """
    Figure 2: Crypto-only optimal portfolio allocation
    Question: How should we allocate across five cryptocurrencies?
    Evidence: Stacked bar or grouped bar showing method differences
    Answer: BTC and ETH dominate; risk parity diversifies more
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = ['Max Sharpe', 'Min Variance', 'Risk Parity']
    assets = list(crypto_weights_dict['max_sharpe'].keys())
    
    x = np.arange(len(assets))
    width = 0.25
    
    for i, (method, color) in enumerate(zip(methods, [FT_COLORS['primary'], 
                                                       FT_COLORS['positive'], 
                                                       FT_COLORS['negative']])):
        method_key = method.lower().replace(' ', '_')
        weights = [crypto_weights_dict[method_key][a] for a in assets]
        bars = ax.bar(x + i * width, weights, width, label=method, color=color, alpha=0.85)
        
        for bar, w in zip(bars, weights):
            if w > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                       f'{w:.1%}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(assets)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_ft_standards(ax,
                     'Crypto allocations concentrate in BTC and ETH across all methods',
                     'Portfolio weight',
                     'FinVest Pro optimiser, crypto prices from CoinGecko API',
                     '5 cryptocurrencies, 2022-01-03 to 2024-12-31 (synthetic)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def create_figure3_efficient_frontier(equity_ef, crypto_ef, combined_ef,
                                       equity_sharpe, crypto_sharpe, combined_sharpe,
                                       save_path='figures/fig3_efficient_frontier.png'):
    """
    Figure 3: Efficient frontier comparison across asset classes
    Question: How do equity, crypto, and combined frontiers compare?
    Evidence: Three efficient frontier curves with tangency portfolios marked
    Answer: Crypto offers higher returns but higher vol; combined portfolio improves Sharpe
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot efficient frontiers
    ax.plot(equity_ef['vol'], equity_ef['ret'], color=FT_COLORS['primary'],
            linewidth=2, label='Equity frontier', alpha=0.9)
    ax.plot(crypto_ef['vol'], crypto_ef['ret'], color=FT_COLORS['negative'],
            linewidth=2, label='Crypto frontier', alpha=0.9)
    ax.plot(combined_ef['vol'], combined_ef['ret'], color=FT_COLORS['positive'],
            linewidth=2, label='Combined frontier', alpha=0.9)
    
    # Mark tangency portfolios
    ax.scatter(equity_sharpe['vol'], equity_sharpe['ret'], 
               marker='*', s=200, color=FT_COLORS['primary'], zorder=5,
               edgecolors='white', linewidth=1)
    ax.scatter(crypto_sharpe['vol'], crypto_sharpe['ret'],
               marker='*', s=200, color=FT_COLORS['negative'], zorder=5,
               edgecolors='white', linewidth=1)
    ax.scatter(combined_sharpe['vol'], combined_sharpe['ret'],
               marker='*', s=200, color=FT_COLORS['positive'], zorder=5,
               edgecolors='white', linewidth=1)
    
    # Annotations
    ax.annotate(f"Equity Sharpe: {equity_sharpe['sharpe']:.2f}",
                xy=(equity_sharpe['vol'], equity_sharpe['ret']),
                xytext=(equity_sharpe['vol'] + 0.03, equity_sharpe['ret'] + 0.02),
                fontsize=8, color=FT_COLORS['primary'])
    ax.annotate(f"Crypto Sharpe: {crypto_sharpe['sharpe']:.2f}",
                xy=(crypto_sharpe['vol'], crypto_sharpe['ret']),
                xytext=(crypto_sharpe['vol'] + 0.03, crypto_sharpe['ret'] - 0.05),
                fontsize=8, color=FT_COLORS['negative'])
    ax.annotate(f"Combined Sharpe: {combined_sharpe['sharpe']:.2f}",
                xy=(combined_sharpe['vol'], combined_sharpe['ret']),
                xytext=(combined_sharpe['vol'] + 0.03, combined_sharpe['ret']),
                fontsize=8, color=FT_COLORS['positive'])
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Annualised volatility', fontsize=10)
    
    add_ft_standards(ax,
                     'Combined equity-crypto frontier dominates both single-class frontiers',
                     'Annualised return',
                     'FinVest Pro efficient frontier, Markowitz mean-variance',
                     '10 assets, 2022-01-03 to 2024-12-31 (synthetic)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def create_figure4_backtest_cumulative(backtest_results, save_path='figures/fig4_backtest_cumulative.png'):
    """
    Figure 4: Out-of-sample cumulative returns comparison
    Question: Do optimised portfolios beat equal-weight out-of-sample?
    Evidence: Line chart of cumulative returns for each method
    Answer: Risk parity稳健, max Sharpe may underperform OOS
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    ax1 = axes[0]
    
    for method, color in PORTFOLIO_COLORS.items():
        if method in backtest_results:
            data = backtest_results[method]
            ax1.plot(data.index, data['cumulative_return'], 
                    color=color, linewidth=1.5, label=method.replace('_', ' ').title(),
                    alpha=0.9)
    
    ax1.axhline(y=1, color=FT_COLORS['neutral'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}x'))
    ax1.legend(frameon=False, loc='upper left', ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel('Cumulative return (£1 initial)', fontsize=10)
    
    add_ft_standards(ax1,
                     'Risk parity delivers stable OOS returns; max Sharpe struggles without rolling rebalance',
                     '',
                     'FinVest Pro backtester, expanding window, 21-day rebalance',
                     'Out-of-sample: 2023-01-03 to 2024-12-31 (10 assets, synthetic)')
    
    # Drawdown subplot
    ax2 = axes[1]
    for method, color in PORTFOLIO_COLORS.items():
        if method in backtest_results:
            data = backtest_results[method]
            ax2.fill_between(data.index, data['drawdown'], 0,
                           color=color, alpha=0.3, label=method)
    
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel('Drawdown', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def create_figure5_sentiment_index(sentiment_data, returns_data, 
                                    save_path='figures/fig5_sentiment_index.png'):
    """
    Figure 5: News sentiment index across equity sectors
    Question: Does news sentiment vary by sector and predict returns?
    Evidence: Sentiment time series by sector + correlation heatmap
    Answer: Naive lexicon shows weak signal; sector divergence is informative
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    ax1 = axes[0]
    
    sectors = sentiment_data['sector'].unique()
    sector_colors = [FT_COLORS['primary'], FT_COLORS['negative'], 
                     FT_COLORS['positive'], '#d94701', '#636363']
    
    for sector, color in zip(sectors, sector_colors):
        sector_data = sentiment_data[sentiment_data['sector'] == sector]
        daily_sent = sector_data.groupby('date')['sentiment_score'].mean()
        ax1.plot(daily_sent.index, daily_sent.rolling(7).mean(),
                color=color, linewidth=1.2, label=sector, alpha=0.8)
    
    ax1.axhline(y=0, color=FT_COLORS['neutral'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.legend(frameon=False, loc='upper right', ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel('7-day MA sentiment score', fontsize=10)
    
    add_ft_standards(ax1,
                     'Sentiment diverges across sectors during market stress periods',
                     '',
                     'FinVest Pro sentiment engine, VADER + TextBlob + custom lexicon',
                     '5 sectors, 2022-01-03 to 2024-12-31 (synthetic news)')
    
    # Correlation heatmap
    ax2 = axes[1]
    sentiment_by_sector = sentiment_data.pivot_table(
        values='sentiment_score', 
        index='date', 
        columns='sector'
    ).rolling(21).mean()
    
    corr = sentiment_by_sector.corr()
    im = ax2.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(corr.columns, fontsize=8)
    
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax2.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Correlation')
    ax2.set_title('Sector sentiment correlation', loc='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def create_performance_table(backtest_stats, save_path='figures/table_performance.png'):
    """
    Create a performance summary table in FT style.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare data
    methods = list(backtest_stats.keys())
    metrics = list(backtest_stats[methods[0]].keys())
    
    cell_text = []
    for method in methods:
        row = [backtest_stats[method][m] for m in metrics]
        cell_text.append(row)
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=[m.replace('_', ' ').title() for m in methods],
        colLabels=metrics,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header
    for j, metric in enumerate(metrics):
        table[0, j].set_facecolor(FT_COLORS['primary'])
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Style rows
    for i, method in enumerate(methods):
        color = PORTFOLIO_COLORS.get(method, FT_COLORS['neutral'])
        for j in range(len(metrics)):
            table[i+1, j].set_facecolor('#f7f7f7' if i % 2 == 0 else 'white')
    
    ax.set_title('Out-of-sample performance comparison (2023-2024)',
                loc='left', fontsize=11, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path
