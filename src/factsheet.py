import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


class FactSheetGenerator:
    """Generates fund fact sheets with key performance metrics."""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#7f7f7f',
            'background': '#f8f9fa'
        }
    
    def create_growth_chart(
        self,
        cumulative_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Growth of $1"
    ) -> go.Figure:
        """Create growth of $1 chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            name='Fund',
            line=dict(color=self.color_scheme['primary'], width=2)
        ))
        
        if benchmark_returns is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_returns.index,
                y=benchmark_returns.values,
                name='Benchmark',
                line=dict(color=self.color_scheme['neutral'], width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_drawdown_chart(
        self,
        drawdowns: pd.Series,
        title: str = "Drawdowns"
    ) -> go.Figure:
        """Create drawdown chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color=self.color_scheme['negative'], width=1),
            fillcolor='rgba(214, 39, 40, 0.3)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown',
            yaxis_tickformat='.1%',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_allocation_chart(
        self,
        weights: np.ndarray,
        asset_names: list,
        title: str = "Current Holdings"
    ) -> go.Figure:
        """Create pie chart for fund allocation."""
        fig = go.Figure(data=[go.Pie(
            labels=asset_names,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=[
                self.color_scheme['primary'],
                self.color_scheme['secondary'],
                self.color_scheme['positive'],
                self.color_scheme['negative']
            ])
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        return fig
    
    def generate_factsheet(
        self,
        fund_name: str,
        metrics: dict,
        cumulative_returns: pd.Series,
        drawdowns: pd.Series,
        weights: np.ndarray,
        asset_names: list
    ) -> Dict[str, go.Figure]:
        """Generate complete fact sheet with all charts."""
        charts = {
            'growth': self.create_growth_chart(
                cumulative_returns,
                title=f"{fund_name} - Growth of $1"
            ),
            'drawdowns': self.create_drawdown_chart(
                drawdowns,
                title=f"{fund_name} - Drawdowns"
            ),
            'holdings': self.create_allocation_chart(
                weights,
                asset_names,
                title=f"{fund_name} - Current Holdings"
            )
        }
        
        return charts
    
    def create_comparison_chart(
        self,
        funds_data: Dict[str, pd.Series],
        title: str = "Fund Comparison"
    ) -> go.Figure:
        """Create comparison chart for multiple funds."""
        fig = go.Figure()
        
        colors = [
            self.color_scheme['primary'],
            self.color_scheme['secondary'],
            self.color_scheme['positive'],
            self.color_scheme['negative']
        ]
        
        for i, (fund_name, returns) in enumerate(funds_data.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns.values,
                name=fund_name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_metrics_table(
        self,
        metrics: Dict[str, str]
    ) -> pd.DataFrame:
        """Create metrics table for display."""
        df = pd.DataFrame([
            {'Metric': k, 'Value': v} 
            for k, v in metrics.items()
        ])
        return df
    
    def format_metrics(self, raw_metrics: dict) -> Dict[str, str]:
        """Format metrics for display."""
        formatted = {}
        
        if 'annual_return' in raw_metrics:
            formatted['Annual Return'] = f"{raw_metrics['annual_return']:.2%}"
        if 'annual_volatility' in raw_metrics:
            formatted['Annual Volatility'] = f"{raw_metrics['annual_volatility']:.2%}"
        if 'sharpe_ratio' in raw_metrics:
            formatted['Sharpe Ratio'] = f"{raw_metrics['sharpe_ratio']:.2f}"
        if 'max_drawdown' in raw_metrics:
            formatted['Max Drawdown'] = f"{raw_metrics['max_drawdown']:.2%}"
        if 'total_return' in raw_metrics:
            formatted['Total Return'] = f"{raw_metrics['total_return']:.2%}"
        if 'calmar_ratio' in raw_metrics:
            formatted['Calmar Ratio'] = f"{raw_metrics['calmar_ratio']:.2f}"
        
        return formatted
