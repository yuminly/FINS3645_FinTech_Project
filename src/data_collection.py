import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import os


class MarketDataCollector:
    """Collects and processes market data for equities and crypto."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_equity_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Fetch equity price data from Yahoo Finance."""
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    
    def fetch_crypto_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch cryptocurrency price data."""
        crypto_tickers = [f"{sym}-USD" for sym in symbols]
        data = yf.download(crypto_tickers, start=start_date, end=end_date)['Adj Close']
        return data
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns from prices."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def get_risk_free_rate(self, period: str = "3mo") -> float:
        """Fetch current risk-free rate (3-month T-bill)."""
        try:
            tbill = yf.download("^IRX", period="5d")['Adj Close']
            return tbill.iloc[-1] / 100
        except:
            return 0.04
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        train_ratio: float = 0.7
    ) -> tuple:
        """Split data into train and test sets."""
        split_idx = int(len(data) * train_ratio)
        return data.iloc[:split_idx], data.iloc[split_idx:]


class NewsCollector:
    """Collects financial news headlines."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_news(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = "en",
        page_size: int = 100
    ) -> List[Dict]:
        """Fetch news articles from NewsAPI."""
        if not self.api_key:
            return self._generate_mock_news(from_date, to_date)
        
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': language,
            'pageSize': page_size,
            'apiKey': self.api_key,
            'sortBy': 'publishedAt'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._generate_mock_news(from_date, to_date)
    
    def _generate_mock_news(
        self, 
        from_date: str, 
        to_date: str
    ) -> List[Dict]:
        """Generate mock news data for demonstration."""
        mock_headlines = [
            "Stock market rallies on strong earnings reports",
            "Fed signals potential rate cuts amid cooling inflation",
            "Tech stocks lead market gains on AI optimism",
            "Cryptocurrency market sees renewed investor interest",
            "Oil prices surge on supply concerns",
            "Bond yields fall as economic data softens",
            "Retail sector outperforms on strong consumer spending",
            "Market volatility increases amid geopolitical tensions",
            "Central banks maintain hawkish stance on monetary policy",
            "Growth stocks rebound after recent selloff"
        ]
        
        dates = pd.date_range(from_date, to_date, freq='D')
        news = []
        for i, date in enumerate(dates):
            headline_idx = i % len(mock_headlines)
            news.append({
                'title': mock_headlines[headline_idx],
                'publishedAt': date.isoformat(),
                'source': {'name': 'Mock Source'}
            })
        
        return news
