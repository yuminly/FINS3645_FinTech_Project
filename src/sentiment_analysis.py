import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import re


class SentimentAnalyzer:
    """
    News sentiment analyzer using multiple methods:
    - VADER (lexicon-based)
    - TextBlob (pattern-based)
    - Custom financial lexicon
    """
    
    def __init__(self):
        self.financial_lexicon = self._load_financial_lexicon()
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize sentiment analyzers."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.vader_available = True
        except ImportError:
            self.vader_available = False
            print("VADER not available, using custom lexicon only")
        
        try:
            from textblob import TextBlob
            self.textblob_available = True
        except ImportError:
            self.textblob_available = False
            print("TextBlob not available")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load custom financial sentiment lexicon."""
        return {
            'bullish': 0.8, 'bearish': -0.8, 'rally': 0.7, 'crash': -0.9,
            'surge': 0.6, 'plunge': -0.7, 'soar': 0.7, 'tumble': -0.6,
            'gain': 0.5, 'loss': -0.5, 'profit': 0.6, 'deficit': -0.6,
            'growth': 0.5, 'decline': -0.5, 'recovery': 0.6, 'recession': -0.7,
            'boom': 0.7, 'bust': -0.7, 'upgrade': 0.5, 'downgrade': -0.5,
            'outperform': 0.6, 'underperform': -0.6, 'beat': 0.5, 'miss': -0.5,
            'strong': 0.4, 'weak': -0.4, 'optimistic': 0.6, 'pessimistic': -0.6,
            'uncertainty': -0.3, 'volatility': -0.2, 'stable': 0.3,
            'inflation': -0.4, 'deflation': -0.3, 'stimulus': 0.5,
            'hawkish': -0.3, 'dovish': 0.4, 'rate hike': -0.5, 'rate cut': 0.5,
            'breakout': 0.5, 'breakdown': -0.5, 'support': 0.3, 'resistance': -0.2,
            'momentum': 0.3, 'divergence': -0.2, 'convergence': 0.2,
            'overbought': -0.3, 'oversold': 0.3, 'bubble': -0.6, 'correction': -0.4
        }
    
    def analyze_sentiment_vader(self, text: str) -> float:
        """Analyze sentiment using VADER."""
        if not self.vader_available:
            return 0.0
        scores = self.vader.polarity_scores(text)
        return scores['compound']
    
    def analyze_sentiment_textblob(self, text: str) -> float:
        """Analyze sentiment using TextBlob."""
        if not self.textblob_available:
            return 0.0
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_sentiment_custom(self, text: str) -> float:
        """Analyze sentiment using custom financial lexicon."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        scores = []
        for word in words:
            if word in self.financial_lexicon:
                scores.append(self.financial_lexicon[word])
        
        if not scores:
            return 0.0
        
        return np.mean(scores)
    
    def analyze_headline(
        self,
        headline: str,
        method: str = 'ensemble'
    ) -> Dict[str, float]:
        """
        Analyze sentiment of a single headline.
        
        Parameters:
        -----------
        headline : str
            News headline text
        method : str
            'vader', 'textblob', 'custom', or 'ensemble'
        
        Returns:
        --------
        Dictionary with sentiment scores
        """
        if method == 'vader':
            score = self.analyze_sentiment_vader(headline)
        elif method == 'textblob':
            score = self.analyze_sentiment_textblob(headline)
        elif method == 'custom':
            score = self.analyze_sentiment_custom(headline)
        elif method == 'ensemble':
            scores = []
            if self.vader_available:
                scores.append(self.analyze_sentiment_vader(headline))
            if self.textblob_available:
                scores.append(self.analyze_sentiment_textblob(headline))
            scores.append(self.analyze_sentiment_custom(headline))
            score = np.mean(scores) if scores else 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'sentiment_score': score,
            'sentiment_label': self._get_sentiment_label(score),
            'confidence': abs(score)
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to label."""
        if score > 0.3:
            return 'Positive'
        elif score < -0.3:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_news_batch(
        self,
        news_data: List[Dict],
        method: str = 'ensemble'
    ) -> pd.DataFrame:
        """Analyze sentiment for a batch of news articles."""
        results = []
        
        for article in news_data:
            headline = article.get('title', '')
            if headline:
                sentiment = self.analyze_headline(headline, method)
                results.append({
                    'headline': headline,
                    'date': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    **sentiment
                })
        
        return pd.DataFrame(results)
    
    def build_sentiment_index(
        self,
        sentiment_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Build a daily sentiment index from news sentiment data.
        """
        sentiment_df['date'] = pd.to_datetime(sentiment_df[date_col]).dt.date
        
        daily_index = sentiment_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'confidence': 'mean',
            'headline': 'count'
        }).rename(columns={'headline': 'num_articles'})
        
        daily_index['sentiment_ma7'] = daily_index['sentiment_score'].rolling(7).mean()
        daily_index['sentiment_ma30'] = daily_index['sentiment_score'].rolling(30).mean()
        
        return daily_index
    
    def get_sentiment_summary(
        self,
        sentiment_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Get summary statistics for sentiment data."""
        return {
            'mean_sentiment': sentiment_df['sentiment_score'].mean(),
            'std_sentiment': sentiment_df['sentiment_score'].std(),
            'pct_positive': (sentiment_df['sentiment_label'] == 'Positive').mean(),
            'pct_negative': (sentiment_df['sentiment_label'] == 'Negative').mean(),
            'pct_neutral': (sentiment_df['sentiment_label'] == 'Neutral').mean(),
            'total_articles': len(sentiment_df)
        }
