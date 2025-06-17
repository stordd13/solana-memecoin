import pandas as pd
import numpy as np
from typing import Dict, List
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis

class TimeSeriesAnalyzer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators to the DataFrame.
        """
        df['ma_5'] = df['price'].rolling_mean(5)
        df['ma_30'] = df['price'].rolling_mean(30)
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Calculate rolling volatility (30-minute window)
        df['volatility_30'] = df['returns'].rolling_std(30) * np.sqrt(30) # Annualized for 30min window
        
        # Calculate momentum (e.g., 30-minute price change)
        df['momentum_30'] = df['price'].diff(periods=30)
        
        return df

    def get_price_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculates key price statistics for a given token DataFrame.
        """
        initial_price = df['price'][0]
        final_price = df['price'][-1]
        
        price_change_pct = ((final_price - initial_price) / initial_price) * 100 if initial_price else 0
        
        # Max gain from initial price
        max_price = df['price'].max()
        max_gain_pct = ((max_price - initial_price) / initial_price) * 100 if initial_price else 0
        
        # Volatility (standard deviation of returns)
        volatility = df['returns'].std()
        
        # Max Drawdown
        # Calculate the cumulative maximum price
        cumulative_max = df['price'].cummax()
        # Calculate drawdown
        drawdown = (df['price'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        return {
            'initial_price': initial_price,
            'final_price': final_price,
            'price_change_pct': price_change_pct,
            'total_return': price_change_pct, # Alias for consistency with some terms
            'max_price': max_price,
            'max_gain_pct': max_gain_pct,
            'volatility': volatility,
            'max_drawdown': max_drawdown
        }

    def find_pumps(self, df: pd.DataFrame, threshold: float = 50, window: int = 30) -> List[Dict]:
        """Find pump events (rapid price increases)"""
        pumps = []
        
        for i in range(window, len(df)):
            gain = (df['price'][i] / df['price'][i-window] - 1) * 100
            
            if gain > threshold:
                pumps.append({
                    'start_idx': i - window,
                    'end_idx': i,
                    'start_time': df['datetime'][i-window],
                    'end_time': df['datetime'][i],
                    'start_price': df['price'][i-window],
                    'end_price': df['price'][i],
                    'gain_pct': gain,
                    'duration_minutes': window
                })
        
        return pumps
    
    def analyze_first_hour(self, df: pd.DataFrame) -> Dict:
        """Analyze first hour patterns specifically"""
        first_hour = df[:60] if len(df) >= 60 else df
        
        # Basic metrics
        initial_price = first_hour['price'][0]
        final_price_1h = first_hour['price'][-1]
        max_price_1h = first_hour['price'].max()
        
        # Calculate features
        features = {
            'gain_1h': ((final_price_1h / initial_price - 1) * 100),
            'max_gain_1h': ((max_price_1h / initial_price - 1) * 100),
            'volatility_1h': first_hour['price'].pct_change().std(),
            'price_changes_1h': (first_hour['price'].diff() != 0).sum(),
        }
        
        # Momentum at different intervals
        if len(first_hour) >= 10:
            features['gain_10m'] = ((first_hour['price'][9] / initial_price - 1) * 100)
        if len(first_hour) >= 30:
            features['gain_30m'] = ((first_hour['price'][29] / initial_price - 1) * 100)
        
        # Consecutive gains
        price_changes = first_hour['price'].diff()
        gains = (price_changes > 0).astype(int)
        features['consecutive_gains'] = gains.groupby((gains != gains.shift()).cumsum()).sum().max()
        
        # Time to first 2x
        double_price = initial_price * 2
        doubled = first_hour[first_hour['price'] >= double_price]
        features['time_to_2x'] = doubled.index[0] if len(doubled) > 0 else None
        
        return features
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect common price patterns"""
        patterns = {
            'pump_and_dump': False,
            'steady_growth': False,
            'volatile_sideways': False,
            'dead_coin': False
        }
        
        # Get basic metrics
        stats = self.get_price_statistics(df)
        
        # Pump and dump: High max gain but negative or low final return
        if stats['max_gain_pct'] > 500 and stats['price_change_pct'] < -50:
            patterns['pump_and_dump'] = True
        
        # Steady growth: Positive return with low volatility
        if stats['price_change_pct'] > 50 and stats['volatility'] < 0.1:
            patterns['steady_growth'] = True
        
        # Volatile sideways: High volatility but small final change
        if stats['volatility'] > 0.15 and abs(stats['price_change_pct']) < 20:
            patterns['volatile_sideways'] = True
        
        # Dead coin: Minimal price movement
        if stats['max_gain_pct'] < 10 and stats['volatility'] < 0.02:
            patterns['dead_coin'] = True
        
        return patterns

    def extract_features(self, df: pd.DataFrame) -> Dict:
        """
        Extracts a comprehensive set of features from the price time series, including trend, seasonality, volatility, momentum, and pattern features.
        """
        features = {}
        price = df['price']
        returns = price.pct_change().drop_nulls()
        initial_price = price[0]
        normalized_price = price / initial_price
        features['initial_price'] = initial_price
        features['final_price'] = price[-1]
        features['max_price'] = price.max()
        features['min_price'] = price.min()
        features['normalized_final'] = normalized_price[-1]
        features['max_gain_pct'] = (price.max() / initial_price - 1) * 100
        features['final_return_pct'] = (price[-1] / initial_price - 1) * 100
        features['num_doublings'] = (normalized_price > 2).sum()
        features['time_to_peak'] = price.idxmax() if hasattr(price, 'idxmax') else np.nan
        features['time_to_bottom'] = price.idxmin() if hasattr(price, 'idxmin') else np.nan
        features['volatility_5m'] = returns.rolling_std(5).mean()
        features['volatility_30m'] = returns.rolling_std(30).mean()
        features['volatility_full'] = returns.std()
        features['max_drawdown'] = self.calculate_max_drawdown(price)
        features['skew'] = skew(returns)
        features['kurtosis'] = kurtosis(returns)
        features['momentum_shifts'] = (np.diff(np.sign(returns)) != 0).sum()
        features['consecutive_gains'] = self.max_consecutive_streak(returns > 0)
        features['consecutive_losses'] = self.max_consecutive_streak(returns < 0)
        features['autocorr_1'] = returns.autocorr(lag=1)
        features['autocorr_5'] = returns.autocorr(lag=5)
        features['hurst'] = self.hurst_exponent(price)
        # Trend (linear regression slope)
        x = np.arange(len(price))
        slope = np.polyfit(x, price, 1)[0]
        features['trend_slope'] = slope
        # STL decomposition
        stl = self.decompose_series(df)
        features['trend_strength'] = stl['trend'].std() / price.std() if price.std() > 0 else 0
        features['seasonal_strength'] = stl['seasonal'].std() / price.std() if price.std() > 0 else 0
        # Stationarity
        adf = self.adf_test(price)
        features['adf_stat'] = adf['adf_statistic']
        features['adf_pvalue'] = adf['p_value']
        features['is_stationary'] = adf['is_stationary']
        return features

    def decompose_series(self, df: pd.DataFrame) -> Dict:
        """
        Performs STL decomposition and returns trend, seasonal, and residual components.
        """
        price = df['price']
        stl = STL(price, period=60, robust=True).fit()
        return {
            'trend': stl.trend,
            'seasonal': stl.seasonal,
            'resid': stl.resid
        }

    def adf_test(self, series: pd.Series) -> Dict:
        """
        Performs Augmented Dickey-Fuller test for stationarity.
        """
        series = series.dropna()
        if len(series) < 2:
            return {'adf_statistic': None, 'p_value': None, 'is_stationary': False}
        result = adfuller(series)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }

    def classify_pattern(self, features: Dict) -> str:
        """
        Classifies the price pattern using feature-based logic from the README.
        """
        if features['max_gain_pct'] > 1000 and features['time_to_peak'] is not None and features['time_to_peak'] < 60:
            return 'explosive_pump'
        if features['trend_slope'] > 0 and features['volatility_full'] < 0.1:
            return 'steady_climb'
        if features['max_gain_pct'] > 500 and features['final_return_pct'] < -80:
            return 'pump_and_dump'
        if features['volatility_full'] > 0.2 and features['final_return_pct'] > 0:
            return 'volatile_survivor'
        if features['trend_slope'] < 0 and features['final_return_pct'] < -50:
            return 'slow_death'
        if features['max_gain_pct'] < 20 and features['final_return_pct'] < 20:
            return 'instant_fail'
        return 'other'

    def calculate_max_drawdown(self, price: pd.Series) -> float:
        """
        Calculates the maximum drawdown of a price series.
        """
        roll_max = price.cummax()
        drawdown = (price - roll_max) / roll_max
        return drawdown.min() if not drawdown.empty else 0

    def max_consecutive_streak(self, arr) -> int:
        """
        Returns the length of the maximum consecutive True streak in a boolean array.
        """
        max_streak = streak = 0
        for val in arr:
            if val:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def hurst_exponent(self, price: pd.Series) -> float:
        """
        Estimates the Hurst exponent to distinguish trending vs mean-reverting behavior.
        """
        lags = range(2, 20)
        tau = [np.std(np.subtract(price[lag:], price[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0
