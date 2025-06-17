"""
Advanced trading analytics for memecoin analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class TradingAnalytics:
    """Advanced trading analytics and signal generation"""
    
    def __init__(self):
        self.fee_rate = 0.001  # 0.1% trading fee assumption
    
    def calculate_optimal_stop_loss_take_profit(self, df: pd.DataFrame, 
                                               lookback: int = 100) -> pd.DataFrame:
        """
        Calculate optimal stop-loss and take-profit levels based on historical volatility
        and price distributions
        """
        df = df.copy()
        
        # Calculate ATR (Average True Range) proxy
        df['high'] = df['price'].rolling(2).max()
        df['low'] = df['price'].rolling(2).min()
        df['tr'] = df['high'] - df['low']
        df['atr'] = df['tr'].rolling(lookback).mean()
        
        # Calculate optimal levels based on risk/reward
        df['returns'] = df['price'].pct_change()
        
        # Use rolling statistics for dynamic levels
        rolling_returns = df['returns'].rolling(lookback)
        
        # Calculate percentiles for stop-loss and take-profit
        df['stop_loss_1x'] = -rolling_returns.std() * 1.5  # 1.5x volatility
        df['stop_loss_2x'] = -rolling_returns.std() * 2.0  # 2x volatility
        df['take_profit_1x'] = rolling_returns.std() * 2.0  # 2x volatility
        df['take_profit_2x'] = rolling_returns.std() * 3.0  # 3x volatility
        
        # Calculate win rates for different levels
        results = []
        for i in range(lookback, len(df) - lookback, 10):  # Sample every 10 minutes
            current_price = df['price'].iloc[i]
            
            # Simulate trades with different SL/TP levels
            for sl_mult in [1.0, 1.5, 2.0]:
                for tp_mult in [1.0, 2.0, 3.0]:
                    sl_level = -rolling_returns.std().iloc[i] * sl_mult
                    tp_level = rolling_returns.std().iloc[i] * tp_mult
                    
                    # Check future prices
                    future_returns = (df['price'].iloc[i:i+lookback] / current_price - 1)
                    
                    # Did we hit stop loss first?
                    sl_hit = (future_returns < sl_level).any()
                    tp_hit = (future_returns > tp_level).any()
                    
                    if sl_hit and tp_hit:
                        # Which came first?
                        sl_idx = future_returns[future_returns < sl_level].index[0]
                        tp_idx = future_returns[future_returns > tp_level].index[0]
                        win = tp_idx < sl_idx
                    else:
                        win = tp_hit and not sl_hit
                    
                    results.append({
                        'sl_multiplier': sl_mult,
                        'tp_multiplier': tp_mult,
                        'win': win,
                        'risk_reward_ratio': tp_mult / sl_mult
                    })
        
        # Aggregate results
        results_df = pd.DataFrame(results)
        optimal_levels = results_df.groupby(['sl_multiplier', 'tp_multiplier']).agg({
            'win': ['mean', 'count'],
            'risk_reward_ratio': 'first'
        }).reset_index()
        
        return optimal_levels
    
    def calculate_order_flow_imbalance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate order flow imbalance using price movements and volume proxy
        """
        df = df.copy()
        
        # Classify trades as buy or sell based on price movement
        df['price_change'] = df['price'].diff()
        df['volume_proxy'] = df['price'].pct_change().abs()  # Volatility as volume proxy
        
        # Buy volume: positive price changes
        df['buy_volume'] = np.where(df['price_change'] > 0, df['volume_proxy'], 0)
        # Sell volume: negative price changes
        df['sell_volume'] = np.where(df['price_change'] < 0, df['volume_proxy'], 0)
        
        # Calculate rolling imbalance
        df['buy_volume_rolling'] = df['buy_volume'].rolling(window).sum()
        df['sell_volume_rolling'] = df['sell_volume'].rolling(window).sum()
        
        # Order flow imbalance
        total_volume = df['buy_volume_rolling'] + df['sell_volume_rolling']
        df['order_flow_imbalance'] = np.where(
            total_volume > 0,
            (df['buy_volume_rolling'] - df['sell_volume_rolling']) / total_volume,
            0
        )
        
        # Cumulative volume delta
        df['cumulative_delta'] = (df['buy_volume'] - df['sell_volume']).cumsum()
        
        return df[['datetime', 'price', 'order_flow_imbalance', 'cumulative_delta']]
    
    def vwap_analysis(self, df: pd.DataFrame, anchors: List[int] = [60, 240, 480]) -> pd.DataFrame:
        """
        Calculate VWAP (Volume Weighted Average Price) with multiple anchor points
        Using volatility as volume proxy
        """
        df = df.copy()
        df['volume_proxy'] = df['price'].pct_change().abs() + 1e-10
        
        for anchor in anchors:
            # Calculate VWAP for each anchor period
            df[f'vwap_{anchor}'] = (
                (df['price'] * df['volume_proxy']).rolling(anchor).sum() / 
                df['volume_proxy'].rolling(anchor).sum()
            )
            
            # VWAP bands (standard deviation)
            df[f'vwap_upper_{anchor}'] = df[f'vwap_{anchor}'] + df['price'].rolling(anchor).std()
            df[f'vwap_lower_{anchor}'] = df[f'vwap_{anchor}'] - df['price'].rolling(anchor).std()
            
            # Distance from VWAP (normalized)
            df[f'vwap_distance_{anchor}'] = (df['price'] - df[f'vwap_{anchor}']) / df[f'vwap_{anchor}']
        
        return df
    
    def market_profile_tpo(self, df: pd.DataFrame, tpo_size: int = 30, 
                          value_area_pct: float = 0.70) -> Dict:
        """
        Time Price Opportunity (TPO) Market Profile Analysis
        """
        # Define price buckets
        price_range = df['price'].max() - df['price'].min()
        n_buckets = max(30, int(len(df) / tpo_size))
        price_buckets = pd.cut(df['price'], bins=n_buckets)
        
        # Count time spent at each price level
        tpo_profile = df.groupby(price_buckets).size()
        tpo_profile_pct = tpo_profile / tpo_profile.sum()
        
        # Calculate Value Area (70% of time)
        tpo_cumsum = tpo_profile_pct.sort_values(ascending=False).cumsum()
        value_area_buckets = tpo_cumsum[tpo_cumsum <= value_area_pct].index
        
        # Find POC (Point of Control) - price with most time
        poc_bucket = tpo_profile.idxmax()
        poc_price = poc_bucket.mid
        
        # Value Area High and Low
        value_area_prices = [bucket.mid for bucket in value_area_buckets]
        vah = max(value_area_prices) if value_area_prices else df['price'].max()
        val = min(value_area_prices) if value_area_prices else df['price'].min()
        
        return {
            'tpo_profile': tpo_profile,
            'poc': poc_price,
            'value_area_high': vah,
            'value_area_low': val,
            'value_area_range': vah - val,
            'profile_df': pd.DataFrame({
                'price_bucket': tpo_profile.index,
                'time_count': tpo_profile.values,
                'percentage': tpo_profile_pct.values * 100
            })
        }
    
    def elliott_wave_detection(self, df: pd.DataFrame, min_wave_size: float = 0.05) -> Dict:
        """
        Simplified Elliott Wave pattern detection
        """
        # Find local maxima and minima
        prices = df['price'].values
        
        # Use scipy to find peaks and troughs
        peaks, _ = signal.find_peaks(prices, distance=20)
        troughs, _ = signal.find_peaks(-prices, distance=20)
        
        # Combine and sort
        extrema = sorted(list(peaks) + list(troughs))
        
        if len(extrema) < 5:
            return {'waves_detected': False}
        
        # Identify potential 5-wave patterns
        waves = []
        for i in range(len(extrema) - 4):
            # Get 5 consecutive extrema
            wave_points = extrema[i:i+5]
            wave_prices = [prices[idx] for idx in wave_points]
            
            # Check if it forms a valid Elliott Wave pattern
            # Simplified: just check if it's trending with corrections
            if len(wave_prices) == 5:
                # Calculate moves
                move1 = wave_prices[1] - wave_prices[0]
                move2 = wave_prices[2] - wave_prices[1]
                move3 = wave_prices[3] - wave_prices[2]
                move4 = wave_prices[4] - wave_prices[3]
                
                # Basic Elliott Wave rules (simplified)
                # Wave 2 should not retrace more than 100% of wave 1
                # Wave 3 should be the longest
                # Wave 4 should not overlap wave 1
                
                if abs(move1) > prices[wave_points[0]] * min_wave_size:
                    waves.append({
                        'start_idx': wave_points[0],
                        'end_idx': wave_points[4],
                        'start_time': df['datetime'].iloc[wave_points[0]],
                        'end_time': df['datetime'].iloc[wave_points[4]],
                        'wave_1': move1,
                        'wave_2': move2,
                        'wave_3': move3,
                        'wave_4': move4,
                        'total_move': wave_prices[4] - wave_prices[0]
                    })
        
        return {
            'waves_detected': len(waves) > 0,
            'wave_count': len(waves),
            'waves': waves[:5]  # Return top 5 waves
        }
    
    def calculate_market_efficiency(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate Kaufman's Efficiency Ratio and Fractal Efficiency
        """
        df = df.copy()
        
        # Kaufman's Efficiency Ratio
        df['price_change'] = df['price'].diff(window).abs()
        df['path_sum'] = df['price'].diff().abs().rolling(window).sum()
        df['efficiency_ratio'] = df['price_change'] / (df['path_sum'] + 1e-10)
        
        # Fractal Dimension (simplified)
        # Using Higuchi method
        df['fractal_dimension'] = df['price'].rolling(window).apply(
            lambda x: self._calculate_fractal_dimension(x.values) if len(x) == window else np.nan
        )
        
        # Market efficiency score (combination)
        df['market_efficiency'] = (df['efficiency_ratio'] + (2 - df['fractal_dimension'])) / 2
        
        return df[['datetime', 'price', 'efficiency_ratio', 'fractal_dimension', 'market_efficiency']]
    
    def _calculate_fractal_dimension(self, series: np.ndarray, kmax: int = 5) -> float:
        """Helper function to calculate fractal dimension"""
        N = len(series)
        L = []
        
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(series[m + i * k] - series[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / (k * int((N - m) / k))
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        
        # Linear regression on log-log plot
        if len(L) > 1:
            x = np.log(range(1, kmax + 1))
            y = np.log(L)
            
            # Remove any inf or nan
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 1:
                slope = np.polyfit(x[mask], y[mask], 1)[0]
                return -slope
        
        return 1.5  # Default fractal dimension
    
    def entropy_analysis(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate price entropy and information content
        Higher entropy = more random/unpredictable
        """
        df = df.copy()
        
        # Discretize returns into bins
        df['returns'] = df['price'].pct_change()
        
        def calc_entropy(returns):
            if len(returns) < 2:
                return np.nan
            # Discretize into 10 bins
            hist, _ = np.histogram(returns.dropna(), bins=10)
            # Normalize
            hist = hist + 1e-10  # Avoid log(0)
            prob = hist / hist.sum()
            return entropy(prob)
        
        # Rolling entropy
        df['price_entropy'] = df['returns'].rolling(window).apply(calc_entropy)
        
        # Relative entropy (current vs historical)
        historical_returns = df['returns'].dropna()
        if len(historical_returns) > window:
            hist_hist, bins = np.histogram(historical_returns, bins=10)
            hist_prob = hist_hist / hist_hist.sum()
            
            df['relative_entropy'] = df['returns'].rolling(window).apply(
                lambda x: entropy(hist_prob, np.histogram(x, bins=bins)[0] / len(x) + 1e-10)
            )
        
        return df[['datetime', 'price', 'price_entropy', 'relative_entropy']]
    
    def liquidity_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze liquidity metrics using price impact and volatility
        """
        df = df.copy()
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Kyle's Lambda (price impact)
        df['volume_proxy'] = df['returns'].abs()
        df['signed_volume'] = np.sign(df['returns']) * df['volume_proxy']
        
        # Amihud illiquidity
        df['amihud'] = df['returns'].abs() / (df['volume_proxy'] + 1e-10)
        
        # Estimate bid-ask spread using Roll's measure
        returns = df['returns'].dropna()
        if len(returns) > 2:
            cov_returns = returns.cov(returns.shift(1))
            roll_spread = 2 * np.sqrt(-cov_returns) if cov_returns < 0 else 0
        else:
            roll_spread = 0
        
        # Liquidity score (lower is better)
        liquidity_score = (
            df['amihud'].mean() * 0.4 +
            roll_spread * 0.3 +
            df['returns'].std() * 0.3
        )
        
        return {
            'amihud_illiquidity': df['amihud'].mean(),
            'roll_spread_estimate': roll_spread,
            'avg_price_impact': df['volume_proxy'].mean(),
            'liquidity_score': liquidity_score,
            'liquidity_df': df[['datetime', 'price', 'amihud', 'volume_proxy']]
        }
    
    def advanced_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced momentum indicators
        """
        df = df.copy()
        
        # Rate of Change (ROC) multiple timeframes
        for period in [10, 30, 60]:
            df[f'roc_{period}'] = ((df['price'] / df['price'].shift(period)) - 1) * 100
        
        # Momentum oscillator
        df['momentum_oscillator'] = df['price'] - df['price'].shift(10)
        
        # Coppock Curve (momentum indicator)
        df['roc_14'] = ((df['price'] / df['price'].shift(14)) - 1) * 100
        df['roc_11'] = ((df['price'] / df['price'].shift(11)) - 1) * 100
        df['coppock'] = (df['roc_14'] + df['roc_11']).rolling(10).mean()
        
        # Know Sure Thing (KST)
        # Four different ROCs with different weights
        rocma1 = ((df['price'] / df['price'].shift(10)) - 1).rolling(10).mean()
        rocma2 = ((df['price'] / df['price'].shift(15)) - 1).rolling(10).mean()
        rocma3 = ((df['price'] / df['price'].shift(20)) - 1).rolling(10).mean()
        rocma4 = ((df['price'] / df['price'].shift(30)) - 1).rolling(15).mean()
        
        df['kst'] = (rocma1 * 1) + (rocma2 * 2) + (rocma3 * 3) + (rocma4 * 4)
        df['kst_signal'] = df['kst'].rolling(9).mean()
        
        # TRIX - Triple exponential average
        ema1 = df['price'].ewm(span=14, adjust=False).mean()
        ema2 = ema1.ewm(span=14, adjust=False).mean()
        ema3 = ema2.ewm(span=14, adjust=False).mean()
        df['trix'] = (ema3.pct_change() * 10000)
        
        return df
    
    def calculate_risk_metrics(self, df: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Calculate comprehensive risk metrics
        """
        returns = df['price'].pct_change().dropna()
        
        risk_metrics = {}
        
        # Value at Risk (VaR) - Historical method
        for conf in confidence_levels:
            var_level = returns.quantile(1 - conf)
            risk_metrics[f'var_{int(conf*100)}'] = var_level
        
        # Conditional VaR (CVaR) / Expected Shortfall
        for conf in confidence_levels:
            var_level = returns.quantile(1 - conf)
            cvar = returns[returns <= var_level].mean()
            risk_metrics[f'cvar_{int(conf*100)}'] = cvar
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        risk_metrics['max_drawdown'] = drawdown.min()
        risk_metrics['avg_drawdown'] = drawdown[drawdown < 0].mean()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        risk_metrics['downside_deviation'] = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Omega ratio (probability weighted ratio of gains vs losses)
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() > 0:
            risk_metrics['omega_ratio'] = gains.sum() / losses.sum()
        else:
            risk_metrics['omega_ratio'] = np.inf
        
        # Ulcer Index (measures downside volatility)
        draw