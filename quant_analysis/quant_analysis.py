"""
Quantitative analysis module for memecoin trading
Professional financial market analysis tools
"""

import polars as pl
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QuantAnalysis:
    """Professional quantitative analysis for memecoin trading"""
    
    def __init__(self):
        self.risk_free_rate = 0  # Crypto markets, no risk-free rate
        
    def calculate_sharpe_ratio(self, returns: pl.Series, periods_per_year: int = 525600) -> float:
        """
        Calculate Sharpe ratio (annualized)
        periods_per_year: 525600 for minute data (365.25 * 24 * 60)
        """
        if len(returns) < 2:
            return np.nan
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return np.nan
            
        sharpe = (mean_return - self.risk_free_rate) / std_return
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)
        
        return annualized_sharpe
    
    def calculate_sortino_ratio(self, returns: pl.Series, periods_per_year: int = 525600) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) < 2:
            return np.nan
            
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
            
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.nan
            
        sortino = (mean_return - self.risk_free_rate) / downside_std
        annualized_sortino = sortino * np.sqrt(periods_per_year)
        
        return annualized_sortino
    
    def calculate_calmar_ratio(self, df: pl.DataFrame, periods_per_year: int = 525600) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        returns = df['price'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        
        # Annualized return
        total_return = cumulative_returns.iloc[-1] - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Max drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if max_drawdown == 0:
            return np.nan
            
        return annualized_return / abs(max_drawdown)
    
    def optimal_entry_exit_matrix(self, df: pl.DataFrame, 
                                  entry_windows: List[int] = [5, 10, 15, 30, 60],
                                  exit_windows: List[int] = [5, 10, 15, 30, 60],
                                  momentum_threshold: float = 0.0) -> pl.DataFrame:
        """
        Calculate optimal entry/exit matrix based on different time windows
        Returns a matrix of average returns for each entry/exit combination
        
        Entry window: lookback period for momentum calculation
        Exit window: holding period after entry
        momentum_threshold: minimum momentum required to enter (default 0%)
        """
        matrix = pl.DataFrame(index=entry_windows, columns=exit_windows)
        trade_counts = pl.DataFrame(index=entry_windows, columns=exit_windows)
        
        for entry_window in entry_windows:
            for exit_window in exit_windows:
                returns = []
                
                # Use while loop to properly skip after trades
                i = entry_window
                while i < len(df) - exit_window:
                    # Calculate momentum over the entry window
                    entry_momentum = (df['price'].iloc[i] / df['price'].iloc[i-entry_window] - 1)
                    
                    # Entry signal: momentum above threshold
                    if entry_momentum > momentum_threshold:
                        entry_price = df['price'].iloc[i]
                        exit_price = df['price'].iloc[i + exit_window]
                        trade_return = (exit_price / entry_price - 1) * 100
                        returns.append(trade_return)
                        
                        # Skip ahead to avoid overlapping trades
                        i += exit_window
                    else:
                        # Move to next minute if no trade
                        i += 1
                
                # Store results
                if returns:
                    matrix.loc[entry_window, exit_window] = np.mean(returns)
                    trade_counts.loc[entry_window, exit_window] = len(returns)
                else:
                    matrix.loc[entry_window, exit_window] = 0
                    trade_counts.loc[entry_window, exit_window] = 0
        
        # Store trade counts as an attribute for analysis
        self.last_trade_counts = trade_counts
        
        return matrix.astype(float)
    
    def temporal_risk_reward_analysis(self, df: pl.DataFrame, 
                                     time_horizons: List[int] = [5, 15, 30, 60, 120]) -> pl.DataFrame:
        """
        Analyze risk/reward ratios across different time horizons
        """
        results = []
        
        for horizon in time_horizons:
            # Calculate rolling returns for this horizon
            rolling_returns = df['price'].pct_change(horizon)
            
            # Separate positive and negative returns
            positive_returns = rolling_returns[rolling_returns > 0]
            negative_returns = rolling_returns[rolling_returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_gain = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                win_rate = len(positive_returns) / len(rolling_returns.dropna())
                
                # Risk/Reward ratio
                risk_reward = avg_gain / avg_loss if avg_loss > 0 else np.inf
                
                # Expected value
                expected_value = (win_rate * avg_gain) - ((1 - win_rate) * avg_loss)
                
                results.append({
                    'horizon_minutes': horizon,
                    'win_rate': win_rate * 100,
                    'avg_gain_%': avg_gain * 100,
                    'avg_loss_%': avg_loss * 100,
                    'risk_reward_ratio': risk_reward,
                    'expected_value_%': expected_value * 100,
                    'sharpe_ratio': self.calculate_sharpe_ratio(rolling_returns.dropna())
                })
        
        return pl.DataFrame(results)
    
    def volume_profile_analysis(self, df: pl.DataFrame, n_bins: int = 50) -> Dict:
        """
        Analyze volume profile (using volatility as proxy for volume)
        """
        # Use rolling volatility as volume proxy
        df['volume_proxy'] = df['price'].pct_change().rolling(10).std()
        
        # Create price bins
        price_bins = pd.qcut(df['price'], n_bins, duplicates='drop')
        
        # Calculate volume profile
        volume_profile = df.groupby(price_bins)['volume_proxy'].agg(['sum', 'mean', 'count'])
        volume_profile['price_level'] = volume_profile.index.map(lambda x: x.mid)
        
        # Find high volume nodes (support/resistance)
        high_volume_threshold = volume_profile['sum'].quantile(0.7)
        high_volume_nodes = volume_profile[volume_profile['sum'] > high_volume_threshold]
        
        return {
            'volume_profile': volume_profile,
            'high_volume_nodes': high_volume_nodes,
            'poc': volume_profile.loc[volume_profile['sum'].idxmax(), 'price_level']  # Point of Control
        }
    
    def market_regime_detection(self, df: pl.DataFrame, lookback: int = 60) -> pl.DataFrame:
        """
        Detect market regimes (trending up, trending down, ranging)
        """
        df = df.copy()
        
        # Calculate indicators
        df['returns'] = df['price'].pct_change()
        df['sma_short'] = df['price'].rolling(lookback // 4).mean()
        df['sma_long'] = df['price'].rolling(lookback).mean()
        df['volatility'] = df['returns'].rolling(lookback // 2).std()
        
        # ADX for trend strength (simplified)
        df['price_change'] = df['price'].diff()
        df['high'] = df['price'].rolling(2).max()
        df['low'] = df['price'].rolling(2).min()
        
        # Directional indicators
        df['plus_dm'] = np.where((df['high'].diff() > df['low'].diff().abs()) & 
                                 (df['high'].diff() > 0), df['high'].diff(), 0)
        df['minus_dm'] = np.where((df['low'].diff().abs() > df['high'].diff()) & 
                                  (df['low'].diff() < 0), df['low'].diff().abs(), 0)
        
        # Trend classification
        conditions = [
            (df['sma_short'] > df['sma_long']) & (df['volatility'] < df['volatility'].rolling(100).mean()),
            (df['sma_short'] < df['sma_long']) & (df['volatility'] < df['volatility'].rolling(100).mean()),
            (df['volatility'] > df['volatility'].rolling(100).mean())
        ]
        
        choices = ['uptrend', 'downtrend', 'high_volatility']
        df['regime'] = np.select(conditions, choices, default='ranging')
        
        return df[['datetime', 'price', 'regime', 'volatility']]
    
    def calculate_hurst_exponent(self, price_series: pl.Series, lags: int = 20) -> float:
        """
        Calculate Hurst exponent to determine if series is trending or mean-reverting
        H > 0.5: Trending
        H = 0.5: Random walk
        H < 0.5: Mean reverting
        """
        if len(price_series) < lags * 2:
            return np.nan
            
        # Calculate log returns
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        
        # Range of lags
        lag_range = range(2, lags)
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lag_range:
            # Divide series into chunks
            chunks = [log_returns[i:i+lag] for i in range(0, len(log_returns)-lag+1, lag)]
            
            rs_lag = []
            for chunk in chunks:
                if len(chunk) == lag:
                    # Demean the chunk
                    demeaned = chunk - chunk.mean()
                    # Cumulative sum
                    cumsum = demeaned.cumsum()
                    # Range
                    R = cumsum.max() - cumsum.min()
                    # Standard deviation
                    S = chunk.std()
                    
                    if S > 0:
                        rs_lag.append(R / S)
            
            if rs_lag:
                rs_values.append(np.mean(rs_lag))
        
        if len(rs_values) < 2:
            return np.nan
            
        # Linear regression of log(R/S) on log(lag)
        log_lags = np.log(list(lag_range)[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove any inf or nan values
        mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if mask.sum() < 2:
            return np.nan
            
        slope, _, _, _, _ = stats.linregress(log_lags[mask], log_rs[mask])
        
        return slope
    
    def microstructure_analysis(self, df: pl.DataFrame) -> Dict:
        """
        Analyze market microstructure: bid-ask spread proxy, price impact, etc.
        """
        # Calculate high-frequency metrics
        df = df.copy()
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Realized volatility (1-minute)
        df['realized_vol'] = df['returns'].rolling(60).std() * np.sqrt(60)
        
        # Amihud illiquidity measure (price impact proxy)
        df['volume_proxy'] = df['returns'].abs()  # Use absolute returns as volume proxy
        df['amihud'] = df['returns'].abs() / (df['volume_proxy'] + 1e-10)
        
        # Roll's bid-ask spread estimator
        # Based on serial covariance of returns
        returns = df['returns'].dropna()
        if len(returns) > 1:
            cov = returns.cov(returns.shift(1))
            spread_estimate = 2 * np.sqrt(-cov) if cov < 0 else 0
        else:
            spread_estimate = 0
        
        # Kyle's lambda (price impact coefficient)
        # Regression of price changes on signed volume
        df['signed_volume'] = np.sign(df['returns']) * df['volume_proxy']
        
        # Remove outliers for regression
        price_changes = df['returns'].dropna()
        signed_volume = df['signed_volume'].dropna()
        
        if len(price_changes) > 10:
            # Remove top and bottom 1%
            lower = np.percentile(price_changes, 1)
            upper = np.percentile(price_changes, 99)
            mask = (price_changes > lower) & (price_changes < upper)
            
            if mask.sum() > 10:
                kyle_lambda, _, _, _, _ = stats.linregress(signed_volume[mask], price_changes[mask])
            else:
                kyle_lambda = np.nan
        else:
            kyle_lambda = np.nan
        
        return {
            'avg_realized_volatility': df['realized_vol'].mean(),
            'bid_ask_spread_estimate': spread_estimate,
            'kyle_lambda': kyle_lambda,
            'avg_amihud_illiquidity': df['amihud'].mean(),
            'volatility_of_volatility': df['realized_vol'].std()
        }
    
    def calculate_information_ratio(self, returns: pl.Series, benchmark_returns: pl.Series = None) -> float:
        """
        Calculate Information Ratio (active return / tracking error)
        If no benchmark provided, use zero returns
        """
        if benchmark_returns is None:
            benchmark_returns = pl.Series(0, index=returns.index)
        
        active_returns = returns - benchmark_returns
        
        if len(active_returns) < 2:
            return np.nan
            
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return np.nan
            
        return active_returns.mean() / tracking_error * np.sqrt(252 * 24 * 60)  # Annualized
    
    def momentum_quality_score(self, df: pl.DataFrame, lookbacks: List[int] = [10, 30, 60]) -> pl.DataFrame:
        """
        Calculate momentum quality score based on multiple timeframes
        Higher score = better momentum quality
        """
        df = df.copy()
        
        for lookback in lookbacks:
            # Calculate momentum
            df[f'momentum_{lookback}'] = df['price'].pct_change(lookback)
            
            # Calculate smoothness (inverse of volatility during the move)
            rolling_returns = df['price'].pct_change().rolling(lookback)
            df[f'smoothness_{lookback}'] = 1 / (rolling_returns.std() + 1e-10)
            
            # Consistency score (% of positive returns during uptrend)
            df[f'consistency_{lookback}'] = rolling_returns.apply(lambda x: (x > 0).sum() / len(x))
        
        # Combine into quality score
        momentum_cols = [f'momentum_{lb}' for lb in lookbacks]
        smoothness_cols = [f'smoothness_{lb}' for lb in lookbacks]
        consistency_cols = [f'consistency_{lb}' for lb in lookbacks]
        
        # Normalize each component
        for cols in [momentum_cols, smoothness_cols, consistency_cols]:
            for col in cols:
                if col in df.columns:
                    df[f'{col}_norm'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        
        # Calculate composite score
        norm_cols = [f'{col}_norm' for cols in [momentum_cols, smoothness_cols, consistency_cols] 
                    for col in cols if f'{col}_norm' in df.columns]
        
        df['momentum_quality_score'] = df[norm_cols].mean(axis=1)
        
        return df[['datetime', 'price', 'momentum_quality_score'] + momentum_cols]
    
    def enhanced_entry_exit_analysis(self, df: pl.DataFrame,
                                   entry_windows: List[int] = [5, 10, 15, 30, 60],
                                   exit_windows: List[int] = [5, 10, 15, 30, 60],
                                   entry_method: str = 'momentum',
                                   momentum_threshold: float = 0.01) -> Dict:
        """
        Enhanced entry/exit analysis with multiple strategies
        
        entry_method: 'momentum', 'breakout', 'mean_reversion'
        momentum_threshold: minimum momentum for entry (1% = 0.01)
        """
        results = {}
        
        if entry_method == 'momentum':
            # Standard momentum strategy
            matrix = self.optimal_entry_exit_matrix(df, entry_windows, exit_windows, momentum_threshold)
            results['matrix'] = matrix
            results['trade_counts'] = self.last_trade_counts
            
        elif entry_method == 'breakout':
            # Breakout strategy: enter when price breaks above recent high
            matrix = pl.DataFrame(index=entry_windows, columns=exit_windows)
            trade_counts = pl.DataFrame(index=entry_windows, columns=exit_windows)
            
            for entry_window in entry_windows:
                for exit_window in exit_windows:
                    returns = []
                    i = entry_window
                    
                    while i < len(df) - exit_window:
                        # Check if current price breaks above window high
                        window_high = df['price'].iloc[i-entry_window:i].max()
                        current_price = df['price'].iloc[i]
                        
                        if current_price > window_high * (1 + momentum_threshold):
                            entry_price = current_price
                            exit_price = df['price'].iloc[i + exit_window]
                            trade_return = (exit_price / entry_price - 1) * 100
                            returns.append(trade_return)
                            i += exit_window
                        else:
                            i += 1
                    
                    matrix.loc[entry_window, exit_window] = np.mean(returns) if returns else 0
                    trade_counts.loc[entry_window, exit_window] = len(returns)
            
            results['matrix'] = matrix
            results['trade_counts'] = trade_counts
            
        elif entry_method == 'mean_reversion':
            # Mean reversion: enter when price is below moving average
            matrix = pl.DataFrame(index=entry_windows, columns=exit_windows)
            trade_counts = pl.DataFrame(index=entry_windows, columns=exit_windows)
            
            for entry_window in entry_windows:
                # Calculate moving average
                df[f'ma_{entry_window}'] = df['price'].rolling(entry_window).mean()
                
                for exit_window in exit_windows:
                    returns = []
                    i = entry_window
                    
                    while i < len(df) - exit_window:
                        # Check if price is below MA (oversold)
                        current_price = df['price'].iloc[i]
                        ma_price = df[f'ma_{entry_window}'].iloc[i]
                        
                        if current_price < ma_price * (1 - momentum_threshold) and not pd.isna(ma_price):
                            entry_price = current_price
                            exit_price = df['price'].iloc[i + exit_window]
                            trade_return = (exit_price / entry_price - 1) * 100
                            returns.append(trade_return)
                            i += exit_window
                        else:
                            i += 1
                    
                    matrix.loc[entry_window, exit_window] = np.mean(returns) if returns else 0
                    trade_counts.loc[entry_window, exit_window] = len(returns)
            
            results['matrix'] = matrix
            results['trade_counts'] = trade_counts
        
        # Calculate best combination
        best_entry, best_exit = np.unravel_index(results['matrix'].values.argmax(), 
                                                results['matrix'].values.shape)
        results['best_combination'] = {
            'entry_window': entry_windows[best_entry],
            'exit_window': exit_windows[best_exit],
            'expected_return': results['matrix'].values[best_entry, best_exit],
            'trade_count': results['trade_counts'].values[best_entry, best_exit]
        }
        
        return results