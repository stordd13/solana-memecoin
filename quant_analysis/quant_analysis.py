"""
Quantitative analysis module for memecoin trading
Professional financial market analysis tools
"""

import polars as pl
import numpy as np
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
            rolling_returns = df['price'].pct_change(horizon).drop_nulls()
            
            # Separate positive and negative returns using filter
            positive_returns = rolling_returns.filter(rolling_returns > 0)
            negative_returns = rolling_returns.filter(rolling_returns < 0)
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_gain = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                win_rate = len(positive_returns) / len(rolling_returns)
                
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
                    'sharpe_ratio': self.calculate_sharpe_ratio(rolling_returns)
                })
        
        return pl.DataFrame(results)
    
    def volume_profile_analysis(self, df: pl.DataFrame, n_bins: int = 50) -> Dict:
        """
        Analyze volume profile (using volatility as proxy for volume)
        """
        # Use rolling volatility as volume proxy
        df = df.with_columns([
            df['price'].pct_change().rolling_std(10, min_periods=1).alias('volume_proxy')
        ])
        
        # Create price bins using quantiles
        price_quantiles = df['price'].quantile([i/n_bins for i in range(n_bins+1)])
        
        # Create bins and calculate volume profile
        volume_profile_data = []
        for i in range(len(price_quantiles)-1):
            lower = price_quantiles[i]
            upper = price_quantiles[i+1]
            price_mid = (lower + upper) / 2
            
            # Filter data in this price range
            mask = (df['price'] >= lower) & (df['price'] < upper) if i < len(price_quantiles)-2 else (df['price'] >= lower) & (df['price'] <= upper)
            bin_data = df.filter(mask)
            
            if len(bin_data) > 0:
                volume_sum = bin_data['volume_proxy'].sum()
                volume_mean = bin_data['volume_proxy'].mean()
                count = len(bin_data)
            else:
                volume_sum = 0
                volume_mean = 0
                count = 0
            
            volume_profile_data.append({
                'price_level': price_mid,
                'volume_sum': volume_sum,
                'volume_mean': volume_mean,
                'count': count
            })
        
        volume_profile_df = pl.DataFrame(volume_profile_data)
        
        # Find high volume nodes (support/resistance)
        if len(volume_profile_df) > 0:
            high_volume_threshold = volume_profile_df['volume_sum'].quantile(0.7)
            high_volume_nodes = volume_profile_df.filter(pl.col('volume_sum') > high_volume_threshold)
            
            # Point of Control (price with most volume)
            poc_idx = volume_profile_df['volume_sum'].arg_max()
            poc_price = volume_profile_df['price_level'][poc_idx] if poc_idx is not None else 0
        else:
            high_volume_nodes = pl.DataFrame()
            poc_price = 0
        
        return {
            'volume_profile': volume_profile_df,
            'high_volume_nodes': high_volume_nodes,
            'poc': poc_price
        }
    
    def market_regime_detection(self, df: pl.DataFrame, lookback: int = 60) -> pl.DataFrame:
        """
        Detect market regimes (trending up, trending down, ranging)
        """
        # Calculate indicators using Polars
        df = df.with_columns([
            df['price'].pct_change().alias('returns'),
            df['price'].rolling_mean(lookback // 4, min_periods=1).alias('sma_short'),
            df['price'].rolling_mean(lookback, min_periods=1).alias('sma_long'),
            df['price'].pct_change().rolling_std(lookback // 2, min_periods=1).alias('volatility'),
            df['price'].diff().alias('price_change'),
            df['price'].rolling_max(2, min_periods=1).alias('high'),
            df['price'].rolling_min(2, min_periods=1).alias('low')
        ])
        
        # Calculate directional indicators
        df = df.with_columns([
            df['high'].diff().alias('high_diff'),
            df['low'].diff().alias('low_diff')
        ])
        
        df = df.with_columns([
            pl.when(
                (pl.col('high_diff') > pl.col('low_diff').abs()) & 
                (pl.col('high_diff') > 0)
            ).then(pl.col('high_diff')).otherwise(0).alias('plus_dm'),
            pl.when(
                (pl.col('low_diff').abs() > pl.col('high_diff')) & 
                (pl.col('low_diff') < 0)
            ).then(pl.col('low_diff').abs()).otherwise(0).alias('minus_dm')
        ])
        
        # Calculate volatility mean for comparison
        df = df.with_columns([
            df['volatility'].rolling_mean(100, min_periods=1).alias('vol_mean')
        ])
        
        # Trend classification
        df = df.with_columns([
            pl.when(
                (pl.col('sma_short') > pl.col('sma_long')) & 
                (pl.col('volatility') < pl.col('vol_mean'))
            ).then(pl.lit('uptrend'))
            .when(
                (pl.col('sma_short') < pl.col('sma_long')) & 
                (pl.col('volatility') < pl.col('vol_mean'))
            ).then(pl.lit('downtrend'))
            .when(pl.col('volatility') > pl.col('vol_mean'))
            .then(pl.lit('high_volatility'))
            .otherwise(pl.lit('ranging'))
            .alias('regime')
        ])
        
        return df.select(['datetime', 'price', 'regime', 'volatility'])
    
    def calculate_hurst_exponent(self, price_series: pl.Series, lags: int = 20) -> float:
        """
        Calculate Hurst exponent to determine if series is trending or mean-reverting
        H > 0.5: Trending
        H = 0.5: Random walk
        H < 0.5: Mean reverting
        """
        if len(price_series) < lags * 2:
            return np.nan
            
        # Convert to numpy for easier manipulation
        prices = price_series.to_numpy()
        
        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        log_returns = log_returns[~np.isnan(log_returns)]
        
        if len(log_returns) < lags * 2:
            return np.nan
        
        # Range of lags
        lag_range = range(2, lags)
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lag_range:
            # Divide series into chunks
            n_chunks = len(log_returns) // lag
            if n_chunks == 0:
                continue
                
            rs_lag = []
            for i in range(n_chunks):
                chunk = log_returns[i*lag:(i+1)*lag]
                
                if len(chunk) == lag:
                    # Demean the chunk
                    chunk_mean = np.mean(chunk)
                    demeaned = chunk - chunk_mean
                    # Cumulative sum
                    cumsum = np.cumsum(demeaned)
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    # Standard deviation
                    S = np.std(chunk)
                    
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
        # Calculate high-frequency metrics using Polars
        df = df.with_columns([
            df['price'].pct_change().alias('returns'),
            (df['price'] / df['price'].shift(1)).log().alias('log_returns')
        ])
        
        # Realized volatility (1-minute)
        df = df.with_columns([
            (df['returns'].rolling_std(60, min_periods=1) * np.sqrt(60)).alias('realized_vol')
        ])
        
        # Amihud illiquidity measure (price impact proxy)
        df = df.with_columns([
            df['returns'].abs().alias('volume_proxy')
        ])
        
        df = df.with_columns([
            (df['returns'].abs() / (df['volume_proxy'] + 1e-10)).alias('amihud')
        ])
        
        # Roll's bid-ask spread estimator
        # Based on serial covariance of returns
        returns_clean = df['returns'].drop_nulls()
        if len(returns_clean) > 1:
            returns_np = returns_clean.to_numpy()
            returns_lagged = np.roll(returns_np, 1)[1:]  # Remove first element after shift
            returns_current = returns_np[1:]
            
            if len(returns_current) > 1:
                cov = np.cov(returns_current, returns_lagged)[0, 1]
                spread_estimate = 2 * np.sqrt(-cov) if cov < 0 else 0
            else:
                spread_estimate = 0
        else:
            spread_estimate = 0
        
        # Kyle's lambda (price impact coefficient)
        # Regression of price changes on signed volume
        df = df.with_columns([
            (df['returns'].sign() * df['volume_proxy']).alias('signed_volume')
        ])
        
        # Remove outliers for regression
        price_changes = df['returns'].drop_nulls().to_numpy()
        signed_volume = df['signed_volume'].drop_nulls().to_numpy()
        
        if len(price_changes) > 10 and len(signed_volume) > 10:
            # Ensure same length
            min_len = min(len(price_changes), len(signed_volume))
            price_changes = price_changes[:min_len]
            signed_volume = signed_volume[:min_len]
            
            # Remove top and bottom 1%
            lower = np.percentile(price_changes, 1)
            upper = np.percentile(price_changes, 99)
            mask = (price_changes > lower) & (price_changes < upper)
            
            if mask.sum() > 10:
                try:
                    kyle_lambda, _, _, _, _ = stats.linregress(signed_volume[mask], price_changes[mask])
                except:
                    kyle_lambda = np.nan
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
            benchmark_returns = pl.Series([0] * len(returns))
        
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
        # Calculate momentum for each lookback period
        momentum_exprs = []
        smoothness_exprs = []
        
        for lookback in lookbacks:
            momentum_exprs.append(
                (df['price'] / df['price'].shift(lookback) - 1).alias(f'momentum_{lookback}')
            )
            
            # Calculate smoothness (inverse of volatility during the move)
            smoothness_exprs.append(
                (1 / (df['price'].pct_change().rolling_std(lookback, min_periods=1) + 1e-10)).alias(f'smoothness_{lookback}')
            )
        
        df = df.with_columns(momentum_exprs + smoothness_exprs)
        
        # Calculate composite score (simple average of normalized components)
        momentum_cols = [f'momentum_{lb}' for lb in lookbacks]
        smoothness_cols = [f'smoothness_{lb}' for lb in lookbacks]
        
        # Create normalized versions
        norm_exprs = []
        for col in momentum_cols + smoothness_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            norm_exprs.append(
                ((df[col] - col_mean) / (col_std + 1e-10)).alias(f'{col}_norm')
            )
        
        df = df.with_columns(norm_exprs)
        
        # Calculate composite score
        norm_cols = [f'{col}_norm' for col in momentum_cols + smoothness_cols]
        df = df.with_columns([
            pl.concat_list([pl.col(col) for col in norm_cols]).list.mean().alias('momentum_quality_score')
        ])
        
        return df.select(['datetime', 'price', 'momentum_quality_score'] + momentum_cols)
    
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
            matrix_data = []
            trade_count_data = []
            
            prices = df['price'].to_numpy()
            
            for entry_window in entry_windows:
                matrix_row = []
                count_row = []
                
                for exit_window in exit_windows:
                    returns = []
                    i = entry_window
                    
                    while i < len(prices) - exit_window:
                        # Check if current price breaks above window high
                        window_high = np.max(prices[i-entry_window:i])
                        current_price = prices[i]
                        
                        if current_price > window_high * (1 + momentum_threshold):
                            entry_price = current_price
                            exit_price = prices[i + exit_window]
                            trade_return = (exit_price / entry_price - 1) * 100
                            returns.append(trade_return)
                            i += exit_window
                        else:
                            i += 1
                    
                    matrix_row.append(np.mean(returns) if returns else 0)
                    count_row.append(len(returns))
                
                matrix_data.append(matrix_row)
                trade_count_data.append(count_row)
            
            matrix = pl.DataFrame({str(col): [row[i] for row in matrix_data] for i, col in enumerate(exit_windows)})
            trade_counts = pl.DataFrame({str(col): [row[i] for row in trade_count_data] for i, col in enumerate(exit_windows)})
            
            results['matrix'] = matrix
            results['trade_counts'] = trade_counts
            
        elif entry_method == 'mean_reversion':
            # Mean reversion: enter when price is below moving average
            matrix_data = []
            trade_count_data = []
            
            prices = df['price'].to_numpy()
            
            for entry_window in entry_windows:
                # Calculate moving average
                ma_values = []
                for i in range(len(prices)):
                    if i >= entry_window - 1:
                        ma_values.append(np.mean(prices[i-entry_window+1:i+1]))
                    else:
                        ma_values.append(np.nan)
                
                matrix_row = []
                count_row = []
                
                for exit_window in exit_windows:
                    returns = []
                    i = entry_window
                    
                    while i < len(prices) - exit_window:
                        # Check if price is below MA (oversold)
                        current_price = prices[i]
                        ma_price = ma_values[i]
                        
                        if not np.isnan(ma_price) and current_price < ma_price * (1 - momentum_threshold):
                            entry_price = current_price
                            exit_price = prices[i + exit_window]
                            trade_return = (exit_price / entry_price - 1) * 100
                            returns.append(trade_return)
                            i += exit_window
                        else:
                            i += 1
                    
                    matrix_row.append(np.mean(returns) if returns else 0)
                    count_row.append(len(returns))
                
                matrix_data.append(matrix_row)
                trade_count_data.append(count_row)
            
            matrix = pl.DataFrame({str(col): [row[i] for row in matrix_data] for i, col in enumerate(exit_windows)})
            trade_counts = pl.DataFrame({str(col): [row[i] for row in trade_count_data] for i, col in enumerate(exit_windows)})
            
            results['matrix'] = matrix
            results['trade_counts'] = trade_counts
        
        # Calculate best combination
        matrix_np = results['matrix'].to_numpy()
        best_entry, best_exit = np.unravel_index(matrix_np.argmax(), matrix_np.shape)
        results['best_combination'] = {
            'entry_window': entry_windows[best_entry],
            'exit_window': exit_windows[best_exit],
            'expected_return': matrix_np[best_entry, best_exit],
            'trade_count': results['trade_counts'].to_numpy()[best_entry, best_exit]
        }
        
        return results