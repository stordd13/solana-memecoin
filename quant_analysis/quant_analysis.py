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
        
    def calculate_sharpe_ratio(self, returns: pl.Series, annualize: bool = False) -> float:
        """
        Calculate Sharpe ratio 
        For crypto analysis, we typically use non-annualized ratios for better interpretability
        
        annualize: If True, annualizes using sqrt(525600) - use sparingly for crypto
        """
        if len(returns) < 2:
            return 0.0
        
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Handle edge cases
            if std_return == 0 or std_return is None or np.isnan(std_return):
                return 0.0
            
            if mean_return is None or np.isnan(mean_return):
                return 0.0
                
            # Calculate raw Sharpe ratio (more appropriate for crypto)
            sharpe = (mean_return - self.risk_free_rate) / std_return
            
            # Only annualize if explicitly requested (not recommended for crypto)
            if annualize:
                # Use conservative annualization factor for crypto (daily equivalent)
                periods_per_year = 365  # Daily equivalent rather than minute-level
                annualized_sharpe = sharpe * np.sqrt(periods_per_year)
                sharpe = annualized_sharpe
            
            # Check for infinite or NaN result
            if np.isnan(sharpe) or np.isinf(sharpe):
                return 0.0
            
            # Cap extreme values for crypto analysis
            if abs(sharpe) > 10:  # Reasonable upper bound for crypto Sharpe ratios
                return 10.0 if sharpe > 0 else -10.0
            
            return sharpe
        except Exception:
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pl.Series, annualize: bool = False) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) < 2:
            return 0.0
            
        try:
            mean_return = returns.mean()
            downside_returns = returns.filter(returns < 0)
            
            if len(downside_returns) == 0:
                return 10.0  # All positive returns - cap at reasonable value
                
            downside_std = downside_returns.std()
            
            if downside_std == 0 or downside_std is None or np.isnan(downside_std):
                return 0.0
                
            sortino = (mean_return - self.risk_free_rate) / downside_std
            
            # Only annualize if explicitly requested
            if annualize:
                periods_per_year = 365  # Daily equivalent
                sortino = sortino * np.sqrt(periods_per_year)
            
            # Handle infinite or NaN results
            if np.isnan(sortino) or np.isinf(sortino):
                return 0.0
            
            # Cap extreme values
            if abs(sortino) > 10:
                return 10.0 if sortino > 0 else -10.0
            
            return sortino
        except Exception:
            return 0.0
    
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
        Returns consistent DataFrame structure even with insufficient data
        
        Enhanced to handle edge cases and provide more meaningful results for crypto analysis
        """
        results = []
        
        for horizon in time_horizons:
            try:
                # Minimum data requirement: need at least 3x the horizon for meaningful analysis
                min_data_points = max(horizon * 3, 50)
                
                if len(df) < min_data_points:
                    # Insufficient data for meaningful analysis
                    results.append({
                        'horizon_minutes': horizon,
                        'win_rate': 0.0,
                        'avg_gain_%': 0.0,
                        'avg_loss_%': 0.0,
                        'risk_reward_ratio': 0.0,
                        'expected_value_%': 0.0,
                        'sharpe_ratio': 0.0
                    })
                    continue
                
                # Calculate FORWARD-LOOKING returns for trading analysis
                # For each position i, calculate return from buying at i and selling at i+horizon
                prices = df['price'].to_list()
                forward_returns = []
                
                for i in range(len(prices) - horizon):
                    entry_price = prices[i]
                    exit_price = prices[i + horizon]
                    return_pct = (exit_price - entry_price) / entry_price
                    forward_returns.append(return_pct)
                
                if len(forward_returns) == 0:
                    rolling_returns = pl.Series([], dtype=pl.Float64)
                else:
                    rolling_returns = pl.Series(forward_returns)
                
                # Need minimum number of return observations
                if len(rolling_returns) < 10:
                    results.append({
                        'horizon_minutes': horizon,
                        'win_rate': 0.0,
                        'avg_gain_%': 0.0,
                        'avg_loss_%': 0.0,
                        'risk_reward_ratio': 0.0,
                        'expected_value_%': 0.0,
                        'sharpe_ratio': 0.0
                    })
                    continue
                
                # Separate positive and negative returns using filter
                positive_returns = rolling_returns.filter(rolling_returns > 0)
                negative_returns = rolling_returns.filter(rolling_returns < 0)
                zero_returns = rolling_returns.filter(rolling_returns == 0)
                
                total_returns = len(rolling_returns)
                pos_count = len(positive_returns)
                neg_count = len(negative_returns)
                zero_count = len(zero_returns)
                
                # Calculate metrics based on available data
                if pos_count > 0 and neg_count > 0:
                    # Normal case: both gains and losses
                    avg_gain = positive_returns.mean()
                    avg_loss = abs(negative_returns.mean())
                    win_rate = pos_count / total_returns
                    
                    # Risk/Reward ratio with safety check
                    risk_reward = avg_gain / avg_loss if avg_loss > 0 else 0.0
                    if np.isinf(risk_reward) or np.isnan(risk_reward):
                        risk_reward = 0.0
                    
                    # Expected value
                    expected_value = (win_rate * avg_gain) - ((1 - win_rate) * avg_loss)
                    
                elif pos_count > 0 and neg_count == 0:
                    # Only positive returns - handle carefully
                    avg_gain = positive_returns.mean()
                    avg_loss = 0.0
                    
                    # Adjust win rate based on data quality
                    if total_returns < 20:
                        # Very limited data - reduce confidence
                        win_rate = min(0.95, pos_count / total_returns)  # Cap at 95%
                    else:
                        win_rate = pos_count / total_returns
                    
                    risk_reward = 0.0  # Can't calculate without losses
                    expected_value = win_rate * avg_gain  # Only gains
                    
                elif pos_count == 0 and neg_count > 0:
                    # Only negative returns
                    avg_gain = 0.0
                    avg_loss = abs(negative_returns.mean())
                    win_rate = 0.0
                    risk_reward = 0.0
                    expected_value = -avg_loss  # Only losses
                    
                else:
                    # No meaningful returns (all zeros or very small movements)
                    avg_gain = 0.0
                    avg_loss = 0.0
                    win_rate = 0.0
                    risk_reward = 0.0
                    expected_value = 0.0
                
                # Calculate Sharpe ratio safely
                sharpe = self.calculate_sharpe_ratio(rolling_returns)
                if np.isnan(sharpe) or np.isinf(sharpe):
                    sharpe = 0.0
                
                # Additional data quality check
                data_quality_score = min(1.0, total_returns / 50.0)  # Quality score based on sample size
                
                results.append({
                    'horizon_minutes': horizon,
                    'win_rate': win_rate * 100,
                    'avg_gain_%': avg_gain * 100,
                    'avg_loss_%': avg_loss * 100,
                    'risk_reward_ratio': risk_reward,
                    'expected_value_%': expected_value * 100,
                    'sharpe_ratio': sharpe
                })
                    
            except Exception as e:
                # Handle any unexpected errors for this horizon
                results.append({
                    'horizon_minutes': horizon,
                    'win_rate': 0.0,
                    'avg_gain_%': 0.0,
                    'avg_loss_%': 0.0,
                    'risk_reward_ratio': 0.0,
                    'expected_value_%': 0.0,
                    'sharpe_ratio': 0.0
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
        Analyze market microstructure: bid-ask spread proxy, price impact, liquidity metrics
        
        Key Metrics:
        - Roll's bid-ask spread estimator (from return serial covariance)
        - Kyle's lambda (price impact coefficient)
        - Amihud illiquidity measure (modified for crypto without volume)
        - Realized volatility and volatility clustering
        - Price efficiency and market quality indicators
        """
        # Calculate high-frequency metrics using Polars
        df_analysis = df.with_columns([
            df['price'].pct_change().alias('returns'),
            (df['price'] / df['price'].shift(1)).log().alias('log_returns')
        ])
        
        # Realized volatility (multiple windows for robustness)
        df_analysis = df_analysis.with_columns([
            # 1-hour realized volatility (annualized)
            (df_analysis['returns'].rolling_std(60, min_periods=30) * np.sqrt(525600)).alias('realized_vol_1h'),
            # 4-hour realized volatility (annualized)  
            (df_analysis['returns'].rolling_std(240, min_periods=120) * np.sqrt(525600)).alias('realized_vol_4h'),
            # Intraday volatility (within current session)
            df_analysis['returns'].rolling_std(120, min_periods=60).alias('intraday_vol')
        ])
        
        # Price-based volume proxy (better than just returns)
        df_analysis = df_analysis.with_columns([
            # High-low spread proxy (when available, use price range)
            (df_analysis['price'].rolling_max(2) - df_analysis['price'].rolling_min(2)).alias('hl_spread'),
            # Price velocity (rate of price change)
            df_analysis['price'].diff().abs().alias('price_velocity'),
            # Volatility-weighted volume proxy
            (df_analysis['returns'].abs() * df_analysis['intraday_vol']).alias('vol_weighted_proxy')
        ])
        
        # Improved Amihud illiquidity measure
        # Use price velocity instead of returns for denominator to avoid division by itself
        df_analysis = df_analysis.with_columns([
            (df_analysis['returns'].abs() / (df_analysis['price_velocity'] + 1e-10)).alias('amihud_illiquidity'),
            # Alternative: use rolling average of price changes
            (df_analysis['returns'].abs() / (df_analysis['price_velocity'].rolling_mean(10, min_periods=1) + 1e-10)).alias('amihud_smooth')
        ])
        
        # Roll's bid-ask spread estimator (improved)
        returns_clean = df_analysis['returns'].drop_nulls()
        if len(returns_clean) > 10:
            returns_np = returns_clean.to_numpy()
            
            # Calculate serial covariance more robustly
            returns_current = returns_np[1:]
            returns_lagged = returns_np[:-1]
            
            if len(returns_current) > 10:
                # Use numpy for covariance calculation
                cov_matrix = np.cov(returns_current, returns_lagged)
                serial_cov = cov_matrix[0, 1] if cov_matrix.shape == (2, 2) else 0
                
                # Roll's estimator: spread = 2 * sqrt(-covariance) if covariance < 0
                if serial_cov < 0:
                    spread_estimate = 2 * np.sqrt(-serial_cov)
                    spread_confidence = "High" if abs(serial_cov) > 1e-6 else "Low"
                else:
                    spread_estimate = 0
                    spread_confidence = "Low"
            else:
                spread_estimate = 0
                spread_confidence = "Insufficient Data"
        else:
            spread_estimate = 0
            spread_confidence = "Insufficient Data"
        
        # Kyle's lambda (price impact coefficient) - improved
        df_analysis = df_analysis.with_columns([
            # Signed volume using price momentum direction
            (df_analysis['returns'].sign() * df_analysis['vol_weighted_proxy']).alias('signed_volume'),
            # Trade direction indicator
            pl.when(df_analysis['returns'] > 0).then(1)
              .when(df_analysis['returns'] < 0).then(-1)
              .otherwise(0).alias('trade_direction')
        ])
        
        # Calculate Kyle's lambda with outlier filtering
        price_changes = df_analysis['returns'].drop_nulls().to_numpy()
        signed_volume = df_analysis['signed_volume'].drop_nulls().to_numpy()
        
        kyle_lambda = np.nan
        kyle_r_squared = 0
        
        if len(price_changes) > 20 and len(signed_volume) > 20:
            # Ensure same length
            min_len = min(len(price_changes), len(signed_volume))
            price_changes = price_changes[:min_len]
            signed_volume = signed_volume[:min_len]
            
            # Remove extreme outliers (top/bottom 2.5%)
            lower_p = np.percentile(price_changes, 2.5)
            upper_p = np.percentile(price_changes, 97.5)
            mask = (price_changes >= lower_p) & (price_changes <= upper_p)
            mask = mask & (np.abs(signed_volume) > 1e-10)  # Remove zero volume
            
            if mask.sum() > 15:
                try:
                    kyle_lambda, intercept, r_value, p_value, std_err = stats.linregress(
                        signed_volume[mask], price_changes[mask]
                    )
                    kyle_r_squared = r_value ** 2
                except:
                    kyle_lambda = np.nan
                    kyle_r_squared = 0
        
        # Market efficiency indicators
        df_analysis = df_analysis.with_columns([
            # Price efficiency: how directly price moves (less noise = more efficient)
            (df_analysis['price'].diff(60).abs() / 
             df_analysis['price'].diff().abs().rolling_sum(60, min_periods=30)).alias('price_efficiency_1h')
        ])
        
        # Calculate simplified autocorrelation measures using Polars operations
        # Since rolling_corr doesn't exist, use simpler proxy measures
        
        # Return autocorrelation proxy: rolling covariance normalized by variance
        df_analysis = df_analysis.with_columns([
            # Lagged returns
            df_analysis['returns'].shift(1).alias('returns_lag1'),
            df_analysis['returns'].abs().shift(1).alias('abs_returns_lag1')
        ])
        
        # Calculate rolling means for correlation approximation
        window_size = 120
        df_analysis = df_analysis.with_columns([
            # Rolling means
            df_analysis['returns'].rolling_mean(window_size, min_periods=30).alias('returns_ma'),
            df_analysis['returns_lag1'].rolling_mean(window_size, min_periods=30).alias('returns_lag1_ma'),
            df_analysis['returns'].abs().rolling_mean(window_size, min_periods=30).alias('abs_returns_ma'),
            df_analysis['abs_returns_lag1'].rolling_mean(window_size, min_periods=30).alias('abs_returns_lag1_ma')
        ])
        
        # Calculate correlation approximation using covariance formula
        df_analysis = df_analysis.with_columns([
            # Covariance approximation for returns
            ((df_analysis['returns'] - df_analysis['returns_ma']) * 
             (df_analysis['returns_lag1'] - df_analysis['returns_lag1_ma'])).rolling_mean(window_size, min_periods=30).alias('returns_cov'),
            # Variance for returns
            ((df_analysis['returns'] - df_analysis['returns_ma']) ** 2).rolling_mean(window_size, min_periods=30).alias('returns_var'),
            ((df_analysis['returns_lag1'] - df_analysis['returns_lag1_ma']) ** 2).rolling_mean(window_size, min_periods=30).alias('returns_lag1_var'),
            
            # Covariance approximation for volatility clustering
            ((df_analysis['returns'].abs() - df_analysis['abs_returns_ma']) * 
             (df_analysis['abs_returns_lag1'] - df_analysis['abs_returns_lag1_ma'])).rolling_mean(window_size, min_periods=30).alias('vol_cov'),
            # Variance for absolute returns
            ((df_analysis['returns'].abs() - df_analysis['abs_returns_ma']) ** 2).rolling_mean(window_size, min_periods=30).alias('vol_var'),
            ((df_analysis['abs_returns_lag1'] - df_analysis['abs_returns_lag1_ma']) ** 2).rolling_mean(window_size, min_periods=30).alias('vol_lag1_var')
        ])
        
        # Calculate final correlation estimates
        df_analysis = df_analysis.with_columns([
            # Return autocorrelation = cov(x,y) / sqrt(var(x) * var(y))
            (df_analysis['returns_cov'] / 
             (df_analysis['returns_var'] * df_analysis['returns_lag1_var']).sqrt()).alias('return_autocorr'),
            # Volatility clustering
            (df_analysis['vol_cov'] / 
             (df_analysis['vol_var'] * df_analysis['vol_lag1_var']).sqrt()).alias('vol_clustering')
        ])
        
        # Calculate summary statistics
        try:
            # Core metrics
            avg_realized_vol_1h = df_analysis['realized_vol_1h'].mean()
            avg_realized_vol_4h = df_analysis['realized_vol_4h'].mean()
            vol_of_vol = df_analysis['realized_vol_1h'].std()
            
            # Liquidity metrics
            avg_amihud = df_analysis['amihud_illiquidity'].mean()
            avg_amihud_smooth = df_analysis['amihud_smooth'].mean()
            
            # Market quality metrics
            avg_price_efficiency = df_analysis['price_efficiency_1h'].mean()
            avg_return_autocorr = df_analysis['return_autocorr'].mean()
            avg_vol_clustering = df_analysis['vol_clustering'].mean()
            
            # Market impact metrics
            avg_price_velocity = df_analysis['price_velocity'].mean()
            price_velocity_vol = df_analysis['price_velocity'].std()
            
        except Exception as e:
            # Fallback values if calculations fail
            avg_realized_vol_1h = np.nan
            avg_realized_vol_4h = np.nan
            vol_of_vol = np.nan
            avg_amihud = np.nan
            avg_amihud_smooth = np.nan
            avg_price_efficiency = np.nan
            avg_return_autocorr = np.nan
            avg_vol_clustering = np.nan
            avg_price_velocity = np.nan
            price_velocity_vol = np.nan
        
        return {
            # Volatility metrics
            'avg_realized_volatility_1h': avg_realized_vol_1h,
            'avg_realized_volatility_4h': avg_realized_vol_4h,
            'volatility_of_volatility': vol_of_vol,
            
            # Liquidity/Impact metrics
            'bid_ask_spread_estimate': spread_estimate,
            'spread_confidence': spread_confidence,
            'kyle_lambda': kyle_lambda,
            'kyle_r_squared': kyle_r_squared,
            'avg_amihud_illiquidity': avg_amihud,
            'avg_amihud_smooth': avg_amihud_smooth,
            
            # Market quality metrics
            'avg_price_efficiency': avg_price_efficiency,
            'avg_return_autocorr': avg_return_autocorr,
            'volatility_clustering': avg_vol_clustering,
            
            # Price dynamics
            'avg_price_velocity': avg_price_velocity,
            'price_velocity_volatility': price_velocity_vol,
            
            # Data for visualization
            'time_series_data': df_analysis.select([
                'datetime', 'price', 'returns', 'realized_vol_1h', 'realized_vol_4h',
                'price_efficiency_1h', 'return_autocorr', 'vol_clustering',
                'amihud_illiquidity', 'price_velocity'
            ])
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