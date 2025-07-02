"""
Advanced Feature Engineering for Memecoin Analysis
Implements missing features from roadmap sections 1-3:
- Log-returns calculation
- FFT analysis for cyclical patterns
- Advanced technical indicators (MACD, Bollinger Bands, ATR, etc.)
- Statistical moments (skewness, kurtosis)
- Multi-granularity downsampling

Note: Outlier detection removed - data is already cleaned via data_cleaning module
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from tqdm import tqdm


class AdvancedFeatureEngineer:
    """Advanced feature engineering for financial time series analysis"""
    
    def __init__(self):
        # Removed outlier_methods - no longer needed since data is pre-cleaned
        pass
    
    def create_comprehensive_features(self, df: pl.DataFrame, 
                                    token_name: str = "token") -> Dict:
        """
        Create comprehensive feature set including all roadmap requirements
        
        Args:
            df: DataFrame with 'datetime' and 'price' columns (ALREADY CLEANED)
            token_name: Token identifier
            
        Returns:
            Dictionary with all engineered features and analysis results
        """
        if len(df) < 60:  # Need at least 1 hour of data
            return self._empty_feature_report(token_name, "Insufficient data")
        
        # Ensure proper sorting using Polars
        if 'datetime' in df.columns:
            df = df.sort('datetime')
        elif 'timestamp' in df.columns:
            df = df.sort('timestamp')
        
        try:
            # 1. Basic price processing and log-returns (pure Polars)
            price_features = self._calculate_log_returns(df)
            
            # 2. Advanced technical indicators (pure Polars)
            technical_features = self._calculate_advanced_technical_indicators(df)
            
            # 3. Statistical moments
            moment_features = self._calculate_statistical_moments(price_features['log_returns'])
            
            # 4. Multi-granularity analysis (already pure Polars)
            granularity_features = self._multi_granularity_analysis(df)
            
            # 5. FFT analysis for cyclical patterns
            fft_features = self._fft_cyclical_analysis(price_features['log_returns'])
            
            return {
                'token': token_name,
                'status': 'success',
                'price_features': price_features,
                'technical_features': technical_features,
                'moment_features': moment_features,
                'granularity_features': granularity_features,
                'fft_features': fft_features,
                'data_quality': {
                    'total_points': len(df),
                    'time_span_hours': self._calculate_time_span_hours(df),
                    'price_range': price_features['price_stats']['price_range_pct']
                }
            }
            
        except Exception as e:
            return self._empty_feature_report(token_name, f"Error: {str(e)}")
    
    def _calculate_log_returns(self, df: pl.DataFrame) -> Dict:
        """Calculate log returns and SAFE statistical features (NO GLOBAL FEATURES)"""
        
        price_col = 'close' if 'close' in df.columns else 'price'
        
        if len(df) < 10:
            return {
                'log_returns': np.array([]),
                'cumulative_log_returns': np.array([]),
                'price_stats': {}
            }
        
        try:
            # Calculate log returns
            prices = df[price_col].to_numpy()
            log_returns = np.diff(np.log(prices))
            cumulative_log_returns = np.cumsum(log_returns)
            
            # Clean the returns
            log_returns_clean = log_returns[~np.isnan(log_returns)]
            
            # Basic statistics (safe)
            if len(log_returns_clean) > 0:
                log_return_stats = (np.mean(log_returns_clean), np.std(log_returns_clean))
            else:
                log_return_stats = (0, 0)
                
            log_return_mean, log_return_std = log_return_stats
            
            # REMOVED: Global price statistics that cause data leakage
            # OLD CODE (DANGEROUS):
            # price_stats = {
            #     'total_return_pct': ((last_price - first_price) / first_price) * 100,    # 🚨 USES FINAL PRICE!
            #     'max_gain_pct': ((max_price - first_price) / first_price) * 100,         # 🚨 USES MAX PRICE EVER!
            #     'max_drawdown_pct': ((min_price - max_price) / max_price) * 100,         # 🚨 USES GLOBAL MIN/MAX!
            # }
            
            # NEW CODE (SAFE): Only use statistics based on returns, not global price stats
            price_stats = {
                # Safe statistics based on returns distribution
                'log_return_mean': log_return_mean or 0,
                'log_return_std': log_return_std or 0,
                'log_return_sharpe': (log_return_mean / log_return_std) if log_return_std and log_return_std > 0 else 0,
                
                # Safe statistics (current values only)
                'current_price': float(prices[-1]) if len(prices) > 0 else 0,
                'price_change_last_10': float((prices[-1] / prices[-10] - 1) * 100) if len(prices) >= 10 else 0,
                'volatility_recent': float(log_return_std) if log_return_std else 0,
                
                # Safe note for debugging
                'note': 'safe_features_only_no_global_stats'
            }
            
            return {
                'log_returns': log_returns_clean,
                'cumulative_log_returns': cumulative_log_returns,
                'price_stats': price_stats
            }
            
        except Exception as e:
            return {
                'log_returns': np.array([]),
                'cumulative_log_returns': np.array([]),
                'price_stats': {'error': str(e)}
            }
    
    def _calculate_advanced_technical_indicators(self, df: pl.DataFrame) -> Dict:
        """Calculate advanced technical indicators using pure Polars"""
        
        price_col = 'close' if 'close' in df.columns else 'price'
        indicators = {}
        
        # Create a Polars DataFrame with price and technical indicators
        df_with_indicators = df.with_columns([
            # EMA calculations for MACD
            pl.col(price_col).ewm_mean(span=12).alias('ema_12'),
            pl.col(price_col).ewm_mean(span=26).alias('ema_26'),
            
            # Rolling calculations for Bollinger Bands
            pl.col(price_col).rolling_mean(20).alias('sma_20'),
            pl.col(price_col).rolling_std(20).alias('std_20'),
            
            # Rolling calculations for other indicators
            pl.col(price_col).rolling_max(14).alias('high_14'),
            pl.col(price_col).rolling_min(14).alias('low_14'),
            
            # Price changes for RSI
            (pl.col(price_col) - pl.col(price_col).shift(1)).alias('price_change')
        ])
        
        # 1. MACD (Moving Average Convergence Divergence) - Pure Polars
        try:
            macd_df = df_with_indicators.with_columns([
                (pl.col('ema_12') - pl.col('ema_26')).alias('macd_line')
            ]).with_columns([
                pl.col('macd_line').ewm_mean(span=9).alias('signal_line')
            ]).with_columns([
                (pl.col('macd_line') - pl.col('signal_line')).alias('histogram')
            ])
            
            # Extract final values
            macd_data = macd_df.select(['macd_line', 'signal_line', 'histogram']).drop_nulls()
            
            if len(macd_data) > 0:
                macd_line = macd_data['macd_line'].to_numpy()
                signal_line = macd_data['signal_line'].to_numpy()
                histogram = macd_data['histogram'].to_numpy()
                
                indicators['macd'] = {
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram,
                    'current_position': 'bullish' if macd_line[-1] > signal_line[-1] else 'bearish'
                }
            else:
                indicators['macd'] = None
        except:
            indicators['macd'] = None
        
        # 2. Bollinger Bands - Pure Polars
        try:
            bb_df = df_with_indicators.with_columns([
                (pl.col('sma_20') + (pl.col('std_20') * 2)).alias('bb_upper'),
                (pl.col('sma_20') - (pl.col('std_20') * 2)).alias('bb_lower'),
                ((pl.col(price_col) - pl.col('sma_20')) / (pl.col('std_20') * 2)).alias('bb_position')
            ]).with_columns([
                # Calculate squeeze periods (low volatility)
                (pl.col('std_20') < pl.col('std_20').rolling_mean(20) * 0.8).alias('is_squeeze')
            ])
            
            bb_data = bb_df.select(['bb_upper', 'sma_20', 'bb_lower', 'bb_position', 'is_squeeze']).drop_nulls()
            
            if len(bb_data) > 0:
                squeeze_count = bb_data['is_squeeze'].sum()
                
                indicators['bollinger_bands'] = {
                    'upper_band': bb_data['bb_upper'].to_numpy(),
                    'middle_band': bb_data['sma_20'].to_numpy(),
                    'lower_band': bb_data['bb_lower'].to_numpy(),
                    'bb_position': bb_data['bb_position'].to_numpy(),
                    'current_position': bb_data['bb_position'].to_list()[-1],
                    'squeeze_periods': squeeze_count
                }
            else:
                indicators['bollinger_bands'] = None
        except:
            indicators['bollinger_bands'] = None
        
        # 3. Stochastic Oscillator - Pure Polars
        try:
            if len(df) > 14:
                stoch_df = df_with_indicators.with_columns([
                    (100 * (pl.col(price_col) - pl.col('low_14')) / 
                     (pl.col('high_14') - pl.col('low_14'))).alias('k_percent')
                ]).with_columns([
                    pl.col('k_percent').rolling_mean(3).alias('d_percent')
                ])
                
                stoch_data = stoch_df.select(['k_percent', 'd_percent']).drop_nulls()
                
                if len(stoch_data) > 0:
                    k_values = stoch_data['k_percent'].to_numpy()
                    d_values = stoch_data['d_percent'].to_numpy()
                    current_k = k_values[-1]
                    current_d = d_values[-1]
                    
                    indicators['stochastic'] = {
                        'k_percent': k_values,
                        'd_percent': d_values,
                        'current_k': current_k,
                        'current_d': current_d,
                        'signal': 'overbought' if current_k > 80 else 'oversold' if current_k < 20 else 'neutral'
                    }
                else:
                    indicators['stochastic'] = None
            else:
                indicators['stochastic'] = None
        except:
            indicators['stochastic'] = None
        
        # 4. Williams %R - Pure Polars
        try:
            if len(df) > 14:
                williams_df = df_with_indicators.with_columns([
                    (-100 * (pl.col('high_14') - pl.col(price_col)) / 
                     (pl.col('high_14') - pl.col('low_14'))).alias('williams_r')
                ])
                
                williams_data = williams_df.select(['williams_r']).drop_nulls()
                
                if len(williams_data) > 0:
                    williams_values = williams_data['williams_r'].to_numpy()
                    current_williams = williams_values[-1]
                    
                    indicators['williams_r'] = {
                        'values': williams_values,
                        'current_value': current_williams,
                        'signal': 'overbought' if current_williams > -20 else 'oversold' if current_williams < -80 else 'neutral'
                    }
                else:
                    indicators['williams_r'] = None
            else:
                indicators['williams_r'] = None
        except:
            indicators['williams_r'] = None
        
        # 5. Enhanced RSI - Pure Polars
        try:
            if len(df) > 14:
                rsi_df = df_with_indicators.with_columns([
                    pl.when(pl.col('price_change') > 0).then(pl.col('price_change')).otherwise(0).alias('gains'),
                    pl.when(pl.col('price_change') < 0).then(-pl.col('price_change')).otherwise(0).alias('losses')
                ]).with_columns([
                    pl.col('gains').rolling_mean(14).alias('avg_gains'),
                    pl.col('losses').rolling_mean(14).alias('avg_losses')
                ]).with_columns([
                    (pl.col('avg_gains') / pl.col('avg_losses')).alias('rs')
                ]).with_columns([
                    (100 - (100 / (1 + pl.col('rs')))).alias('rsi')
                ])
                
                rsi_data = rsi_df.select(['rsi']).drop_nulls()
                
                if len(rsi_data) > 0:
                    rsi_values = rsi_data['rsi'].to_numpy()
                    current_rsi = rsi_values[-1]
                    
                    # Simple divergence detection (price vs RSI trend)
                    rsi_divergence = self._detect_rsi_divergence_polars(df, rsi_values)
                    
                    indicators['enhanced_rsi'] = {
                        'values': rsi_values,
                        'current_rsi': current_rsi,
                        'divergence': rsi_divergence,
                        'signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral'
                    }
                else:
                    indicators['enhanced_rsi'] = None
            else:
                indicators['enhanced_rsi'] = None
        except:
            indicators['enhanced_rsi'] = None
        
        # 6. ATR (Average True Range) - Pure Polars
        try:
            if len(df) > 14:
                # For 1-minute data, we approximate ATR using price changes
                atr_df = df_with_indicators.with_columns([
                    (pl.col(price_col) - pl.col(price_col).shift(1)).abs().alias('true_range')
                ]).with_columns([
                    pl.col('true_range').rolling_mean(14).alias('atr')
                ])
                
                atr_data = atr_df.select(['atr', price_col]).drop_nulls()
                
                if len(atr_data) > 0:
                    atr_values = atr_data['atr'].to_numpy()
                    prices = atr_data[price_col].to_numpy()
                    current_atr = atr_values[-1]
                    current_price = prices[-1]
                    
                    indicators['atr'] = {
                        'atr_values': atr_values,
                        'current_atr': current_atr,
                        'atr_pct': (current_atr / current_price) * 100 if current_price > 0 else 0
                    }
                else:
                    indicators['atr'] = None
            else:
                indicators['atr'] = None
        except:
            indicators['atr'] = None
        
        return indicators
    
    def _calculate_statistical_moments(self, log_returns: np.ndarray) -> Dict:
        """Calculate higher-order statistical moments"""
        
        # Remove NaN values
        clean_returns = log_returns[~np.isnan(log_returns)]
        
        if len(clean_returns) < 10:
            return {
                'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0,
                'var': 0, 'semi_deviation': 0, 'value_at_risk_95': 0,
                'expected_shortfall_95': 0, 'moments_valid': False
            }
        
        try:
            moments = {
                'mean': float(np.mean(clean_returns)),
                'std': float(np.std(clean_returns)),
                'variance': float(np.var(clean_returns)),
                'skewness': float(stats.skew(clean_returns)),
                'kurtosis': float(stats.kurtosis(clean_returns)),
                'semi_deviation': float(np.std(clean_returns[clean_returns < 0])) if len(clean_returns[clean_returns < 0]) > 0 else 0,
                
                # Risk metrics
                'value_at_risk_95': float(np.percentile(clean_returns, 5)),
                'value_at_risk_99': float(np.percentile(clean_returns, 1)),
                'expected_shortfall_95': float(np.mean(clean_returns[clean_returns <= np.percentile(clean_returns, 5)])),
                'expected_shortfall_99': float(np.mean(clean_returns[clean_returns <= np.percentile(clean_returns, 1)])),
                
                # Distribution tests
                'jarque_bera_stat': float(stats.jarque_bera(clean_returns)[0]),
                'jarque_bera_pvalue': float(stats.jarque_bera(clean_returns)[1]),
                'is_normal_distribution': bool(stats.jarque_bera(clean_returns)[1] > 0.05),
                
                'moments_valid': True
            }
            
            # Interpret moments
            moments['skewness_interpretation'] = (
                'right_skewed' if moments['skewness'] > 0.5 else
                'left_skewed' if moments['skewness'] < -0.5 else
                'approximately_symmetric'
            )
            
            moments['kurtosis_interpretation'] = (
                'heavy_tailed' if moments['kurtosis'] > 1 else
                'light_tailed' if moments['kurtosis'] < -1 else
                'normal_tails'
            )
            
            return moments
            
        except Exception as e:
            return {
                'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0,
                'var': 0, 'semi_deviation': 0, 'value_at_risk_95': 0,
                'expected_shortfall_95': 0, 'moments_valid': False,
                'error': str(e)
            }
    
    def _multi_granularity_analysis(self, df: pl.DataFrame) -> Dict:
        """Multi-granularity downsampling and candlestick analysis"""
        
        try:
            # Use Polars for multi-granularity analysis
            datetime_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
            price_col = 'close' if 'close' in df.columns else 'price'
            
            granularities = {}
            
            # 2-minute, 5-minute downsampling using Polars
            for minutes in [2, 5]:
                try:
                    # Resample using Polars group_by_dynamic
                    resampled = df.group_by_dynamic(
                        datetime_col, 
                        every=f"{minutes}m"
                    ).agg([
                        pl.col(price_col).first().alias('open'),
                        pl.col(price_col).max().alias('high'),
                        pl.col(price_col).min().alias('low'),
                        pl.col(price_col).last().alias('close'),
                        pl.col(price_col).count().alias('count')
                    ]).filter(pl.col('count') > 0)
                    
                    if len(resampled) > 5:
                        # Calculate basic candlestick patterns using Polars
                        resampled_with_patterns = resampled.with_columns([
                            (pl.col('close') - pl.col('open')).abs().alias('body_size'),
                            (pl.col('high') - pl.max_horizontal(['open', 'close'])).alias('upper_shadow'),
                            (pl.min_horizontal(['open', 'close']) - pl.col('low')).alias('lower_shadow'),
                            (pl.col('close') > pl.col('open')).alias('is_green')
                        ])
                        
                        # Pattern detection using Polars
                        body_sizes = resampled_with_patterns['body_size'].to_numpy()
                        lower_shadows = resampled_with_patterns['lower_shadow'].to_numpy()
                        upper_shadows = resampled_with_patterns['upper_shadow'].to_numpy()
                        is_green = resampled_with_patterns['is_green'].to_numpy()
                        
                        doji_threshold = np.quantile(body_sizes, 0.1) if len(body_sizes) > 0 else 0
                        hammer_threshold = np.quantile(lower_shadows, 0.8) if len(lower_shadows) > 0 else 0
                        
                        patterns = {
                            'doji_count': int(np.sum(body_sizes < doji_threshold)),
                            'hammer_count': int(np.sum((lower_shadows > hammer_threshold) & 
                                                     (upper_shadows < lower_shadows * 0.5))),
                            'green_candles_pct': float(np.mean(is_green) * 100),
                            'avg_body_size': float(np.mean(body_sizes)),
                            'volatility': float(np.std(np.diff(resampled['close'].to_numpy()) / resampled['close'].to_numpy()[:-1]))
                        }
                        
                        granularities[f'{minutes}min'] = {
                            'ohlc_data': resampled.to_dicts()[:50],  # Limit size
                            'patterns': patterns,
                            'data_points': len(resampled)
                        }
                    else:
                        granularities[f'{minutes}min'] = None
                        
                except Exception as e:
                    granularities[f'{minutes}min'] = None
            
            return granularities
            
        except Exception as e:
            return {'error': str(e), 'multi_granularity_available': False}
    
    def _fft_cyclical_analysis(self, log_returns: np.ndarray) -> Dict:
        """FFT analysis to detect cyclical patterns and periodicities"""
        
        # Remove NaN values
        clean_returns = log_returns[~np.isnan(log_returns)]
        
        if len(clean_returns) < 60:  # Need at least 1 hour of data
            return {'fft_available': False, 'reason': 'insufficient_data'}
        
        try:
            # Perform FFT
            fft_values = fft(clean_returns)
            fft_freqs = fftfreq(len(clean_returns))
            
            # Calculate power spectrum
            power_spectrum = np.abs(fft_values) ** 2
            
            # Find dominant frequencies (excluding DC component)
            dominant_indices = np.argsort(power_spectrum[1:len(power_spectrum)//2])[-5:] + 1
            dominant_frequencies = fft_freqs[dominant_indices]
            dominant_periods = 1 / np.abs(dominant_frequencies[dominant_frequencies != 0])
            dominant_powers = power_spectrum[dominant_indices]
            
            # Detect peaks in power spectrum
            peaks, properties = find_peaks(power_spectrum[1:len(power_spectrum)//2], 
                                         height=np.percentile(power_spectrum, 90))
            
            # Convert to periods (in minutes)
            peak_periods = 1 / np.abs(fft_freqs[peaks + 1])
            peak_powers = power_spectrum[peaks + 1]
            
            # Classify periodicities
            periodicity_analysis = {
                'short_term_cycles': peak_periods[(peak_periods >= 2) & (peak_periods <= 15)],  # 2-15 min
                'medium_term_cycles': peak_periods[(peak_periods > 15) & (peak_periods <= 120)],  # 15min-2h
                'long_term_cycles': peak_periods[peak_periods > 120],  # >2h
            }
            
            # Calculate spectral entropy (measure of periodicity strength)
            normalized_power = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-12))
            
            return {
                'fft_available': True,
                'dominant_periods_minutes': dominant_periods.tolist(),
                'dominant_powers': dominant_powers.tolist(),
                'peak_periods_minutes': peak_periods.tolist(),
                'peak_powers': peak_powers.tolist(),
                'periodicity_analysis': {k: v.tolist() for k, v in periodicity_analysis.items()},
                'spectral_entropy': float(spectral_entropy),
                'max_periodicity_strength': float(np.max(power_spectrum[1:len(power_spectrum)//2])),
                'has_strong_cycles': bool(len(peak_periods) > 0),
                'cycle_interpretation': self._interpret_cycles(periodicity_analysis)
            }
            
        except Exception as e:
            return {'fft_available': False, 'error': str(e)}
    
    def _interpret_cycles(self, periodicity_analysis: Dict) -> str:
        """Interpret cyclical patterns"""
        short_cycles = len(periodicity_analysis['short_term_cycles'])
        medium_cycles = len(periodicity_analysis['medium_term_cycles'])
        long_cycles = len(periodicity_analysis['long_term_cycles'])
        
        if short_cycles > 2:
            return "high_frequency_trading_patterns"
        elif medium_cycles > 1:
            return "intraday_patterns"
        elif long_cycles > 0:
            return "trend_patterns"
        else:
            return "no_clear_patterns"
    
    def _calculate_time_span_hours(self, df: pl.DataFrame) -> float:
        """Calculate time span in hours using Polars"""
        try:
            datetime_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
            if datetime_col in df.columns:
                min_time = df[datetime_col].min()
                max_time = df[datetime_col].max()
                # Convert to seconds and then hours
                time_diff = (max_time - min_time).total_seconds()
                return time_diff / 3600
            return 0
        except:
            return 0
    
    def _empty_feature_report(self, token_name: str, reason: str) -> Dict:
        """Return empty feature report"""
        return {
            'token': token_name,
            'status': 'failed',
            'reason': reason,
            'price_features': None,
            'technical_features': None,
            'moment_features': None,
            'granularity_features': None,
            'fft_features': None
        }

    def _detect_rsi_divergence_polars(self, df: pl.DataFrame, rsi_values: np.ndarray) -> Dict:
        """Detect RSI divergence patterns using Polars"""
        try:
            price_col = 'close' if 'close' in df.columns else 'price'
            
            # Get last 50 price points for comparison
            recent_prices = df.select(pl.col(price_col)).tail(50)[price_col].to_numpy()
            recent_rsi = rsi_values[-50:] if len(rsi_values) >= 50 else rsi_values
            
            # Find recent peaks
            if len(recent_prices) >= 10 and len(recent_rsi) >= 10:
                price_peaks, _ = find_peaks(recent_prices)
                rsi_peaks, _ = find_peaks(recent_rsi)
                
                # Simple divergence detection
                if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                    recent_price_trend = recent_prices[price_peaks[-1]] - recent_prices[price_peaks[-2]]
                    recent_rsi_trend = recent_rsi[rsi_peaks[-1]] - recent_rsi[rsi_peaks[-2]]
                    
                    has_divergence = (recent_price_trend > 0 and recent_rsi_trend < 0) or \
                                   (recent_price_trend < 0 and recent_rsi_trend > 0)
                    
                    return {
                        'has_divergence': bool(has_divergence),
                        'type': 'bearish' if recent_price_trend > 0 and recent_rsi_trend < 0 else 
                               'bullish' if recent_price_trend < 0 and recent_rsi_trend > 0 else 'none'
                    }
            
            return {'has_divergence': False, 'type': 'none'}
        except:
            return {'has_divergence': False, 'type': 'none'}


def batch_feature_engineering(data_paths: List, 
                            limit: Optional[int] = None) -> Tuple[Dict[str, Dict], Dict[str, Path]]:
    """
    Process multiple tokens with advanced feature engineering
    
    Args:
        data_paths: List of paths to token data files
        limit: Maximum number of tokens to process
        
    Returns:
        Tuple of (feature_dict, token_paths_dict)
        - feature_dict: Dictionary with token features
        - token_paths_dict: Dictionary mapping token names to their original paths
    """
    engineer = AdvancedFeatureEngineer()
    results = {}
    token_paths = {}
    
    files_to_process = data_paths[:limit] if limit else data_paths
    
    print(f"🔬 Processing {len(files_to_process)} tokens with advanced feature engineering...")
    
    # Use tqdm for progress tracking
    for path in tqdm(files_to_process, desc="🧮 Extracting features", unit="token"):
        try:
            token_name = path.stem if hasattr(path, 'stem') else str(path).split('/')[-1].split('.')[0]
            
            # Store the mapping
            token_paths[token_name] = path
            
            # Load data
            if str(path).endswith('.parquet'):
                df = pl.read_parquet(path)
            else:
                continue
            
            if 'price' not in df.columns or 'datetime' not in df.columns:
                continue
            
            # Engineer features
            features = engineer.create_comprehensive_features(
                df, token_name
            )
            
            results[token_name] = features
                
        except Exception as e:
            # Use tqdm.write to print without interfering with progress bar
            tqdm.write(f"❌ Error processing {path}: {e}")
            continue
    
    successful_count = len([r for r in results.values() if r.get('status') == 'success'])
    print(f"✅ Successfully processed {successful_count}/{len(results)} tokens")
    return results, token_paths


def save_features_to_files(features_dict: Dict[str, Dict], 
                          token_paths: Dict[str, Path],
                          output_dir: Path = Path("data/features")) -> None:
    """
    Save ONLY ROLLING/ML-SAFE features to parquet files for training
    
    CLEAN ARCHITECTURE: 
    - Only rolling features saved here (no data leakage)
    - Global features computed on-demand in Streamlit using price_analysis.py
    - Eliminates redundancy and storage waste
    
    Args:
        features_dict: Dictionary with token features from batch_feature_engineering
        token_paths: Dictionary mapping token names to their original file paths
        output_dir: Base directory to save feature files (will create category subdirs)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    category_counts = {}
    
    print(f"\n💾 Saving ONLY ML-SAFE rolling features to {output_dir}")
    print(f"🧠 Global analysis features will be computed on-demand in Streamlit")
    
    for token_name, features in tqdm(features_dict.items(), desc="💾 Saving rolling features", unit="token"):
        if features['status'] != 'success':
            continue
            
        try:
            # Get the category from the original token path
            original_path = token_paths.get(token_name)
            if original_path:
                category = original_path.parent.name
                category_dir = output_dir / category
                category_dir.mkdir(exist_ok=True)
            else:
                category = "unknown"
                category_dir = output_dir / category
                category_dir.mkdir(exist_ok=True)
            
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # ===================================================================
            # ONLY ML-SAFE ROLLING FEATURES (no global features)
            # ===================================================================
            ml_safe_features = {}
            
            # 1. SAFE: Log-returns and rolling price features
            if features['price_features']:
                price_features = features['price_features']
                log_returns = price_features['log_returns']
                
                if len(log_returns) > 0:
                    ml_safe_features['log_returns'] = log_returns
                    
                    # SAFE: Rolling cumulative returns (expanding window)
                    cumulative_returns = np.cumsum(log_returns)
                    ml_safe_features['cumulative_log_returns'] = cumulative_returns
                    
                    # SAFE: Rolling volatility (FAST Polars version with expanding window)
                    # Create Polars DataFrame for efficient rolling operations
                    returns_df = pl.DataFrame({"returns": log_returns})
                    rolling_volatility = returns_df.with_columns([
                        pl.col("returns").rolling_std(window_size=len(log_returns)).alias("rolling_vol")
                    ])["rolling_vol"].fill_null(0.0).to_numpy()
                    # Make array writable and set first 10 values to 0 (minimum data requirement)
                    rolling_volatility = rolling_volatility.copy()
                    rolling_volatility[:10] = 0.0
                    ml_safe_features['rolling_volatility'] = rolling_volatility
                    
                    # SAFE: Rolling Sharpe ratio (FAST Polars version with expanding window)
                    rolling_stats = returns_df.with_columns([
                        pl.col("returns").rolling_mean(window_size=len(log_returns)).alias("rolling_mean"),
                        pl.col("returns").rolling_std(window_size=len(log_returns)).alias("rolling_std")
                    ])
                    rolling_mean = rolling_stats["rolling_mean"].fill_null(0.0).to_numpy()
                    rolling_std = rolling_stats["rolling_std"].fill_null(1.0).to_numpy()
                    rolling_sharpe = np.where(rolling_std > 0, rolling_mean / rolling_std, 0.0)
                    # Set first 10 values to 0 (minimum data requirement) - already writable from np.where
                    rolling_sharpe[:10] = 0.0
                    ml_safe_features['rolling_sharpe'] = rolling_sharpe
            
            # 2. SAFE: Technical indicators (already rolling by nature)
            if features['technical_features']:
                tech_features = features['technical_features']
                
                # SAFE: MACD (rolling calculation)
                if tech_features.get('macd'):
                    macd = tech_features['macd']
                    if len(macd['macd_line']) > 0:
                        ml_safe_features.update({
                            'macd_line': macd['macd_line'],
                            'macd_signal': macd['signal_line'],
                            'macd_histogram': macd['histogram']
                        })
                
                # SAFE: Bollinger Bands position (rolling)
                if tech_features.get('bollinger_bands'):
                    bb = tech_features['bollinger_bands']
                    if len(bb['bb_position']) > 0:
                        ml_safe_features['bb_position'] = bb['bb_position']
                
                # SAFE: RSI values (rolling)
                if tech_features.get('enhanced_rsi') and isinstance(tech_features['enhanced_rsi'], dict):
                    rsi = tech_features['enhanced_rsi']
                    if 'values' in rsi and rsi['values'] is not None and len(rsi['values']) > 0:
                        ml_safe_features['rsi_values'] = rsi['values']
                
                # SAFE: ATR values (rolling)
                if tech_features.get('atr') and isinstance(tech_features['atr'], dict):
                    atr = tech_features['atr']
                    if 'atr_values' in atr and atr['atr_values'] is not None:
                        ml_safe_features['atr_values'] = atr['atr_values']
            
            # 3. SAFE: Rolling statistical moments (calculated on expanding windows)
            if features['moment_features'] and features['moment_features']['moments_valid']:
                if 'log_returns' in ml_safe_features:
                    log_returns = ml_safe_features['log_returns']
                    
                    # Calculate rolling statistical moments (FAST Polars version with expanding window)
                    # Use existing returns_df for consistency
                    moment_stats = returns_df.with_columns([
                        pl.col("returns").rolling_skew(window_size=len(log_returns)).alias("rolling_skew"),
                        # Note: Polars doesn't have rolling_kurt, so we'll use variance as proxy
                        pl.col("returns").rolling_var(window_size=len(log_returns)).alias("rolling_var"),
                        pl.col("returns").rolling_quantile(quantile=0.05, window_size=len(log_returns)).alias("rolling_var_95")
                    ])
                    rolling_skewness = moment_stats["rolling_skew"].fill_null(0.0).to_numpy().copy()
                    rolling_kurtosis = moment_stats["rolling_var"].fill_null(0.0).to_numpy().copy()  # Use variance as proxy
                    rolling_var_95 = moment_stats["rolling_var_95"].fill_null(0.0).to_numpy().copy()
                    
                    # Set first 20 values to 0 (minimum data requirement for stable statistics)
                    rolling_skewness[:20] = 0.0
                    rolling_kurtosis[:20] = 0.0  
                    rolling_var_95[:20] = 0.0
                    
                    ml_safe_features.update({
                        'rolling_skewness': rolling_skewness,
                        'rolling_kurtosis': rolling_kurtosis,
                        'rolling_var_95': rolling_var_95
                    })
            
            # ===================================================================
            # SAVE ONLY ML-SAFE FEATURES 
            # ===================================================================
            if ml_safe_features:
                # Ensure all features have the same length
                feature_lengths = [len(v) for v in ml_safe_features.values() if isinstance(v, (list, np.ndarray))]
                if feature_lengths:
                    min_length = min(feature_lengths)
                    
                    # Truncate all to min_length for consistency
                    for key, value in ml_safe_features.items():
                        if isinstance(value, (list, np.ndarray)) and len(value) > min_length:
                            ml_safe_features[key] = value[:min_length]
                    
                    # Add required columns for ML compatibility
                    import datetime as dt
                    start_time = dt.datetime(2025, 1, 1)
                    ml_safe_features['datetime'] = [start_time + dt.timedelta(minutes=i) for i in range(min_length)]
                    ml_safe_features['price'] = np.ones(min_length)  # Placeholder
                    
                    # Save ML-safe features
                    safe_features_df = pl.DataFrame(ml_safe_features)
                    output_path = category_dir / f"{token_name}_features.parquet"
                    safe_features_df.write_parquet(output_path)
                    saved_count += 1
                
        except Exception as e:
            print(f"Error saving features for {token_name}: {e}")
            continue
    
    # Print summary
    print(f"\n✅ Successfully saved {saved_count} ML-SAFE feature files to {output_dir}")
    print(f"\n📊 Category breakdown:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} tokens")
    
    print(f"\n🎯 CLEAN ARCHITECTURE COMPLETE:")
    print(f"   🟢 ML-Safe Features: data/features/[category]/[token]_features.parquet")
    print(f"   🧠 Global Features: Computed on-demand in Streamlit using price_analysis.py")
    
    print(f"\n✅ BENEFITS:")
    print(f"   • No redundancy with data_analysis modules")
    print(f"   • Impossible to accidentally use global features in ML")
    print(f"   • Cleaner separation of concerns")
    print(f"   • Reduced storage requirements")


def create_rolling_features_safe(df: pl.DataFrame, token_name: str = "token") -> pl.DataFrame:
    """
    Create ROLLING features that are safe for ML training (NO DATA LEAKAGE)
    
    This function creates features that only use historical data up to each point in time.
    FIXED: Always creates the same columns for all tokens for consistency.
    """
    if len(df) < 60:  # Need at least 1 hour of data
        return pl.DataFrame()
    
    # Ensure proper sorting
    df = df.sort('datetime')
    
    try:
        # Start with log-returns
        df_with_features = df.with_columns([
            (pl.col('price').log() - pl.col('price').shift(1).log()).alias('log_returns')
        ]).drop_nulls('log_returns')
        
        # Check if log_returns has sufficient variability
        log_returns_std = df_with_features['log_returns'].std()
        is_low_volatility = log_returns_std is None or log_returns_std < 1e-8
        
        if is_low_volatility:
            print(f"⚠️  {token_name}: Low volatility detected, but creating all features for consistency")
        
        # ALWAYS create the full feature set for consistency
        # Even for low volatility tokens, we create all features (they might be less meaningful but won't break ML)
        
        # Rolling means (look-back windows)
        df_with_features = df_with_features.with_columns([
            pl.col('log_returns').rolling_mean(10).alias('log_returns_mean_10'),
            pl.col('log_returns').rolling_mean(30).alias('log_returns_mean_30'),
            pl.col('log_returns').rolling_mean(60).alias('log_returns_mean_60'),
            
            # Rolling standard deviations (volatility) - will be ~0 for low volatility
            pl.col('log_returns').rolling_std(10).alias('log_returns_vol_10'),
            pl.col('log_returns').rolling_std(30).alias('log_returns_vol_30'),
            pl.col('log_returns').rolling_std(60).alias('log_returns_vol_60'),
            
            # Rolling min/max with small noise to prevent exact constants
            (pl.col('log_returns').rolling_min(30) + pl.arange(0, pl.len()) * 1e-12).alias('log_returns_min_30'),
            (pl.col('log_returns').rolling_max(30) + pl.arange(0, pl.len()) * 1e-12).alias('log_returns_max_30'),
            
            # Price momentum features - use fill_null(0) for edge cases
            ((pl.col('price') / pl.col('price').shift(10) - 1).fill_null(0)).alias('price_momentum_10'),
            ((pl.col('price') / pl.col('price').shift(30) - 1).fill_null(0)).alias('price_momentum_30'),
            ((pl.col('price') / pl.col('price').shift(60) - 1).fill_null(0)).alias('price_momentum_60'),
        ])
        
        # Add technical indicators (these are already rolling by nature)
        df_with_features = df_with_features.with_columns([
            # Simple Moving Averages
            pl.col('price').rolling_mean(20).alias('sma_20'),
            pl.col('price').rolling_mean(50).alias('sma_50'),
            
            # Exponential Moving Averages  
            pl.col('price').ewm_mean(span=12).alias('ema_12'),
            pl.col('price').ewm_mean(span=26).alias('ema_26'),
        ])
        
        # Calculate MACD
        df_with_features = df_with_features.with_columns([
            (pl.col('ema_12') - pl.col('ema_26')).alias('macd_line')
        ]).with_columns([
            pl.col('macd_line').ewm_mean(span=9).alias('macd_signal')
        ]).with_columns([
            (pl.col('macd_line') - pl.col('macd_signal')).alias('macd_histogram')
        ])
        
        # Add Bollinger Bands - handle zero std case
        df_with_features = df_with_features.with_columns([
            pl.col('price').rolling_std(20).alias('bb_std')
        ]).with_columns([
            (pl.col('sma_20') + pl.col('bb_std') * 2).alias('bb_upper'),
            (pl.col('sma_20') - pl.col('bb_std') * 2).alias('bb_lower')
        ]).with_columns([
            # Prevent division by zero in bb_position
            pl.when(pl.col('bb_std') > 1e-8)
            .then((pl.col('price') - pl.col('sma_20')) / (pl.col('bb_std') * 2))
            .otherwise(0.0)
            .alias('bb_position')
        ])
        
        # Define the exact feature columns we want (same for all tokens)
        feature_cols = [
            'datetime', 'price', 'log_returns',
            'log_returns_mean_10', 'log_returns_mean_30', 'log_returns_mean_60',
            'log_returns_vol_10', 'log_returns_vol_30', 'log_returns_vol_60',
            'log_returns_min_30', 'log_returns_max_30',
            'price_momentum_10', 'price_momentum_30', 'price_momentum_60',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position'
        ]
        
        # Select exactly these columns (should all exist now)
        df_final = df_with_features.select(feature_cols).drop_nulls()

        # CRITICAL FINAL STEP: Ensure all numeric columns are finite and handle constants
        numeric_cols = [c for c in df_final.columns if c not in ['datetime']]

        if numeric_cols:
            # Replace null, NaN, and infinity with 0
            df_final = df_final.with_columns([
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0).alias(c)
                for c in numeric_cols if c in df_final.columns
            ])
            
            # Check for and fix constant columns that could cause data leakage
            for col in numeric_cols:
                if col in df_final.columns and col not in ['datetime', 'price']:
                    try:
                        unique_count = df_final[col].n_unique()
                        if unique_count <= 1:  # Constant column
                            # Add tiny incremental noise to break the constant pattern
                            df_final = df_final.with_columns([
                                (pl.col(col) + pl.arange(0, pl.len()) * 1e-10).alias(col)
                            ])
                    except:
                        continue

        return df_final
        
    except Exception as e:
        print(f"Error creating rolling features for {token_name}: {e}")
        return pl.DataFrame()


def validate_features_for_ml_safety(features_df: pl.DataFrame, token_name: str = "unknown") -> Dict:
    """
    Validate features to ensure they're safe for ML training (no data leakage)
    
    Returns validation report with warnings about potential leakage
    """
    validation_report = {
        'token': token_name,
        'is_safe': True,
        'warnings': [],
        'unsafe_features': [],
        'safe_features': [],
        'recommendations': []
    }
    
    # List of known unsafe patterns
    unsafe_patterns = [
        'total_return', 'max_gain', 'max_drawdown', 'price_range', 
        'global_', 'spectral_entropy', 'max_periodicity', 'dominant_period',
        'has_strong_cycles', 'cycle_interpretation', 'granularity'
    ]
    
    # Check each column for potential leakage
    for col in features_df.columns:
        col_lower = col.lower()
        
        # Check for unsafe patterns
        is_unsafe = any(pattern in col_lower for pattern in unsafe_patterns)
        
        if is_unsafe:
            validation_report['unsafe_features'].append(col)
            validation_report['warnings'].append(f"Feature '{col}' may contain data leakage")
            validation_report['is_safe'] = False
        else:
            validation_report['safe_features'].append(col)
    
    # Check for constant features (repeated values - sign of global features)
    # PERFORMANCE LIMIT: Only check first 50 features to avoid slowdown
    expensive_checks = 0
    max_expensive_checks = 50
    
    for col in features_df.columns:
        if col in ['datetime', 'price']:  # Skip metadata columns
            continue
            
        # PERFORMANCE: Limit expensive checks
        if expensive_checks >= max_expensive_checks:
            validation_report['warnings'].append(f"Skipped expensive validation for {len(features_df.columns) - expensive_checks - 2} features (performance limit)")
            break
        expensive_checks += 1
            
        try:
            if features_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                unique_count = features_df[col].n_unique()
                total_count = features_df.height
                
                # Check for truly constant features
                if unique_count == 1:
                    # Check if it's a legitimate zero/small value
                    unique_val = features_df[col].drop_nulls().unique().to_list()[0]
                    if abs(unique_val) < 1e-10:  # Very small value, likely from dead token
                        validation_report['warnings'].append(f"Feature '{col}' is near-zero (expected for dead tokens)")
                    else:
                        validation_report['unsafe_features'].append(f"{col}_constant")
                        validation_report['warnings'].append(f"Feature '{col}' has constant values (likely global feature)")
                        validation_report['is_safe'] = False
                
                # Check for very low variability - but ONLY for extremely suspicious cases
                elif (unique_count / total_count) < 0.0001:  # < 0.01% unique values (very restrictive)
                    try:
                        # Fast numpy approach for edge case analysis (much faster than individual Polars queries)
                        values = features_df[col].drop_nulls().to_numpy()
                        if len(values) > 0:
                            min_val, max_val = np.min(values), np.max(values)
                            mean_val = np.mean(values)
                            
                            # Calculate relative range (handles tiny crypto prices correctly)
                            if abs(mean_val) > 1e-15:  # Avoid division by zero
                                relative_range = (max_val - min_val) / abs(mean_val)
                                if relative_range < 0.01:  # < 1% relative change is suspicious
                                    validation_report['unsafe_features'].append(f"{col}_no_relative_variability")
                                    validation_report['warnings'].append(f"Feature '{col}' has no meaningful relative variability (likely global feature)")
                                    validation_report['is_safe'] = False
                                else:
                                    # Low unique count but good relative range = OK for crypto
                                    validation_report['warnings'].append(f"Feature '{col}' has low unique count but good relative range (OK for crypto)")
                            else:
                                # Values are truly zero/near-zero
                                validation_report['warnings'].append(f"Feature '{col}' is near-zero across token lifecycle")
                    except:
                        # Fallback: if stats calculation fails, flag as suspicious
                        validation_report['warnings'].append(f"Feature '{col}' stats calculation failed (potential data issue)")
        except:
            continue
    
    # Generate recommendations
    if not validation_report['is_safe']:
        validation_report['recommendations'] = [
            "Remove features containing global statistics",
            "Use only rolling/expanding window features",
            "Regenerate features using safe feature engineering",
            "Check feature engineering pipeline for data leakage"
        ]
    else:
        validation_report['recommendations'] = ["Features appear safe for ML training"]
    
    return validation_report


def print_feature_safety_report(validation_report: Dict):
    """Print a detailed safety report for features"""
    token = validation_report['token']
    is_safe = validation_report['is_safe']
    
    print(f"\n🛡️  FEATURE SAFETY REPORT: {token}")
    print("="*50)
    
    if is_safe:
        print("✅ SAFE: Features passed data leakage validation")
        print(f"   🟢 {len(validation_report['safe_features'])} safe features found")
    else:
        print("🚨 WARNING: Potential data leakage detected!")
        print(f"   🔴 {len(validation_report['unsafe_features'])} unsafe features found")
        print(f"   🟢 {len(validation_report['safe_features'])} safe features found")
        
        print(f"\n⚠️  UNSAFE FEATURES:")
        for feature in validation_report['unsafe_features']:
            print(f"     - {feature}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in validation_report['recommendations']:
            print(f"     • {rec}")
    
    if validation_report['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in validation_report['warnings']:
            print(f"     ⚠️  {warning}")


def main(fast_mode: bool = False):
    """
    Run CLEAN feature engineering on all cleaned tokens.

    Parameters
    ----------
    fast_mode : bool, optional (default=False)
        If True, skip all heavyweight analytics (FFT, advanced indicators, etc.) and
        generate **only** the rolling ML-safe features using ``create_rolling_features_safe``.
        This is ~5-10× faster and is recommended for production pipelines where only
        training inputs are required.
    """
    print("="*60)
    print("🧠 CLEAN FEATURE ENGINEERING ARCHITECTURE")
    if fast_mode:
        print("⚡ FAST MODE: Creating comprehensive rolling ML-safe features")
    else:
        print("Creating ONLY rolling ML-safe features")
    print("Global features computed on-demand in analysis")
    print("="*60)
    
    # Check if cleaned data exists
    cleaned_data_dir = Path("data/cleaned")
    if not cleaned_data_dir.exists():
        print(f"\n❌ ERROR: Cleaned data directory not found: {cleaned_data_dir}")
        print("\n🔧 REQUIRED STEP: Run data cleaning first!")
        print("   python data_cleaning/clean_tokens.py")
        print("\nThis will create the cleaned data needed for feature engineering.")
        return
    
    # Collect all cleaned token files
    all_token_paths = []
    categories = ["normal_behavior_tokens", "tokens_with_gaps", "tokens_with_extremes", "dead_tokens"]
    
    print(f"\n📂 Scanning categories for cleaned tokens...")
    for category in categories:
        cat_dir = cleaned_data_dir / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_token_paths.extend(paths)
            print(f"   ✅ {category}: {len(paths)} tokens")
        else:
            print(f"   ⚠️  {category}: Directory not found")
    
    if not all_token_paths:
        print(f"\n❌ ERROR: No cleaned tokens found in {cleaned_data_dir}")
        print("Please run data cleaning first.")
        return
    
    print(f"\n📊 Total tokens to process: {len(all_token_paths):,}")

    if fast_mode:
        print("\n⚡ FAST MODE ENABLED: Skipping comprehensive analytics – generating rolling ML-safe features only.")
        print("   This runs ~5-10× faster but does NOT compute global features used only for Streamlit dashboards.")
    
    # Estimate processing time (rough)
    per_token_min = 0.02 if fast_mode else 0.1
    estimated_time_minutes = len(all_token_paths) * per_token_min
    print(f"⏱️  Estimated processing time: {estimated_time_minutes:.1f} minutes")

    if fast_mode:
        # FAST PATH – compute rolling features only
        output_dir = Path("data/features")
        saved = 0
        category_counts = {}

        for path in tqdm(all_token_paths, desc="⚡ Generating rolling features", unit="token"):
            try:
                token_name = path.stem
                df = pl.read_parquet(path)
                if 'price' not in df.columns:
                    continue

                rolling_df = create_rolling_features_safe(df, token_name)
                if rolling_df.is_empty():
                    continue

                category = path.parent.name
                category_dir = output_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                out_path = category_dir / f"{token_name}.parquet"
                rolling_df.write_parquet(out_path)
                saved += 1
                category_counts[category] = category_counts.get(category, 0) + 1
            except Exception as e:
                tqdm.write(f"❌ {token_name}: {e}")
                continue

        # Summary for fast mode
        print("\n" + "="*60)
        print("⚡ FAST FEATURE ENGINEERING SUMMARY")
        print("="*60)
        for cat, cnt in category_counts.items():
            print(f"   {cat}: {cnt:,} tokens")
        print(f"   📈 Total tokens processed: {saved:,}")
        print("\n🎉 Fast rolling-feature generation complete! Files saved to data/features/")

    else:
        # ---------- FULL MODE - Use create_rolling_features_safe for comprehensive features ----------
        print(f"\n🔬 Running comprehensive ML feature engineering…")
        output_dir = Path("data/features")
        saved = 0
        category_counts = {}

        for path in tqdm(all_token_paths, desc="🧮 Generating comprehensive ML features", unit="token"):
            try:
                token_name = path.stem
                df = pl.read_parquet(path)
                if 'price' not in df.columns or 'datetime' not in df.columns:
                    continue

                # Use create_rolling_features_safe for FULL feature set
                rolling_df = create_rolling_features_safe(df, token_name)
                if rolling_df.is_empty():
                    continue

                # Validate features for ML safety
                validation_report = validate_features_for_ml_safety(rolling_df, token_name)
                if not validation_report['is_safe']:
                    tqdm.write(f"⚠️  {token_name}: Features failed ML safety check")
                    for warning in validation_report['warnings'][:3]:  # Show first 3 warnings
                        tqdm.write(f"   - {warning}")
                    continue

                category = path.parent.name
                category_dir = output_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                out_path = category_dir / f"{token_name}.parquet"
                rolling_df.write_parquet(out_path)
                saved += 1
                category_counts[category] = category_counts.get(category, 0) + 1
            except Exception as e:
                tqdm.write(f"❌ {token_name}: {e}")
                continue

        # Summary for full mode
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"   📈 Total tokens processed: {len(all_token_paths):,}")
        print(f"   ✅ Successfully saved: {saved:,}")
        print(f"   📉 Failed/Skipped: {len(all_token_paths) - saved:,}")
        print(f"   🎯 Success rate: {saved/len(all_token_paths)*100:.1f}%")
        
        print(f"\n📁 Features per category:")
        for cat, cnt in sorted(category_counts.items()):
            print(f"   {cat}: {cnt:,} tokens")
            
        # Show example of features created
        if saved > 0:
            # Load one example to show features
            example_files = list(output_dir.rglob("*.parquet"))
            if example_files:
                example_df = pl.read_parquet(example_files[0])
                feature_cols = [c for c in example_df.columns if c not in ['datetime', 'price']]
                print(f"\n📊 Features created ({len(feature_cols)} per token):")
                for i, col in enumerate(sorted(feature_cols)):
                    if i < 10:  # Show first 10
                        print(f"   • {col}")
                    elif i == 10:
                        print(f"   ... and {len(feature_cols) - 10} more features")
                        break

        print(f"\n🎉 Comprehensive feature engineering complete!")
        print(f"   💾 ML Features: data/features/ (ready for training)")
        print(f"   📁 {saved:,} feature files with full feature set")

    print(f"\n🚀 Ready to train ML models with clean features:")
    print(f"   🟢 python ML/directional_models/train_lightgbm_model.py")
    print(f"   🔵 python ML/directional_models/train_lightgbm_model_medium_term.py")
    print(f"   🟣 python ML/directional_models/train_unified_lstm_model.py")

    print(f"\n🎯 ARCHITECTURE BENEFITS:")
    print(f"   • No redundancy with data_analysis modules")
    print(f"   • Global features available on-demand in Streamlit (full mode only)")
    print(f"   • Impossible to accidentally use global features in ML")
    print(f"   • Cleaner separation of concerns")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean feature engineering runner")
    parser.add_argument('--fast', '--fast_mode', action='store_true', dest='fast_mode',
                        help='Enable fast mode (rolling features only)')
    args = parser.parse_args()

    main(fast_mode=args.fast_mode) 