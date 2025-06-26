"""
Advanced Feature Engineering for Memecoin Analysis
Implements missing features from roadmap sections 1-3:
- Log-returns calculation
- FFT analysis for cyclical patterns
- Advanced technical indicators (MACD, Bollinger Bands, ATR, etc.)
- Statistical moments (skewness, kurtosis)
- Multi-granularity downsampling
- Formal outlier detection (winsorization, z-score, IQR)
"""

import polars as pl
import numpy as np
import pandas as pd
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
        self.outlier_methods = {
            'winsorization': self._winsorize,
            'z_score': self._z_score_outliers,
            'iqr': self._iqr_outliers
        }
    
    def create_comprehensive_features(self, df: pl.DataFrame, 
                                    token_name: str = "token",
                                    outlier_method: str = 'winsorization',
                                    winsor_limits: Tuple[float, float] = (0.01, 0.01)) -> Dict:
        """
        Create comprehensive feature set including all roadmap requirements
        
        Args:
            df: DataFrame with 'datetime' and 'price' columns
            token_name: Token identifier
            outlier_method: Method for outlier detection ('winsorization', 'z_score', 'iqr')
            winsor_limits: Limits for winsorization (lower, upper)
            
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
            # 1. Basic price processing and log-returns
            price_features = self._calculate_log_returns(df, outlier_method, winsor_limits)
            
            # 2. Advanced technical indicators
            technical_features = self._calculate_advanced_technical_indicators(df)
            
            # 3. Statistical moments
            moment_features = self._calculate_statistical_moments(price_features['log_returns'])
            
            # 4. Multi-granularity analysis
            granularity_features = self._multi_granularity_analysis(df)
            
            # 5. FFT analysis for cyclical patterns
            fft_features = self._fft_cyclical_analysis(price_features['log_returns'])
            
            # 6. Outlier analysis
            outlier_features = self._comprehensive_outlier_analysis(df, outlier_method, winsor_limits)
            
            return {
                'token': token_name,
                'status': 'success',
                'price_features': price_features,
                'technical_features': technical_features,
                'moment_features': moment_features,
                'granularity_features': granularity_features,
                'fft_features': fft_features,
                'outlier_features': outlier_features,
                'data_quality': {
                    'total_points': len(df),
                    'time_span_hours': self._calculate_time_span_hours(df),
                    'price_range': price_features['price_stats']['price_range_pct']
                }
            }
            
        except Exception as e:
            return self._empty_feature_report(token_name, f"Error: {str(e)}")
    
    def _calculate_log_returns(self, df: pl.DataFrame, 
                             outlier_method: str,
                             winsor_limits: Tuple[float, float]) -> Dict:
        """Calculate log-returns and basic price statistics"""
        
        # Calculate log-returns using Polars - CRITICAL ROADMAP REQUIREMENT
        price_col = 'close' if 'close' in df.columns else 'price'
        
        # Calculate log-returns with Polars
        df_with_returns = df.with_columns([
            (pl.col(price_col).log() - pl.col(price_col).shift(1).log()).alias('log_returns')
        ])
        
        # Convert to numpy for further processing
        log_returns_raw = df_with_returns['log_returns'].to_numpy()
        
        # Handle outliers in log-returns
        if outlier_method in self.outlier_methods:
            log_returns_clean = self.outlier_methods[outlier_method](
                log_returns_raw, winsor_limits
            )
        else:
            log_returns_clean = log_returns_raw
        
        # Remove NaN values
        log_returns_clean = log_returns_clean[~np.isnan(log_returns_clean)]
        
        # Calculate cumulative log-returns
        cumulative_log_returns = np.cumsum(log_returns_clean)
        
        # Basic price statistics using Polars
        prices = df[price_col].to_numpy()
        first_price = prices[0]
        last_price = prices[-1]
        max_price = np.max(prices)
        min_price = np.min(prices)
        
        price_stats = {
            'total_return_pct': ((last_price - first_price) / first_price) * 100,
            'max_gain_pct': ((max_price - first_price) / first_price) * 100,
            'max_drawdown_pct': ((min_price - max_price) / max_price) * 100,
            'price_range_pct': ((max_price - min_price) / first_price) * 100,
            'log_return_mean': np.mean(log_returns_clean),
            'log_return_std': np.std(log_returns_clean),
            'log_return_sharpe': np.mean(log_returns_clean) / np.std(log_returns_clean) if np.std(log_returns_clean) > 0 else 0
        }
        
        return {
            'log_returns': log_returns_clean,
            'cumulative_log_returns': cumulative_log_returns,
            'price_stats': price_stats,
            'outlier_method_used': outlier_method
        }
    
    def _calculate_advanced_technical_indicators(self, df: pl.DataFrame) -> Dict:
        """Calculate advanced technical indicators: MACD, Bollinger Bands, ATR, etc."""
        
        # Use Polars for technical indicators
        price_col = 'close' if 'close' in df.columns else 'price'
        prices = df[price_col].to_numpy()
        
        indicators = {}
        
        # 1. MACD (Moving Average Convergence Divergence)
        try:
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            macd_line = ema_12 - ema_26
            signal_line = self._calculate_ema(macd_line, 9)
            macd_histogram = macd_line - signal_line
            
            indicators['macd'] = {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': macd_histogram,
                'current_position': 'bullish' if macd_line[-1] > signal_line[-1] else 'bearish'
            }
        except:
            indicators['macd'] = None
        
        # 2. Bollinger Bands
        try:
            sma_20 = pd.Series(prices).rolling(20).mean()
            std_20 = pd.Series(prices).rolling(20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            bb_position = (prices - sma_20) / (std_20 * 2)  # Position within bands
            
            indicators['bollinger_bands'] = {
                'upper_band': bb_upper.values,
                'middle_band': sma_20.values,
                'lower_band': bb_lower.values,
                'bb_position': bb_position.values,
                'current_position': bb_position.iloc[-1] if not pd.isna(bb_position.iloc[-1]) else 0,
                'squeeze_periods': (std_20 < std_20.rolling(20).mean() * 0.8).sum()  # Low volatility periods
            }
        except:
            indicators['bollinger_bands'] = None
        
        # 3. ATR (Average True Range)
        try:
            high = pd.Series(prices)  # Using price as proxy for high
            low = pd.Series(prices)   # Using price as proxy for low
            close = pd.Series(prices)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            indicators['atr'] = {
                'atr_values': atr.values,
                'current_atr': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
                'atr_pct': (atr / prices).iloc[-1] * 100 if not pd.isna(atr.iloc[-1]) else 0
            }
        except:
            indicators['atr'] = None
        
        # 4. Stochastic Oscillator
        try:
            period = min(14, len(prices) - 1)
            if period > 5:
                high_period = pd.Series(prices).rolling(period).max()
                low_period = pd.Series(prices).rolling(period).min()
                k_percent = 100 * (prices - low_period) / (high_period - low_period)
                d_percent = k_percent.rolling(3).mean()
                
                indicators['stochastic'] = {
                    'k_percent': k_percent.values,
                    'd_percent': d_percent.values,
                    'current_k': k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50,
                    'current_d': d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50,
                    'signal': 'overbought' if k_percent.iloc[-1] > 80 else 'oversold' if k_percent.iloc[-1] < 20 else 'neutral'
                }
            else:
                indicators['stochastic'] = None
        except:
            indicators['stochastic'] = None
        
        # 5. Williams %R
        try:
            period = min(14, len(prices) - 1)
            if period > 5:
                high_period = pd.Series(prices).rolling(period).max()
                low_period = pd.Series(prices).rolling(period).min()
                williams_r = -100 * (high_period - prices) / (high_period - low_period)
                
                indicators['williams_r'] = {
                    'values': williams_r.values,
                    'current_value': williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50,
                    'signal': 'overbought' if williams_r.iloc[-1] > -20 else 'oversold' if williams_r.iloc[-1] < -80 else 'neutral'
                }
            else:
                indicators['williams_r'] = None
        except:
            indicators['williams_r'] = None
        
        # 6. Enhanced RSI with divergence detection
        try:
            period = min(14, len(prices) - 1)
            if period > 5:
                rsi_values = self._calculate_rsi(prices, period)
                rsi_divergence = self._detect_rsi_divergence(prices, rsi_values)
                
                indicators['enhanced_rsi'] = {
                    'values': rsi_values,
                    'current_rsi': rsi_values[-1] if len(rsi_values) > 0 else 50,
                    'divergence': rsi_divergence,
                    'signal': 'overbought' if rsi_values[-1] > 70 else 'oversold' if rsi_values[-1] < 30 else 'neutral'
                }
            else:
                indicators['enhanced_rsi'] = None
        except:
            indicators['enhanced_rsi'] = None
        
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
    
    def _comprehensive_outlier_analysis(self, df: pl.DataFrame,
                                      method: str,
                                      winsor_limits: Tuple[float, float]) -> Dict:
        """Comprehensive outlier detection and analysis"""
        
        price_col = 'close' if 'close' in df.columns else 'price'
        prices = df[price_col].to_numpy()
        
        outlier_results = {}
        
        # 1. Winsorization
        try:
            winsorized_prices = self._winsorize(prices, winsor_limits)
            outliers_winsor = np.sum(winsorized_prices != prices)
            outlier_results['winsorization'] = {
                'outliers_detected': int(outliers_winsor),
                'outlier_rate': float(outliers_winsor / len(prices)),
                'method_applied': method == 'winsorization'
            }
        except:
            outlier_results['winsorization'] = None
        
        # 2. Z-score method
        try:
            z_scores = np.abs(stats.zscore(prices, nan_policy='omit'))
            z_outliers = np.sum(z_scores > 3)
            outlier_results['z_score'] = {
                'outliers_detected': int(z_outliers),
                'outlier_rate': float(z_outliers / len(prices)),
                'max_z_score': float(np.max(z_scores)),
                'method_applied': method == 'z_score'
            }
        except:
            outlier_results['z_score'] = None
        
        # 3. IQR method
        try:
            Q1 = np.percentile(prices, 25)
            Q3 = np.percentile(prices, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = np.sum((prices < lower_bound) | (prices > upper_bound))
            
            outlier_results['iqr'] = {
                'outliers_detected': int(iqr_outliers),
                'outlier_rate': float(iqr_outliers / len(prices)),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'iqr_value': float(IQR),
                'method_applied': method == 'iqr'
            }
        except:
            outlier_results['iqr'] = None
        
        # 4. Modified Z-score (robust)
        try:
            median = np.median(prices)
            mad = np.median(np.abs(prices - median))
            modified_z_scores = 0.6745 * (prices - median) / mad if mad > 0 else np.zeros_like(prices)
            modified_z_outliers = np.sum(np.abs(modified_z_scores) > 3.5)
            
            outlier_results['modified_z_score'] = {
                'outliers_detected': int(modified_z_outliers),
                'outlier_rate': float(modified_z_outliers / len(prices)),
                'max_modified_z': float(np.max(np.abs(modified_z_scores)))
            }
        except:
            outlier_results['modified_z_score'] = None
        
        return outlier_results
    
    # Helper methods
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean()
        avg_losses = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def _detect_rsi_divergence(self, prices: np.ndarray, rsi: np.ndarray) -> Dict:
        """Detect RSI divergence patterns"""
        try:
            # Find recent peaks and troughs
            price_peaks, _ = find_peaks(prices[-50:])  # Last 50 points
            rsi_peaks, _ = find_peaks(rsi[-50:])
            
            # Simple divergence detection
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                recent_price_trend = prices[-50:][price_peaks[-1]] - prices[-50:][price_peaks[-2]]
                recent_rsi_trend = rsi[-50:][rsi_peaks[-1]] - rsi[-50:][rsi_peaks[-2]]
                
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
    
    def _winsorize(self, data: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Winsorize data to handle outliers"""
        return stats.mstats.winsorize(data, limits=limits)
    
    def _z_score_outliers(self, data: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Remove outliers using z-score method"""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return np.where(z_scores > 3, np.median(data), data)
    
    def _iqr_outliers(self, data: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Remove outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(data, lower_bound, upper_bound)
    
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
            'fft_features': None,
            'outlier_features': None
        }


def batch_feature_engineering(data_paths: List, 
                            outlier_method: str = 'winsorization',
                            limit: Optional[int] = None) -> Dict[str, Dict]:
    """
    Process multiple tokens with advanced feature engineering
    
    Args:
        data_paths: List of paths to token data files
        outlier_method: Outlier detection method
        limit: Maximum number of tokens to process
        
    Returns:
        Dictionary with token features
    """
    engineer = AdvancedFeatureEngineer()
    results = {}
    
    files_to_process = data_paths[:limit] if limit else data_paths
    
    print(f"🔬 Processing {len(files_to_process)} tokens with advanced feature engineering...")
    
    # Use tqdm for progress tracking
    for path in tqdm(files_to_process, desc="🧮 Extracting features", unit="token"):
        try:
            token_name = path.stem if hasattr(path, 'stem') else str(path).split('/')[-1].split('.')[0]
            
            # Load data
            if str(path).endswith('.parquet'):
                df = pl.read_parquet(path)
            else:
                continue
            
            if 'price' not in df.columns or 'datetime' not in df.columns:
                continue
            
            # Engineer features
            features = engineer.create_comprehensive_features(
                df, token_name, outlier_method
            )
            
            results[token_name] = features
                
        except Exception as e:
            # Use tqdm.write to print without interfering with progress bar
            tqdm.write(f"❌ Error processing {path}: {e}")
            continue
    
    successful_count = len([r for r in results.values() if r.get('status') == 'success'])
    print(f"✅ Successfully processed {successful_count}/{len(results)} tokens")
    return results 


def save_features_to_files(features_dict: Dict[str, Dict], output_dir: Path = Path("data/features")) -> None:
    """
    Save engineered features to individual parquet files for ML training
    
    Args:
        features_dict: Dictionary with token features from batch_feature_engineering
        output_dir: Directory to save feature files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    
    print(f"\nSaving features to {output_dir}...")
    
    for token_name, features in tqdm(features_dict.items(), desc="💾 Saving feature files", unit="token"):
        if features['status'] != 'success':
            continue
            
        try:
            # Extract all features into a flat structure for ML training
            feature_data = {}
            
            # Add basic price features
            if features['price_features']:
                price_features = features['price_features']
                log_returns = price_features['log_returns']
                cumulative_returns = price_features['cumulative_log_returns']
                
                # Create feature vectors by expanding arrays to match length
                min_length = min(len(log_returns), len(cumulative_returns))
                
                feature_data.update({
                    'log_returns': log_returns[:min_length],
                    'cumulative_log_returns': cumulative_returns[:min_length],
                    'total_return_pct': [price_features['price_stats']['total_return_pct']] * min_length,
                    'max_gain_pct': [price_features['price_stats']['max_gain_pct']] * min_length,
                    'volatility': [price_features['price_stats']['log_return_std']] * min_length,
                    'sharpe_ratio': [price_features['price_stats']['log_return_sharpe']] * min_length
                })
            
            # Add technical indicators
            if features['technical_features']:
                tech_features = features['technical_features']
                
                # MACD features
                if tech_features.get('macd'):
                    macd = tech_features['macd']
                    macd_length = len(macd['macd_line'])
                    feature_data.update({
                        'macd_line': macd['macd_line'],
                        'macd_signal': macd['signal_line'],
                        'macd_histogram': macd['histogram'],
                        'macd_position': [1.0 if macd['current_position'] == 'bullish' else 0.0] * macd_length
                    })
                
                # Bollinger Bands
                if tech_features.get('bollinger_bands'):
                    bb = tech_features['bollinger_bands']
                    bb_length = len(bb['bb_position'])
                    feature_data.update({
                        'bb_position': bb['bb_position'],
                        'bb_squeeze_periods': [bb['squeeze_periods']] * bb_length
                    })
                
                # Enhanced RSI
                if tech_features.get('enhanced_rsi') and isinstance(tech_features['enhanced_rsi'], dict):
                    rsi = tech_features['enhanced_rsi']
                    if 'values' in rsi and rsi['values'] is not None and len(rsi['values']) > 0:
                        rsi_length = len(rsi['values'])
                        
                        # Calculate RSI trend from the last values
                        rsi_values = rsi['values']
                        rsi_trend = 1.0 if len(rsi_values) > 10 and rsi_values[-1] > rsi_values[-10] else 0.0
                        
                        # Calculate momentum score from divergence
                        momentum_score = 0.5  # Default neutral
                        if rsi.get('divergence') and rsi['divergence'].get('has_divergence'):
                            momentum_score = 0.8 if rsi['divergence']['type'] == 'bullish' else 0.2
                        
                        feature_data.update({
                            'rsi_values': rsi['values'],
                            'rsi_trend': [rsi_trend] * rsi_length,
                            'rsi_momentum': [momentum_score] * rsi_length,
                            'rsi_current': [rsi.get('current_rsi', 50)] * rsi_length
                        })
                
                # ATR (Average True Range)
                if tech_features.get('atr') and isinstance(tech_features['atr'], dict):
                    atr = tech_features['atr']
                    if 'values' in atr and atr['values'] is not None:
                        atr_length = len(atr['values'])
                        feature_data.update({
                            'atr_values': atr['values'],
                            'atr_normalized': atr.get('normalized_values', atr['values'])
                        })
            
            # Add statistical moments
            if features['moment_features'] and features['moment_features']['moments_valid']:
                moments = features['moment_features']
                base_length = len(feature_data.get('log_returns', [100]))  # Fallback length
                
                feature_data.update({
                    'skewness': [moments['skewness']] * base_length,
                    'kurtosis': [moments['kurtosis']] * base_length,
                    'variance': [moments['variance']] * base_length,
                    'semi_deviation': [moments['semi_deviation']] * base_length,
                    'value_at_risk_95': [moments['value_at_risk_95']] * base_length,
                    'expected_shortfall_95': [moments['expected_shortfall_95']] * base_length
                })
            
            # Add FFT features
            if features['fft_features'] and features['fft_features']['fft_available']:
                fft_features = features['fft_features']
                base_length = len(feature_data.get('log_returns', [100]))  # Fallback length
                
                feature_data.update({
                    'spectral_entropy': [fft_features['spectral_entropy']] * base_length,
                    'max_periodicity_strength': [fft_features['max_periodicity_strength']] * base_length,
                    'has_strong_cycles': [1.0 if fft_features['has_strong_cycles'] else 0.0] * base_length,
                    'dominant_period_1': [fft_features['dominant_periods_minutes'][0] if fft_features['dominant_periods_minutes'] else 0] * base_length
                })
            
            # Ensure all features have the same length
            if feature_data:
                max_length = max(len(v) for v in feature_data.values() if isinstance(v, (list, np.ndarray)))
                min_length = min(len(v) for v in feature_data.values() if isinstance(v, (list, np.ndarray)))
                
                # Truncate all to min_length for consistency
                for key, value in feature_data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > min_length:
                        feature_data[key] = value[:min_length]
                
                # Create synthetic datetime and price columns for ML compatibility
                feature_data['datetime'] = pd.date_range(start='2025-01-01', periods=min_length, freq='1min')
                feature_data['price'] = np.ones(min_length)  # Placeholder prices
                
                # Save as parquet
                features_df = pl.DataFrame(feature_data)
                output_path = output_dir / f"{token_name}_features.parquet"
                features_df.write_parquet(output_path)
                saved_count += 1
                
        except Exception as e:
            print(f"Error saving features for {token_name}: {e}")
            continue
    
    print(f"✅ Successfully saved {saved_count} feature files to {output_dir}")


def main():
    """
    Main function to run feature engineering on all cleaned tokens
    This creates the pre-engineered features needed by ML training scripts
    """
    print("="*60)
    print("ADVANCED FEATURE ENGINEERING")
    print("Creating pre-engineered features for ML training")
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
    
    # Estimate processing time
    estimated_time_minutes = len(all_token_paths) * 0.1  # ~0.1 min per token
    print(f"⏱️  Estimated processing time: {estimated_time_minutes:.1f} minutes")
    
    # Run batch feature engineering
    print(f"\n🔬 Running advanced feature engineering...")
    features_dict = batch_feature_engineering(
        all_token_paths,
        outlier_method='winsorization',
        limit=None  # Process all tokens
    )
    
    # Save features to files for ML training
    save_features_to_files(features_dict)
    
    # Print summary
    successful_features = sum(1 for f in features_dict.values() if f['status'] == 'success')
    
    print(f"\n" + "="*60)
    print(f"📊 FEATURE ENGINEERING SUMMARY")
    print(f"="*60)
    print(f"   📈 Total tokens processed: {len(features_dict):,}")
    print(f"   ✅ Successful extractions: {successful_features:,}")
    print(f"   📉 Failed extractions: {len(features_dict) - successful_features:,}")
    print(f"   🎯 Success rate: {successful_features/len(features_dict)*100:.1f}%")
    
    print(f"\n🎉 Feature engineering complete!")
    print(f"   💾 Features saved to: data/features/")
    print(f"   📁 {successful_features:,} feature files ready for ML training")
    
    print(f"\n🚀 Ready to train ML models:")
    print(f"   🟢 python ML/directional_models/train_lightgbm_model.py")
    print(f"   🔵 python ML/directional_models/train_lightgbm_model_medium_term.py")
    print(f"   🟣 python ML/directional_models/train_unified_lstm_model.py")


if __name__ == "__main__":
    main() 