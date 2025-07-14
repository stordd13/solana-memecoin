# autocorrelation_analysis.py
"""
Core Autocorrelation Analysis Module for Memecoin Time Series
Focused on analyzing autocorrelation patterns to determine optimal prediction horizons
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import mstats
from joblib import Parallel, delayed
import json
from datetime import datetime
from tqdm import tqdm

from time_series.legacy.archetype_utils import prepare_token_data, safe_divide


class AutocorrelationAnalyzer:
    """
    Focused autocorrelation analysis for memecoin time series data.
    Designed to determine optimal prediction horizons for deep learning models.
    """
    
    def __init__(self, max_lag: int = 240, confidence_level: float = 0.95):
        """
        Initialize the autocorrelation analyzer.
        
        Args:
            max_lag: Maximum lag to compute autocorrelation for (default 240 = 4 hours)
            confidence_level: Confidence level for ACF confidence intervals
        """
        self.max_lag = max_lag
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def compute_token_autocorrelation(self, 
                                    prices: np.ndarray, 
                                    returns: np.ndarray,
                                    token_name: str = "unknown",
                                    analysis_type: str = "returns") -> Dict:
        """
        Compute autocorrelation function for a single token.
        
        Args:
            prices: Array of token prices
            returns: Array of token returns
            token_name: Name of the token for debugging
            analysis_type: Type of analysis ('returns', 'log_returns', 'prices')
            
        Returns:
            Dictionary containing ACF results and metadata
        """
        # Select appropriate time series based on analysis type
        if analysis_type == "returns":
            series = returns
        elif analysis_type == "log_returns":
            # Convert returns to log returns for extreme values (more robust)
            series = np.log(1 + returns + 1e-10)
        elif analysis_type == "prices":
            series = prices
        elif analysis_type == "log_prices":
            # Log of prices - reveals multiplicative patterns
            series = np.log(prices + 1e-10)
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}. Valid options: returns, log_returns, prices, log_prices")
        
        # Handle edge cases (relaxed for memecoin short lifespans)
        if len(series) < 5:
            return self._create_empty_acf_result(token_name, analysis_type, "insufficient_data")
        
        # Check for constant series
        if np.std(series) < 1e-10:
            return self._create_empty_acf_result(token_name, analysis_type, "constant_series")
        
        # Remove NaN/inf values
        clean_series = series[np.isfinite(series)]
        if len(clean_series) < 5:
            return self._create_empty_acf_result(token_name, analysis_type, "insufficient_clean_data")
        
        # Apply less aggressive winsorization for memecoins (preserve extreme moves!)
        clean_series = mstats.winsorize(clean_series, limits=[0.001, 0.001])  # Only remove extreme outliers
        
        # Compute ACF with robust error handling
        try:
            # Adjust max_lag based on series length (more generous for short memecoin series)
            effective_max_lag = min(self.max_lag, max(10, len(clean_series) - 10))
            
            print(f"DEBUG: {token_name} - Series length: {len(clean_series)}, Effective max lag: {effective_max_lag}")
            
            # Try robust ACF computation with fallbacks
            acf_vals = None
            conf_int = None
            
            # Method 1: Try statsmodels with confidence intervals
            try:
                acf_result = acf(clean_series, nlags=effective_max_lag, fft=True, alpha=self.alpha)
                if isinstance(acf_result, tuple):
                    acf_vals, conf_int = acf_result
                else:
                    acf_vals = acf_result
                print(f"DEBUG: {token_name} - ACF method 1 succeeded, got {len(acf_vals) if acf_vals is not None else 0} lags")
            except Exception as e:
                print(f"DEBUG: {token_name} - ACF method 1 failed: {e}")
            
            # Method 2: Try statsmodels without confidence intervals as fallback
            if acf_vals is None or len(acf_vals) == 0:
                try:
                    acf_vals = acf(clean_series, nlags=effective_max_lag, fft=False, alpha=None)
                    print(f"DEBUG: {token_name} - ACF method 2 succeeded, got {len(acf_vals)} lags")
                except Exception as e:
                    print(f"DEBUG: {token_name} - ACF method 2 failed: {e}")
            
            # Method 3: Manual correlation computation as last resort
            if acf_vals is None or len(acf_vals) == 0:
                print(f"DEBUG: {token_name} - Using manual ACF computation")
                acf_vals = self._manual_acf_computation(clean_series, effective_max_lag)
            
            # Validate ACF results
            if acf_vals is None or len(acf_vals) == 0:
                print(f"DEBUG: {token_name} - All ACF methods failed!")
                return self._create_empty_acf_result(token_name, analysis_type, "acf_computation_failed")
            
            # Check for NaN/inf in ACF values
            if np.any(~np.isfinite(acf_vals)):
                print(f"DEBUG: {token_name} - ACF contains NaN/inf values")
                # Clean the ACF values
                acf_vals = np.nan_to_num(acf_vals, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"DEBUG: {token_name} - Final ACF: {len(acf_vals)} values, range: [{np.min(acf_vals):.3f}, {np.max(acf_vals):.3f}]")
            
            # Compute PACF for additional analysis
            pacf_values = pacf(clean_series, nlags=min(20, effective_max_lag), method='ols')
            
            # Find significant lags (outside confidence intervals)
            significant_lags = []
            if conf_int is not None:
                for lag in range(1, len(acf_vals)):
                    if acf_vals[lag] < conf_int[lag, 0] or acf_vals[lag] > conf_int[lag, 1]:
                        significant_lags.append(lag)
            
            # Calculate key statistics
            max_acf_lag = np.argmax(np.abs(acf_vals[1:]))  # Exclude lag 0
            max_acf_value = acf_vals[max_acf_lag + 1]
            
            # Find first zero crossing
            zero_crossings = []
            for i in range(1, len(acf_vals) - 1):
                if (acf_vals[i] > 0 and acf_vals[i + 1] < 0) or (acf_vals[i] < 0 and acf_vals[i + 1] > 0):
                    zero_crossings.append(i)
            
            # Compute decay rate (how fast ACF decays)
            decay_rate = 0.0
            if len(acf_vals) > 10:
                try:
                    # Fit exponential decay to first 10 lags
                    lags = np.arange(1, 11)
                    acf_subset = np.abs(acf_vals[1:11])
                    if np.all(acf_subset > 0):
                        decay_rate = -np.polyfit(lags, np.log(acf_subset), 1)[0]
                except:
                    decay_rate = 0.0
            
            return {
                'token_name': token_name,
                'analysis_type': analysis_type,
                'success': True,
                'series_length': len(clean_series),
                'effective_max_lag': effective_max_lag,
                'acf_values': acf_vals.tolist(),
                'pacf_values': pacf_values.tolist(),
                'confidence_intervals': conf_int.tolist() if conf_int is not None else None,
                'significant_lags': significant_lags,
                'max_acf_lag': int(max_acf_lag + 1),
                'max_acf_value': float(max_acf_value),
                'first_zero_crossing': int(zero_crossings[0]) if zero_crossings else effective_max_lag,
                'decay_rate': float(decay_rate),
                'mean_acf_1_to_10': float(np.mean(np.abs(acf_vals[1:11]))),
                'mean_acf_1_to_20': float(np.mean(np.abs(acf_vals[1:21]))) if len(acf_vals) > 20 else None,
                'std_acf_1_to_10': float(np.std(acf_vals[1:11])),
                'series_stats': {
                    'mean': float(np.mean(clean_series)),
                    'std': float(np.std(clean_series)),
                    'min': float(np.min(clean_series)),
                    'max': float(np.max(clean_series)),
                    'skewness': float(mstats.skew(clean_series)),
                    'kurtosis': float(mstats.kurtosis(clean_series))
                }
            }
            
        except Exception as e:
            print(f"DEBUG: {token_name} - Critical error in compute_token_autocorrelation: {e}")
            return self._create_empty_acf_result(token_name, analysis_type, f"computation_error: {str(e)}")
    
    def _manual_acf_computation(self, series: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Manual autocorrelation computation as fallback when statsmodels fails.
        """
        try:
            n = len(series)
            if n < 2:
                return np.array([1.0])
            
            # Center the series
            mean_series = np.mean(series)
            centered = series - mean_series
            
            # Compute variance
            variance = np.var(centered)
            if variance < 1e-10:
                # Constant series
                return np.ones(min(max_lag + 1, n))
            
            # Compute autocorrelations manually
            acf_vals = []
            for lag in range(min(max_lag + 1, n)):
                if lag == 0:
                    acf_vals.append(1.0)
                elif lag < n:
                    # Compute correlation at this lag
                    x1 = centered[:-lag]
                    x2 = centered[lag:]
                    
                    if len(x1) > 0:
                        correlation = np.mean(x1 * x2) / variance
                        acf_vals.append(correlation)
                    else:
                        acf_vals.append(0.0)
                else:
                    acf_vals.append(0.0)
            
            return np.array(acf_vals)
            
        except Exception as e:
            print(f"DEBUG: Manual ACF computation failed: {e}")
            # Return minimal ACF with just lag 0
            return np.array([1.0])
    
    def _create_empty_acf_result(self, token_name: str, analysis_type: str, reason: str) -> Dict:
        """Create empty ACF result for failed computations."""
        return {
            'token_name': token_name,
            'analysis_type': analysis_type,
            'success': False,
            'failure_reason': reason,
            'series_length': 0,
            'effective_max_lag': 0,
            'acf_values': [],
            'pacf_values': [],
            'confidence_intervals': None,
            'significant_lags': [],
            'max_acf_lag': 0,
            'max_acf_value': 0.0,
            'first_zero_crossing': 0,
            'decay_rate': 0.0,
            'mean_acf_1_to_10': 0.0,
            'mean_acf_1_to_20': 0.0,
            'std_acf_1_to_10': 0.0,
            'series_stats': {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            }
        }
    
    def compute_autocorrelation_distributions(self, 
                                            token_data: Dict[str, pl.DataFrame],
                                            analysis_type: str = "returns",
                                            n_jobs: int = -1) -> Dict:
        """
        Compute autocorrelation distributions across all tokens.
        
        Args:
            token_data: Dictionary mapping token names to DataFrames
            analysis_type: Type of analysis ('returns', 'log_returns', 'prices')
            n_jobs: Number of parallel jobs for computation
            
        Returns:
            Dictionary containing distribution analysis results
        """
        print(f"Computing autocorrelation distributions for {len(token_data)} tokens...")
        
        # Prepare token data for parallel processing
        token_params = []
        for token_name, df in token_data.items():
            try:
                prices, returns, death_minute = prepare_token_data(df)
                token_params.append((prices, returns, token_name))
            except Exception as e:
                print(f"Error preparing {token_name}: {e}")
                continue
        
        print(f"Successfully prepared {len(token_params)} tokens for analysis")
        
        # Compute ACF for all tokens in parallel with progress bar
        print("Computing autocorrelation for each token...")
        acf_results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.compute_token_autocorrelation)(
                prices, returns, token_name, analysis_type
            ) for prices, returns, token_name in tqdm(token_params, desc="Processing tokens")
        )
        
        # Filter successful results
        successful_results = [result for result in acf_results if result['success']]
        failed_results = [result for result in acf_results if not result['success']]
        
        print(f"Successfully computed ACF for {len(successful_results)} tokens")
        print(f"Failed to compute ACF for {len(failed_results)} tokens")
        
        if not successful_results:
            return {'success': False, 'error': 'No successful ACF computations'}
        
        # Compute distribution statistics
        distribution_stats = self._compute_distribution_statistics(successful_results)
        
        # Extract category information if available
        category_analysis = self._analyze_by_category(successful_results, token_data)
        
        return {
            'success': True,
            'analysis_type': analysis_type,
            'total_tokens': len(token_data),
            'successful_tokens': len(successful_results),
            'failed_tokens': len(failed_results),
            'distribution_stats': distribution_stats,
            'category_analysis': category_analysis,
            'individual_results': successful_results,
            'failed_results': failed_results,
            'computation_metadata': {
                'max_lag': self.max_lag,
                'confidence_level': self.confidence_level,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _compute_distribution_statistics(self, acf_results: List[Dict]) -> Dict:
        """Compute distribution statistics across all ACF results."""
        # Extract common lag length
        min_lag_length = min(len(result['acf_values']) for result in acf_results)
        
        # Create matrix of ACF values (tokens x lags)
        acf_matrix = np.array([
            result['acf_values'][:min_lag_length] 
            for result in acf_results
        ])
        
        # Compute statistics for each lag
        lag_stats = {}
        for lag in range(min_lag_length):
            lag_values = acf_matrix[:, lag]
            lag_stats[f'lag_{lag}'] = {
                'mean': float(np.mean(lag_values)),
                'std': float(np.std(lag_values)),
                'median': float(np.median(lag_values)),
                'q25': float(np.percentile(lag_values, 25)),
                'q75': float(np.percentile(lag_values, 75)),
                'min': float(np.min(lag_values)),
                'max': float(np.max(lag_values)),
                'significant_tokens': int(np.sum(np.abs(lag_values) > 0.1))  # Threshold for significance
            }
        
        # Compute overall statistics
        overall_stats = {
            'max_acf_lag_distribution': {
                'mean': float(np.mean([r['max_acf_lag'] for r in acf_results])),
                'std': float(np.std([r['max_acf_lag'] for r in acf_results])),
                'median': float(np.median([r['max_acf_lag'] for r in acf_results]))
            },
            'max_acf_value_distribution': {
                'mean': float(np.mean([r['max_acf_value'] for r in acf_results])),
                'std': float(np.std([r['max_acf_value'] for r in acf_results])),
                'median': float(np.median([r['max_acf_value'] for r in acf_results]))
            },
            'decay_rate_distribution': {
                'mean': float(np.mean([r['decay_rate'] for r in acf_results])),
                'std': float(np.std([r['decay_rate'] for r in acf_results])),
                'median': float(np.median([r['decay_rate'] for r in acf_results]))
            },
            'first_zero_crossing_distribution': {
                'mean': float(np.mean([r['first_zero_crossing'] for r in acf_results])),
                'std': float(np.std([r['first_zero_crossing'] for r in acf_results])),
                'median': float(np.median([r['first_zero_crossing'] for r in acf_results]))
            }
        }
        
        return {
            'lag_statistics': lag_stats,
            'overall_statistics': overall_stats,
            'matrix_shape': acf_matrix.shape,
            'common_lag_length': min_lag_length
        }
    
    def _analyze_by_category(self, acf_results: List[Dict], token_data: Dict[str, pl.DataFrame]) -> Dict:
        """Analyze ACF results by token category if available."""
        category_analysis = {}
        
        # Group results by category
        for result in acf_results:
            token_name = result['token_name']
            if token_name in token_data:
                df = token_data[token_name]
                if 'category' in df.columns:
                    category = df['category'][0]
                    if category not in category_analysis:
                        category_analysis[category] = []
                    category_analysis[category].append(result)
        
        # Compute statistics for each category
        category_stats = {}
        for category, results in category_analysis.items():
            if results:
                category_stats[category] = {
                    'token_count': len(results),
                    'mean_max_acf_lag': float(np.mean([r['max_acf_lag'] for r in results])),
                    'mean_max_acf_value': float(np.mean([r['max_acf_value'] for r in results])),
                    'mean_decay_rate': float(np.mean([r['decay_rate'] for r in results])),
                    'mean_first_zero_crossing': float(np.mean([r['first_zero_crossing'] for r in results]))
                }
        
        return category_stats
    
    def identify_optimal_prediction_horizons(self, 
                                           distribution_results: Dict,
                                           significance_threshold: float = 0.1) -> Dict:
        """
        Identify optimal prediction horizons based on ACF analysis.
        
        Args:
            distribution_results: Results from compute_autocorrelation_distributions
            significance_threshold: Threshold for considering ACF values significant
            
        Returns:
            Dictionary with prediction horizon recommendations
        """
        if not distribution_results['success']:
            return {'success': False, 'error': 'Invalid distribution results'}
        
        lag_stats = distribution_results['distribution_stats']['lag_statistics']
        overall_stats = distribution_results['distribution_stats']['overall_statistics']
        
        # Find lags with strongest mean ACF values
        strong_lags = []
        for lag_key, stats in lag_stats.items():
            if 'lag_0' == lag_key:  # Skip lag 0 (always 1.0)
                continue
            if abs(stats['mean']) > significance_threshold:
                lag_num = int(lag_key.split('_')[1])
                strong_lags.append({
                    'lag': lag_num,
                    'mean_acf': stats['mean'],
                    'std_acf': stats['std'],
                    'significant_tokens': stats['significant_tokens']
                })
        
        # Sort by absolute mean ACF value
        strong_lags.sort(key=lambda x: abs(x['mean_acf']), reverse=True)
        
        # Analyze ACF patterns to find optimal horizons
        lag_stats = distribution_results['distribution_stats']['lag_statistics']
        
        # Find lags where mean ACF is still significant
        significant_mean_lags = []
        for lag_key, stats in lag_stats.items():
            if lag_key == 'lag_0':
                continue
            lag_num = int(lag_key.split('_')[1])
            mean_acf = abs(stats['mean'])
            
            # Consider significant if mean ACF is above threshold OR if many tokens show significance
            if mean_acf > significance_threshold or stats['significant_tokens'] > distribution_results['successful_tokens'] * 0.3:
                significant_mean_lags.append({
                    'lag': lag_num,
                    'mean_acf': stats['mean'],
                    'std_acf': stats['std'],  # Add missing std_acf field
                    'median_acf': stats['median'],
                    'significant_tokens': stats['significant_tokens'],
                    'significance_score': mean_acf * (stats['significant_tokens'] / distribution_results['successful_tokens'])
                })
        
        # Sort by significance score
        significant_mean_lags.sort(key=lambda x: x['significance_score'], reverse=True)
        
        # Find natural breakpoints in the ACF function
        first_zero_crossing = int(overall_stats['first_zero_crossing_distribution']['mean'])
        max_acf_lag = int(overall_stats['max_acf_lag_distribution']['mean'])
        
        # Generate smart recommendations based on actual data
        recommendations = {
            'short_term_horizon': 1,  # Always include 1-step
            'medium_term_horizon': 5,  # Default
            'long_term_horizon': 15,  # Default
            'custom_horizons': []
        }
        
        if significant_mean_lags:
            # Short term: First significant lag or strongest lag up to 5
            short_candidates = [l for l in significant_mean_lags if 1 <= l['lag'] <= 5]
            if short_candidates:
                recommendations['short_term_horizon'] = short_candidates[0]['lag']
            elif significant_mean_lags:
                recommendations['short_term_horizon'] = min(significant_mean_lags[0]['lag'], 5)
            
            # Medium term: Strongest lag between 3-15 or around max_acf_lag
            medium_candidates = [l for l in significant_mean_lags if 3 <= l['lag'] <= 15]
            if medium_candidates:
                recommendations['medium_term_horizon'] = medium_candidates[0]['lag']
            elif max_acf_lag <= 15:
                recommendations['medium_term_horizon'] = max_acf_lag
            
            # Long term: Before first zero crossing or strongest lag 10-30
            long_candidates = [l for l in significant_mean_lags if 10 <= l['lag'] <= min(30, first_zero_crossing)]
            if long_candidates:
                recommendations['long_term_horizon'] = long_candidates[0]['lag']
            elif first_zero_crossing > 10:
                recommendations['long_term_horizon'] = min(first_zero_crossing - 1, 30)
            
            # Custom horizons: Top significant lags with diverse spacing
            selected_horizons = []
            last_selected = 0
            for lag_info in significant_mean_lags[:20]:  # Look at top 20
                if lag_info['lag'] >= last_selected + 3:  # Ensure spacing
                    selected_horizons.append(lag_info)
                    last_selected = lag_info['lag']
                    if len(selected_horizons) >= 5:
                        break
            
            recommendations['custom_horizons'] = selected_horizons
        
        # Add data-driven insights
        recommendations['insights'] = {
            'persistence': 'high' if lag_stats.get('lag_1', {}).get('mean', 0) > 0.5 else 'medium' if lag_stats.get('lag_1', {}).get('mean', 0) > 0.2 else 'low',
            'predictability_window': first_zero_crossing,
            'optimal_feature_lags': [l['lag'] for l in significant_mean_lags[:10]],
            'decay_type': 'slow' if overall_stats['decay_rate_distribution']['mean'] < 0.1 else 'moderate' if overall_stats['decay_rate_distribution']['mean'] < 0.3 else 'fast'
        }
        
        # Add category-specific recommendations if available
        category_recommendations = {}
        if 'category_analysis' in distribution_results:
            for category, stats in distribution_results['category_analysis'].items():
                category_recommendations[category] = {
                    'recommended_horizon': int(stats['mean_max_acf_lag']),
                    'confidence': stats['mean_max_acf_value'],
                    'decay_rate': stats['mean_decay_rate']
                }
        
        return {
            'success': True,
            'general_recommendations': recommendations,
            'category_recommendations': category_recommendations,
            'analysis_summary': {
                'total_analyzed_tokens': distribution_results['successful_tokens'],
                'significance_threshold': significance_threshold,
                'strong_lags_found': len(strong_lags),
                'mean_optimal_lag': overall_stats['max_acf_lag_distribution']['mean'],
                'mean_decay_rate': overall_stats['decay_rate_distribution']['mean']
            }
        }
    
    def export_results(self, results: Dict, output_path: Path) -> None:
        """Export analysis results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to: {output_path}")
    
    def load_processed_tokens(self, processed_dir: Path, 
                            max_tokens_per_category: Optional[int] = None) -> Dict[str, pl.DataFrame]:
        """
        Load tokens from processed data categories.
        
        Args:
            processed_dir: Path to processed data directory
            max_tokens_per_category: Maximum tokens per category
            
        Returns:
            Dictionary mapping token names to DataFrames
        """
        token_data = {}
        
        # Load from main categories
        categories = ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes']
        
        for category in categories:
            category_path = processed_dir / category
            if not category_path.exists():
                continue
                
            parquet_files = list(category_path.glob("*.parquet"))
            
            if max_tokens_per_category is not None:
                parquet_files = parquet_files[:max_tokens_per_category]
            
            print(f"Loading {len(parquet_files)} tokens from {category}...")
            
            for file_path in tqdm(parquet_files, desc=f"Loading {category}"):
                try:
                    df = pl.read_parquet(file_path).with_columns([
                        pl.lit(category).alias('category')
                    ])
                    token_name = file_path.stem
                    token_data[token_name] = df
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        print(f"Successfully loaded {len(token_data)} tokens")
        return token_data