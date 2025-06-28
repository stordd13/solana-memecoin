"""
Data quality analysis for memecoin data using Polars
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from data_analysis.export_utils import export_parquet_files
from data_analysis.price_analysis import PriceAnalyzer

logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    def __init__(self):
        """Initialize the data quality analyzer"""
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'completeness_min': 95.0,  # Minimum acceptable completeness %
            'max_gap_minutes': 5,       # Maximum acceptable gap in minutes
            'max_extreme_changes': 5,   # Maximum acceptable extreme price changes
            'max_duplicates': 0,        # Maximum acceptable duplicates
            'min_data_points': 60,      # Minimum data points required (1 hour)
            'max_gap_for_interpolation': 10  # Maximum gap size for interpolation
        }
        # New extreme thresholds
        self.extreme_thresholds = {
            'extreme_minute_return': 100.0,  # 10,000% in one minute
            'extreme_total_return': 10000.0,  # 1,000,000% total return
            'extreme_volatility': 100.0,     # 10,000% volatility
            'extreme_range': 100.0           # 10,000% price range
        }
        
        # Outlier detection methods and thresholds (adjusted for crypto data)
        self.outlier_methods = {
            'winsorization': {'lower': 0.005, 'upper': 0.995},  # More extreme percentiles for crypto
            'z_score': {'threshold': 5.0},                      # Higher threshold for crypto volatility
            'iqr': {'multiplier': 3.0},                        # Use extreme outlier threshold
            'modified_z_score': {'threshold': 5.0}             # Higher threshold for crypto data
        }
    
    def extract_launch_context(self, df: pl.DataFrame) -> Dict:
        """Extract launch context from the first timestamp using Polars"""
        try:
            first_timestamp = df.select(pl.col('datetime').min()).item()
            if first_timestamp is None:
                return {}
            return {
                'launch_year': first_timestamp.year,
                'launch_month': first_timestamp.month,
                'launch_day_of_week': first_timestamp.weekday(),
                'launch_hour': first_timestamp.hour,
                'is_weekend': first_timestamp.weekday() in [5, 6],
                'market_era': self._categorize_market_era(first_timestamp.year)
            }
        except Exception as e:
            self.logger.error(f"Error extracting launch context: {e}")
            return {}
            
    def _categorize_market_era(self, year: int) -> str:
        """Categorize the market era based on launch year"""
        if year == 2022:
            return "2022_bear"
        elif year == 2023:
            return "2023_recovery"
        elif year == 2024:
            return "2024_bull"
        else:
            return "2025_unknown"
            
    def _detect_extreme_movements(self, df: pl.DataFrame) -> Dict:
        """
        Detect extreme price movements based on new thresholds
        
        Args:
            df: DataFrame with token data
            
        Returns:
            Dictionary with extreme movement metrics
        """
        try:
            if df.height == 0:
                return {
                    'has_extreme_minute_jump': False,
                    'max_minute_return': 0.0,
                    'has_extreme_total_return': False,
                    'total_return': 0.0,
                    'extreme_minute_count': 0,
                    'extreme_details': []
                }
            
            # Calculate minute-by-minute returns
            df = df.with_columns([
                pl.col('price').pct_change().alias('returns')
            ])
            
            # Find extreme minute returns (>10,000% in one minute)
            extreme_minute_mask = pl.col('returns') > self.extreme_thresholds['extreme_minute_return']
            extreme_minutes = df.filter(extreme_minute_mask)
            
            # Calculate total return from first to last price
            first_price = df.select(pl.col('price').first()).item()
            last_price = df.select(pl.col('price').last()).item()
            total_return = ((last_price - first_price) / first_price) * 100 if first_price and first_price > 0 else 0
            
            # Get maximum minute return
            max_minute_return = df.select(pl.col('returns').max()).item() or 0
            max_minute_return_pct = max_minute_return * 100  # Convert to percentage
            
            # Check thresholds
            has_extreme_minute_jump = max_minute_return_pct > self.extreme_thresholds['extreme_minute_return'] * 100
            has_extreme_total_return = abs(total_return) > self.extreme_thresholds['extreme_total_return'] * 100
            
            # Collect extreme movement details
            extreme_details = []
            if extreme_minutes.height > 0:
                for row in extreme_minutes.to_dicts():
                    extreme_details.append({
                        'datetime': row['datetime'],
                        'price': row['price'],
                        'return_pct': row['returns'] * 100 if row['returns'] else 0
                    })
            
            return {
                'has_extreme_minute_jump': has_extreme_minute_jump,
                'max_minute_return': max_minute_return_pct,
                'has_extreme_total_return': has_extreme_total_return,
                'total_return': total_return,
                'extreme_minute_count': extreme_minutes.height,
                'extreme_details': extreme_details[:10]  # Limit to first 10 for performance
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting extreme movements: {e}")
            return {
                'has_extreme_minute_jump': False,
                'max_minute_return': 0.0,
                'has_extreme_total_return': False,
                'total_return': 0.0,
                'extreme_minute_count': 0,
                'extreme_details': []
            }
            
    def analyze_single_file(self, df: pl.DataFrame, token: str) -> Dict:
        """
        Analyze data quality for a single token using Polars
        
        Args:
            df: DataFrame with token data
            token: Token symbol
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Basic metrics
            total_rows = df.height
            if total_rows == 0:
                return self._empty_quality_report(token)
                
            # Get unique dates and check for duplicates
            unique_dates = df.select(pl.col('datetime').n_unique()).item()
            duplicate_pct = ((total_rows - unique_dates) / total_rows) * 100 if total_rows > 0 else 0
            
            # Check for gaps in data
            df = df.sort('datetime')
            datetime_series = df.get_column('datetime')
            gaps = self._analyze_gaps(datetime_series)
            
            # Check for price anomalies
            price_anomalies = self._analyze_price_anomalies(df)
            has_extreme_jump = price_anomalies.get('has_extreme_jump', False)

            # --- NEW: Comprehensive outlier detection (using local detection for crypto data) ---
            outlier_analysis = self.comprehensive_outlier_analysis(df, price_col='price', use_local_detection=True, local_window=60)
            
            # --- NEW: Enhanced extreme movement detection ---
            extreme_movements = self._detect_extreme_movements(df)
            
            # --- Existing: Compute volatility, total return, price range ---
            pa = PriceAnalyzer()
            price_stats = pa._calculate_price_stats(df)
            volatility_metrics = pa._calculate_volatility_metrics(df)
            total_return = price_stats['total_return']
            price_range = (price_stats['max_price'] - price_stats['min_price']) / price_stats['min_price'] if price_stats['min_price'] else 0
            avg_volatility = volatility_metrics['avg_volatility']
            # Extreme if > 100x (10,000%)
            has_extreme_volatility = avg_volatility > 100
            has_extreme_return = abs(total_return) > 100
            has_extreme_range = price_range > 100
            # ---
            
            # NEW: Determine if token has extreme characteristics
            is_extreme_token = (
                extreme_movements['has_extreme_minute_jump'] or 
                extreme_movements['has_extreme_total_return'] or
                has_extreme_volatility or 
                has_extreme_return or 
                has_extreme_range
            )
            
            # Extract launch context
            launch_context = self.extract_launch_context(df)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                total_rows=total_rows,
                unique_dates=unique_dates,
                gaps=gaps,
                price_anomalies=price_anomalies,
                has_extreme_jump=has_extreme_jump
            )
            
            # Calculate how long the token has been "dead" (constant price at the end)
            death_duration_hours = self._get_death_duration(df)
            
            # A token is considered dead if its price is constant for at least 2 hours
            is_dead = death_duration_hours >= 1
            
            return {
                'token': token,
                'total_rows': total_rows,
                'unique_dates': unique_dates,
                'duplicate_pct': duplicate_pct,
                'gaps': gaps,
                'price_anomalies': price_anomalies,
                'launch_context': launch_context,
                'quality_score': quality_score,
                'is_dead': is_dead,
                'death_duration_hours': death_duration_hours,
                'has_extreme_jump': has_extreme_jump,
                'has_extreme_volatility': has_extreme_volatility,
                'has_extreme_return': has_extreme_return,
                'has_extreme_range': has_extreme_range,
                # NEW: Comprehensive outlier analysis
                'outlier_analysis': outlier_analysis,
                # NEW: Enhanced extreme movement metrics
                'extreme_movements': extreme_movements,
                'is_extreme_token': is_extreme_token,
                'max_minute_return': extreme_movements['max_minute_return'],
                'total_return_pct': extreme_movements['total_return'],
                'extreme_minute_count': extreme_movements['extreme_minute_count']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing data for {token}: {e}")
            return self._empty_quality_report(token, error=str(e))
            
    def _analyze_gaps(self, datetime_series: pl.Series) -> Dict:
        """Analyze gaps in the time series using Polars"""
        if datetime_series.len() <= 1:
            return {'total_gaps': 0, 'gap_details': [], 'max_gap': 0, 'avg_gap': 0}
            
        # Get launch time (first timestamp)
        launch_time = datetime_series.min()
        
        # Calculate time differences in seconds
        time_diff_seconds = datetime_series.diff().dt.total_seconds()
        
        # Identify gaps - anything significantly longer than 1 minute (60 seconds)
        # Allow some tolerance for slight variations (e.g., 1.5 minutes = 90 seconds)
        gaps = []
        for i, diff_seconds in enumerate(time_diff_seconds):
            if diff_seconds is not None and diff_seconds > 90:  # More than 1.5 minutes
                gap_size_minutes = diff_seconds / 60
                
                # Calculate gap starting point (minutes after launch)
                gap_start_time = datetime_series[i]  # Time when gap starts
                minutes_after_launch = (gap_start_time - launch_time).total_seconds() / 60
                
                gaps.append({
                    'position': i,
                    'size_minutes': gap_size_minutes,
                    'start_minutes_after_launch': minutes_after_launch,
                    'start_time': gap_start_time,
                    'interpolation_type': self._get_interpolation_type(gap_size_minutes)
                })
                
        # Calculate gap statistics
        if gaps:
            gap_sizes = pl.Series([g['size_minutes'] for g in gaps])
            max_gap = gap_sizes.max()
            avg_gap = gap_sizes.mean()
        else:
            max_gap = 0
            avg_gap = 0
                
        return {
            'total_gaps': len(gaps),
            'gap_details': gaps,
            'max_gap': max_gap,
            'avg_gap': avg_gap
        }
        
    def _get_interpolation_type(self, gap_size: float) -> str:
        """Determine the appropriate interpolation method for a gap"""
        if gap_size <= 1:
            return "linear"
        elif gap_size <= 5:
            return "polynomial"
        elif gap_size <= 10:
            return "forward_fill_linear"
        else:
            return "exclude"
            
    def _analyze_price_anomalies(self, df: pl.DataFrame) -> Dict:
        """Analyze price anomalies in the data using Polars"""
        # Check for zero/negative prices
        zero_prices = df.filter(pl.col('price') == 0).height
        negative_prices = df.filter(pl.col('price') < 0).height
        
        # Calculate returns and detect extreme changes
        df = df.with_columns([
            pl.col('price').pct_change().alias('returns')
        ])
        
        extreme_ups = df.filter(pl.col('returns') > 10).height  # >1000% in one minute
        extreme_downs = df.filter(pl.col('returns') < -0.9).height  # <-90% in one minute
        
        # Flag for any return > 100 (10,000%) in one minute
        has_extreme_jump = df.filter(pl.col('returns') > 100).height > 0
        
        # IQR-based outlier detection
        q1 = df.select(pl.col('returns').quantile(0.25)).item()
        q3 = df.select(pl.col('returns').quantile(0.75)).item()
        iqr = q3 - q1 if q1 is not None and q3 is not None else 0
        
        outliers = df.filter(
            (pl.col('returns') < (q1 - 10 * iqr)) | 
            (pl.col('returns') > (q3 + 10 * iqr))
        ).height if iqr > 0 else 0
        
        return {
            'zero_prices': zero_prices,
            'negative_prices': negative_prices,
            'extreme_ups': extreme_ups,
            'extreme_downs': extreme_downs,
            'iqr_outliers': outliers,
            'has_extreme_jump': has_extreme_jump
        }
        
    def _calculate_quality_score(self, total_rows: int, unique_dates: int,
                               gaps: Dict, price_anomalies: Dict, has_extreme_jump: bool = False) -> float:
        """Calculate a simplified quality score (0-100) based only on basic data integrity"""
        score = 100
        
        # Only penalize for basic data integrity issues
        if total_rows == 0:
            return 0
        
        # Penalize for missing data (duplicates)
        completeness = (unique_dates / total_rows) * 100 if total_rows > 0 else 0
        score -= (100 - completeness)
        
        # Penalize for invalid prices (zero/negative)
        invalid_prices = price_anomalies['zero_prices'] + price_anomalies['negative_prices']
        if total_rows > 0:
            invalid_price_pct = (invalid_prices / total_rows) * 100
            score -= invalid_price_pct
        
        # Penalize for missing data points (gaps) - but less severely
        if total_rows > 0:
            gap_penalty = min(gaps['total_gaps'], 10)  # Cap at 10 points
            score -= gap_penalty
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
        
    def _empty_quality_report(self, token: str, error: str = None) -> Dict:
        """Return an empty quality report"""
        report = {
            'token': token,
            'total_rows': 0,
            'unique_dates': 0,
            'duplicate_pct': 0,
            'gaps': {
                'total_gaps': 0,
                'gap_details': [],
                'max_gap': 0,
                'avg_gap': 0
            },
            'price_anomalies': {
                'zero_prices': 0,
                'negative_prices': 0,
                'extreme_ups': 0,
                'extreme_downs': 0,
                'iqr_outliers': 0
            },
            'launch_context': {},
            'quality_score': 0,
            'is_dead': False,
            'death_duration_hours': 0,
            'has_extreme_jump': False,
            'has_extreme_volatility': False,
            'has_extreme_return': False,
            'has_extreme_range': False,
            # NEW: Comprehensive outlier analysis
            'outlier_analysis': {
                'summary': {'error': 'No data available'},
                'processed_dataframe': None,
                'status': 'error'
            },
            # NEW: Enhanced extreme movement metrics
            'extreme_movements': {
                'has_extreme_minute_jump': False,
                'max_minute_return': 0.0,
                'has_extreme_total_return': False,
                'total_return': 0.0,
                'extreme_minute_count': 0,
                'extreme_details': []
            },
            'is_extreme_token': False,
            'max_minute_return': 0.0,
            'total_return_pct': 0.0,
            'extreme_minute_count': 0
        }
        if error:
            report['error'] = error
        return report
    
    def analyze_multiple_files(self, parquet_files: List[Path], 
                              limit: int = None, 
                              progress_callback=None) -> pl.DataFrame:
        """Analyze data quality for multiple files with progress tracking"""
        quality_reports = []
        
        files_to_analyze = parquet_files[:limit] if limit else parquet_files
        total_files = len(files_to_analyze)
        
        for i, pf in enumerate(files_to_analyze):
            try:
                # Use read_parquet instead of scan_parquet to get DataFrame instead of LazyFrame
                df = pl.read_parquet(pf)
                token_name = pf.name.split('_')[0]
                report = self.analyze_single_file(df, token_name)
                quality_reports.append(report)
                
                if progress_callback:
                    progress_callback((i + 1) / total_files)
                    
            except Exception as e:
                logger.error(f"Error processing {pf.name}: {e}")
                token_name = pf.name.split('_')[0]
                quality_reports.append(self._empty_quality_report(token_name, error=str(e)))
                continue
        
        return pl.DataFrame(quality_reports)
    
    def get_quality_summary(self, quality_df: pl.DataFrame) -> Dict:
        """Get enhanced summary statistics from quality analysis"""
        if quality_df.height == 0:
            return self._empty_summary()
        
        # Filter out error rows for statistics
        valid_df = quality_df.filter(pl.col('quality_score') > 0)
        
        return {
            'total_tokens': quality_df.height,
            'tokens_analyzed': valid_df.height,
            'tokens_with_errors': quality_df.height - valid_df.height,
            'avg_completeness': valid_df['completeness_pct'].mean() if valid_df.height > 0 else 0,
            'tokens_with_gaps': valid_df.filter(pl.col('time_gaps') > 0).height if valid_df.height > 0 else 0,
            'tokens_with_duplicates': valid_df.filter(pl.col('duplicate_times') > 0).height if valid_df.height > 0 else 0,
            'tokens_with_extreme_changes': valid_df.filter(pl.col('extreme_changes') > 0).height if valid_df.height > 0 else 0,
            'tokens_with_zero_prices': valid_df.filter(pl.col('zero_prices') > 0).height if valid_df.height > 0 else 0,
            'avg_hours_covered': valid_df['hours_covered'].mean() if valid_df.height > 0 else 0,
            'avg_quality_score': valid_df['quality_score'].mean() if valid_df.height > 0 else 0,
            'excellent_quality_tokens': valid_df.filter(pl.col('quality_rating') == 'Excellent').height if valid_df.height > 0 else 0,
            'good_quality_tokens': valid_df.filter(pl.col('quality_rating') == 'Good').height if valid_df.height > 0 else 0,
            'poor_quality_tokens': valid_df.filter(pl.col('quality_rating').is_in(['Poor', 'Very Poor'])).height if valid_df.height > 0 else 0,
            'perfect_quality_tokens': valid_df.filter(
                (pl.col('time_gaps') == 0) & 
                (pl.col('duplicate_times') == 0) & 
                (pl.col('extreme_changes') == 0) & 
                (pl.col('zero_prices') == 0)
            ).height if valid_df.height > 0 else 0
        }
    
    def _empty_summary(self) -> Dict:
        """Return empty summary statistics"""
        return {
            'total_tokens': 0,
            'tokens_analyzed': 0,
            'tokens_with_errors': 0,
            'avg_completeness': 0,
            'tokens_with_gaps': 0,
            'tokens_with_duplicates': 0,
            'tokens_with_extreme_changes': 0,
            'tokens_with_zero_prices': 0,
            'avg_hours_covered': 0,
            'avg_quality_score': 0,
            'excellent_quality_tokens': 0,
            'good_quality_tokens': 0,
            'poor_quality_tokens': 0,
            'perfect_quality_tokens': 0
        }
    
    def identify_quality_issues(self, quality_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Identify tokens with quality issues"""
        return {
            'time_gaps': quality_df.filter(pl.col('time_gaps') > 0).select(['token', 'time_gaps', 'max_gap_minutes']),
            'extreme_changes': quality_df.filter(pl.col('extreme_changes') > 0).select(['token', 'extreme_changes'])
        }
    
    def identify_extreme_tokens(self, quality_reports: Dict) -> Dict[str, Dict]:
        """
        Identify tokens with extreme price movements (third priority after gaps and normal)
        Priority: gaps > normal > extremes > dead
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of extreme tokens with their metrics
        """
        extreme_tokens = {}
        
        for token, report in quality_reports.items():
            # HIERARCHICAL EXCLUSION: Gaps and normal behavior tokens take priority over extreme classification
            # UPDATED: Check for significant gaps (many gaps OR large gaps)
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            is_extreme = report.get('is_extreme_token', False)
            is_dead = report.get('is_dead', False)
            
            # Check if token qualifies as normal behavior (second priority)
            is_normal_behavior = not (has_significant_gaps or is_extreme or is_dead)
            
            # Only classify as extreme if NOT gaps and NOT normal behavior
            if is_extreme and not (has_significant_gaps or is_normal_behavior):
                extreme_tokens[token] = {
                    'token': token,
                    'max_minute_return': report.get('max_minute_return', 0),
                    'total_return_pct': report.get('total_return_pct', 0),
                    'extreme_minute_count': report.get('extreme_minute_count', 0),
                    'has_extreme_minute_jump': report.get('extreme_movements', {}).get('has_extreme_minute_jump', False),
                    'has_extreme_total_return': report.get('extreme_movements', {}).get('has_extreme_total_return', False),
                    'has_extreme_volatility': report.get('has_extreme_volatility', False),
                    'has_extreme_return': report.get('has_extreme_return', False),
                    'has_extreme_range': report.get('has_extreme_range', False),
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0),
                    'is_also_dead': report.get('is_dead', False),  # Track if also dead
                    'note': 'Third priority: gaps > normal > extremes > dead (VALUABLE patterns)'
                }
        
        return extreme_tokens
    
    def export_extreme_tokens(self, quality_reports: Dict) -> List[str]:
        """
        Export extreme tokens to parquet files in data/processed/tokens_with_extremes/
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            List of exported token names
        """
        extreme_tokens = self.identify_extreme_tokens(quality_reports)
        
        if not extreme_tokens:
            raise ValueError("No extreme tokens found to export.")
        
        token_list = list(extreme_tokens.keys())
        
        try:
            exported = export_parquet_files(token_list, "Tokens with Extremes")
            self.logger.info(f"Exported {len(exported)} extreme tokens to data/processed/tokens_with_extremes/")
            return exported
        except Exception as e:
            self.logger.error(f"Error exporting extreme tokens: {e}")
            raise
    
    def identify_normal_behavior_tokens(self, quality_reports: Dict) -> Dict[str, Dict]:
        """
        Identify tokens with normal behavior (second priority after gaps)
        Priority: gaps > normal > extremes > dead
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of normal behavior tokens with their metrics
        """
        normal_tokens = {}
        
        for token, report in quality_reports.items():
            # Check characteristics - UPDATED LOGIC
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            is_extreme = report.get('is_extreme_token', False)
            is_dead = report.get('is_dead', False)  
            
            # SECOND PRIORITY: Normal behavior tokens (after gaps are excluded)
            # A token is normal if it has no gaps and none of the other problematic characteristics
            if not (has_significant_gaps or is_extreme or is_dead):
                normal_tokens[token] = {
                    'token': token,
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0),
                    'gaps': report.get('gaps', {}).get('total_gaps', 0),
                    'is_extreme': is_extreme,
                    'is_dead': is_dead,
                    'has_significant_gaps': has_significant_gaps,
                    'note': 'Second priority - normal behavior tokens (BEST for training)'
                }
        
        return normal_tokens
    
    def export_normal_behavior_tokens(self, quality_reports: Dict) -> List[str]:
        """
        Export normal behavior tokens to parquet files in data/processed/normal_behavior_tokens/
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            List of exported token names
        """
        normal_tokens = self.identify_normal_behavior_tokens(quality_reports)
        
        if not normal_tokens:
            raise ValueError("No normal behavior tokens found to export.")
        
        token_list = list(normal_tokens.keys())
        
        try:
            exported = export_parquet_files(token_list, "Normal Behavior Tokens")
            self.logger.info(f"Exported {len(exported)} normal behavior tokens to data/processed/normal_behavior_tokens/")
            return exported
        except Exception as e:
            self.logger.error(f"Error exporting normal behavior tokens: {e}")
            raise
    
    def identify_dead_tokens(self, quality_reports: Dict) -> Dict[str, Dict]:
        """
        Identify dead tokens (lowest priority after gaps, normal, and extreme)
        Priority: gaps > normal > extremes > dead
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of dead tokens with their metrics
        """
        dead_tokens = {}
        
        for token, report in quality_reports.items():
            # Check characteristics - UPDATED LOGIC
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            is_extreme = report.get('is_extreme_token', False)
            is_dead = report.get('is_dead', False)
            
            # Check if token qualifies as normal behavior (second priority)
            is_normal_behavior = not (has_significant_gaps or is_extreme or is_dead)
            
            # Only classify as dead if NOT gaps, normal, or extreme (lowest priority)
            if is_dead and not (has_significant_gaps or is_normal_behavior or is_extreme):
                dead_tokens[token] = {
                    'token': token,
                    'death_duration_hours': report.get('death_duration_hours', 0),
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0),
                    'extreme_movements': report.get('extreme_movements', {}),
                    'note': 'Lowest priority: gaps > normal > extremes > dead (COMPLETION data)'
                }
        
        return dead_tokens

    def export_dead_tokens(self, quality_reports: Dict) -> List[str]:
        """
        Export dead tokens to parquet files in data/processed/dead_tokens/
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            List of exported token names
        """
        dead_tokens = self.identify_dead_tokens(quality_reports)
        
        if not dead_tokens:
            raise ValueError("No dead tokens found to export.")
        
        token_list = list(dead_tokens.keys())
        
        try:
            exported = export_parquet_files(token_list, "Dead Tokens")
            self.logger.info(f"Exported {len(exported)} dead tokens to data/processed/dead_tokens/")
            return exported
        except Exception as e:
            self.logger.error(f"Error exporting dead tokens: {e}")
            raise
    
    def recommend_tokens_for_analysis(self, quality_df: pl.DataFrame, 
                                    min_hours: float = 1.0,
                                    min_quality_score: float = 80.0) -> pl.DataFrame:
        """Recommend tokens for analysis based on quality criteria"""
        return quality_df.filter(
            (pl.col('hours_covered') >= min_hours) &
            (pl.col('quality_score') >= min_quality_score)
        ).select(['token', 'quality_score', 'hours_covered']).sort('quality_score', descending=True)
    
    def display_quality_metrics(self, quality_df: pl.DataFrame):
        """
        Display quality metrics in Streamlit
        
        Args:
            quality_df: DataFrame with quality metrics
        """
        if quality_df.height == 0:
            st.warning("No quality metrics available to display.")
            return
            
        # Overall summary
        st.subheader("Overall Data Quality Summary")
        
        # Use Polars for processing
        # Display average quality score
        avg_score = quality_df['quality_score'].mean()
        st.metric("Average Quality Score", f"{avg_score:.1f}/100")
        
        # Quality score distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=quality_df['quality_score'].to_list(),
            nbinsx=20,
            name="Quality Score Distribution"
        ))
        fig.update_layout(
            title="Distribution of Data Quality Scores",
            xaxis_title="Quality Score",
            yaxis_title="Number of Tokens",
            showlegend=False
        )
        st.plotly_chart(fig)
        
        # Detailed metrics
        st.subheader("Detailed Quality Metrics")
        
        # Sort by quality score using Polars
        quality_sorted = quality_df.sort('quality_score', descending=True)
        
        # Display metrics for each token using Polars iteration
        for row in quality_sorted.iter_rows(named=True):
            with st.expander(f"{row['token']} (Score: {row['quality_score']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Rows", row['total_rows'])
                    st.metric("Unique Dates", row['unique_dates'])
                    st.metric("Duplicate Percentage", f"{row['duplicate_pct']:.1f}%")
                
                with col2:
                    if row['avg_gap'] is not None:
                        st.metric("Average Time Gap", str(row['avg_gap']))
                    if row['max_gap'] is not None:
                        st.metric("Maximum Time Gap", str(row['max_gap']))
                    
                    # Display missing values
                    if 'missing_values' in row and isinstance(row['missing_values'], dict):
                        st.write("Missing Values (%):")
                        for col, pct in row['missing_values'].items():
                            st.write(f"- {col}: {pct:.1f}%")
                
                # Recommendations
                st.write("Recommendations:")
                if row['quality_score'] < 50:
                    st.error("⚠️ Poor data quality. Consider collecting more data or fixing data issues.")
                elif row['quality_score'] < 80:
                    st.warning("⚠️ Moderate data quality. Some improvements needed.")
                else:
                    st.success("✅ Good data quality.")
                    
                if row['duplicate_pct'] > 5:
                    st.warning("High percentage of duplicates. Consider cleaning the data.")
                if row['max_gap'] is not None and isinstance(row['max_gap'], (timedelta, np.timedelta64)) and row['max_gap'] > timedelta(hours=24):
                    st.warning("Large gaps in data. Consider filling missing values.")

    def display_quality_summary(self, quality_reports: Dict):
        """Display a summary of data quality analysis"""
        try:
            # Convert reports to DataFrame
            quality_df = pl.DataFrame([
                {
                    'token': report['token'],
                    'total_rows': report['total_rows'],
                    'unique_dates': report['unique_dates'],
                    'duplicate_pct': report['duplicate_pct'],
                    'total_gaps': report['gaps']['total_gaps'],
                    'max_gap': report['gaps']['max_gap'],
                    'avg_gap': report['gaps']['avg_gap'],
                    'zero_prices': report['price_anomalies']['zero_prices'],
                    'negative_prices': report['price_anomalies']['negative_prices'],
                    'extreme_ups': report['price_anomalies']['extreme_ups'],
                    'extreme_downs': report['price_anomalies']['extreme_downs'],
                    'iqr_outliers': report['price_anomalies']['iqr_outliers'],
                    'quality_score': report['quality_score'],
                    'is_dead': report['is_dead'],
                    'has_extreme_jump': report['has_extreme_jump'],
                    'has_extreme_volatility': report['has_extreme_volatility'],
                    'has_extreme_return': report['has_extreme_return'],
                    'has_extreme_range': report['has_extreme_range'],
                    'is_extreme_token': report.get('is_extreme_token', False),
                    'max_minute_return': report.get('max_minute_return', 0),
                    'total_return_pct': report.get('total_return_pct', 0),
                    'extreme_minute_count': report.get('extreme_minute_count', 0)
                }
                for report in quality_reports.values()
            ])
            
            # Overall statistics
            st.subheader("Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tokens", len(quality_reports))
                st.metric("Average Quality Score", f"{quality_df['quality_score'].mean():.1f}")
            
            with col2:
                # Missing data statistics
                tokens_with_gaps = sum(1 for report in quality_reports.values() 
                                     if report['gaps']['total_gaps'] > 0)
                tokens_with_missing_data = sum(1 for report in quality_reports.values() 
                                             if report['gaps']['total_gaps'] > 0 or 
                                             report['duplicate_pct'] > 0 or
                                             report['total_rows'] != report['unique_dates'])
                st.metric("Tokens with Missing Data", tokens_with_missing_data)
                st.metric("Tokens with Gaps", tokens_with_gaps)
            
            with col3:
                # Price anomaly statistics
                total_zero = sum(report['price_anomalies']['zero_prices'] 
                               for report in quality_reports.values())
                total_negative = sum(report['price_anomalies']['negative_prices'] 
                                   for report in quality_reports.values())
                st.metric("Total Zero Prices", total_zero)
                st.metric("Total Negative Prices", total_negative)
            
            with col4:
                tokens_with_extremes = sum(1 for report in quality_reports.values() if report.get('is_extreme_token', False))
                dead_tokens_count = sum(1 for report in quality_reports.values() if report.get('is_dead', False))
                st.metric("Tokens with Extremes", tokens_with_extremes)
                st.metric("Dead Tokens", dead_tokens_count)
            
            # Outlier Detection Summary
            st.subheader("Outlier Detection Summary (Local Detection)")
            st.info("🔍 **Local Outlier Detection**: Using 60-minute rolling windows to detect outliers relative to local patterns (better for cryptocurrency data)")
            self._display_outlier_summary(quality_reports)
            
            # Quality score distribution
            st.subheader("Quality Score Distribution")
            fig = go.Figure(data=[go.Histogram(x=quality_df['quality_score'], 
                                             nbinsx=20,
                                             name="Quality Score")])
            fig.update_layout(title="Distribution of Quality Scores",
                            xaxis_title="Quality Score",
                            yaxis_title="Number of Tokens")
            st.plotly_chart(fig)
            
            # Gap analysis
            st.subheader("Gap Analysis")
            
            # Calculate gap statistics from tokens that have gaps
            tokens_with_gaps_data = [report for report in quality_reports.values() if report['gaps']['total_gaps'] > 0]
            if tokens_with_gaps_data:
                all_gap_sizes = []
                for report in tokens_with_gaps_data:
                    all_gap_sizes.extend([gap['size_minutes'] for gap in report['gaps']['gap_details']])
                avg_gap_size = sum(all_gap_sizes) / len(all_gap_sizes) if all_gap_sizes else 0
                max_gap_all = max(all_gap_sizes) if all_gap_sizes else 0
            else:
                avg_gap_size = 0
                max_gap_all = 0
            
            gap_stats = {
                'tokens_with_gaps': tokens_with_gaps,
                'max_gap_all': max_gap_all,
                'avg_gap_size': avg_gap_size,
                'total_gaps': quality_df['total_gaps'].sum()
            }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tokens with Gaps", gap_stats['tokens_with_gaps'])
            with col2:
                st.metric("Maximum Gap (minutes)", f"{gap_stats['max_gap_all']:.1f}")
            with col3:
                st.metric("Average Gap Size (minutes)", f"{gap_stats['avg_gap_size']:.1f}")
            with col4:
                st.metric("Total Number of Gaps", gap_stats['total_gaps'])
            
            # Price anomaly analysis
            st.subheader("Price Anomaly Analysis")
            anomaly_stats = {
                'total_zero': total_zero,
                'total_negative': total_negative,
                'total_extreme_ups': quality_df['extreme_ups'].sum(),
                'total_extreme_downs': quality_df['extreme_downs'].sum()
            }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Zero Prices", anomaly_stats['total_zero'])
            with col2:
                st.metric("Negative Prices", anomaly_stats['total_negative'])
            with col3:
                st.metric("Extreme Ups", anomaly_stats['total_extreme_ups'])
            with col4:
                st.metric("Extreme Downs", anomaly_stats['total_extreme_downs'])
            
            # Group tokens by quality score
            st.subheader("Token Quality Groups")
            
            # Define quality groups
            high_quality = {token: report for token, report in quality_reports.items() 
                          if report['quality_score'] >= 80}
            medium_quality = {token: report for token, report in quality_reports.items() 
                            if 60 <= report['quality_score'] < 80}
            low_quality = {token: report for token, report in quality_reports.items() 
                         if report['quality_score'] < 60}
            
            # Display group statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Quality Tokens (≥80)", len(high_quality))
            with col2:
                st.metric("Medium Quality Tokens (60-79)", len(medium_quality))
            with col3:
                st.metric("Low Quality Tokens (<60)", len(low_quality))
            
            # Normal behavior tokens section
            normal_tokens = self.identify_normal_behavior_tokens(quality_reports)
            if normal_tokens:
                st.subheader("Normal Behavior Tokens")
                st.write(f"Found {len(normal_tokens)} tokens with normal behavior (not dead, not extreme, minimal gaps)")
                
                # Create a table with normal token metrics
                normal_table_data = []
                for token, metrics in normal_tokens.items():
                    normal_table_data.append({
                        'Token': token,
                        'Quality Score': f"{metrics['quality_score']:.1f}",
                        'Total Rows': metrics['total_rows'],
                        'Gaps': metrics['gaps']
                    })
                
                if normal_table_data:
                    normal_df = pl.DataFrame(normal_table_data)
                    # Add pagination for normal tokens
                    items_per_page = 10
                    total_pages = (len(normal_df) + items_per_page - 1) // items_per_page
                    page = st.selectbox("Normal Tokens Page", range(1, total_pages + 1), index=0, key="normal_tokens_page")
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(normal_df))
                    st.dataframe(
                        normal_df.slice(start_idx, end_idx - start_idx),
                        height=400,
                        use_container_width=True
                    )
                    st.write(f"Showing {start_idx + 1}-{end_idx} of {len(normal_df)} normal behavior tokens")
                    
                    # Add download button for normal tokens list
                    if st.button("Download Normal Behavior Tokens List", key="download_normal_tokens_csv"):
                        normal_tokens_list = list(normal_tokens.keys())
                        st.download_button(
                            label="Download as CSV",
                            data="\n".join(normal_tokens_list),
                            file_name="normal_behavior_tokens.csv",
                            mime="text/csv",
                            key="download_normal_tokens_csv_file"
                        )
                    
                    # Add export button for normal token parquet files
                    if st.button("Export Normal Behavior Token Parquet Files to processed/", key="export_normal_behavior_parquet_dq"):
                        try:
                            exported = self.export_normal_behavior_tokens(quality_reports)
                            st.success(f'Exported {len(exported)} normal behavior token parquet files to data/processed/normal_behavior_tokens/')
                        except Exception as e:
                            st.error(f'Export failed: {e}')
            

            
        except Exception as e:
            self.logger.error(f"Error displaying quality summary: {e}")
            st.error("Error displaying quality summary.")

    def _get_death_duration(self, df: pl.DataFrame) -> int:
        """
        Calculates the number of consecutive hours at the end of the series
        for which the price was constant.

        Returns:
            int: The number of hours rounded down.
        """
        # Sort descending to start from the latest timestamp
        df_sorted = df.sort('datetime', descending=True)

        prices = df_sorted.get_column('price').to_list()
        if not prices:
            return 0

        last_price = prices[0]
        constant_minutes = 0
        for p in prices:
            if p == last_price:
                constant_minutes += 1
            else:
                break

        # Convert minutes to whole hours
        constant_hours = constant_minutes // 60
        return int(constant_hours)

    def debug_gap_detection(self, quality_reports: Dict, min_gap_size: float = 10.0) -> Dict:
        """
        Debug function to help identify tokens with large gaps that might not be getting detected
        
        Args:
            quality_reports: Dictionary of quality reports
            min_gap_size: Minimum gap size in minutes to report
            
        Returns:
            Dictionary of debug information
        """
        print(f"\n🔍 DEBUG: Gap Detection Analysis (gaps >= {min_gap_size} minutes)")
        print("=" * 70)
        
        tokens_with_large_gaps = {}
        gap_size_distribution = []
        
        for token, report in quality_reports.items():
            gaps_info = report.get('gaps', {})
            total_gaps = gaps_info.get('total_gaps', 0)
            max_gap = gaps_info.get('max_gap', 0)
            gap_details = gaps_info.get('gap_details', [])
            
            # Find gaps larger than threshold
            large_gaps = [g for g in gap_details if g.get('size_minutes', 0) >= min_gap_size]
            
            if large_gaps:
                tokens_with_large_gaps[token] = {
                    'total_gaps': total_gaps,
                    'large_gaps_count': len(large_gaps),
                    'max_gap_minutes': max_gap,
                    'large_gap_sizes': [g.get('size_minutes', 0) for g in large_gaps],
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0)
                }
                
                # Collect all gap sizes for distribution analysis
                gap_size_distribution.extend([g.get('size_minutes', 0) for g in gap_details])
        
        # Print summary
        print(f"📊 SUMMARY:")
        print(f"  Tokens with gaps >= {min_gap_size} minutes: {len(tokens_with_large_gaps)}")
        print(f"  NEW threshold for 'significant gaps': >5 total gaps OR >30 minute gap")
        print(f"  Tokens meeting old threshold (>5 gaps only): {len([t for t, r in quality_reports.items() if r.get('gaps', {}).get('total_gaps', 0) > 5])}")
        print(f"  Tokens meeting NEW threshold (>5 gaps OR >30min gap): {len([t for t, r in quality_reports.items() if (r.get('gaps', {}).get('total_gaps', 0) > 5) or (r.get('gaps', {}).get('max_gap', 0) > 30)])}")
        
        if tokens_with_large_gaps:
            print(f"\n📈 TOP TOKENS WITH LARGE GAPS:")
            sorted_tokens = sorted(tokens_with_large_gaps.items(), 
                                 key=lambda x: x[1]['max_gap_minutes'], reverse=True)
            
            for i, (token, info) in enumerate(sorted_tokens[:10]):  # Show top 10
                print(f"  {i+1:2}. {token[:20]:20} | Max gap: {info['max_gap_minutes']:6.1f}m | Total gaps: {info['total_gaps']:2} | Large gaps: {info['large_gaps_count']}")
        
        if gap_size_distribution:
            import numpy as np
            gaps_array = np.array(gap_size_distribution)
            print(f"\n📏 GAP SIZE DISTRIBUTION:")
            print(f"  Total gaps found: {len(gaps_array)}")
            print(f"  Min gap size: {gaps_array.min():.1f} minutes")
            print(f"  Max gap size: {gaps_array.max():.1f} minutes")
            print(f"  Mean gap size: {gaps_array.mean():.1f} minutes")
            print(f"  Median gap size: {np.median(gaps_array):.1f} minutes")
            print(f"  Gaps >= 60 minutes: {len(gaps_array[gaps_array >= 60])}")
            print(f"  Gaps >= 30 minutes: {len(gaps_array[gaps_array >= 30])}")
            print(f"  Gaps >= 10 minutes: {len(gaps_array[gaps_array >= 10])}")
        
        return {
            'tokens_with_large_gaps': tokens_with_large_gaps,
            'gap_size_distribution': gap_size_distribution,
            'current_threshold_count': len([t for t, r in quality_reports.items() if r.get('gaps', {}).get('total_gaps', 0) > 5])
        }

    def identify_tokens_with_gaps(self, quality_reports: Dict, debug: bool = False) -> Dict[str, Dict]:
        """
        Identify tokens with significant gaps (highest priority - exclude from training)
        Priority: gaps > normal > extremes > dead
        
        Uses EITHER condition:
        - More than 5 total gaps, OR
        - Any single gap larger than 30 minutes
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of tokens with gaps and their metrics
        """
        gap_tokens = {}
        
        for token, report in quality_reports.items():
            # Check characteristics - UPDATED LOGIC
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            
            # NEW: Significant gaps = many small gaps OR one large gap
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            # Classify as gap token if it has significant gaps (highest priority)
            if has_significant_gaps:
                gap_tokens[token] = {
                    'token': token,
                    'total_gaps': total_gaps,
                    'max_gap': max_gap,
                    'avg_gap': report.get('gaps', {}).get('avg_gap', 0),
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0),
                    'gap_type': 'many_gaps' if has_many_gaps else 'large_gap',
                    'note': 'Highest priority: gaps > normal > extremes > dead (EXCLUDE from training)'
                }
        
        return gap_tokens

    def export_tokens_with_gaps(self, quality_reports: Dict) -> List[str]:
        """
        Export tokens with gaps to parquet files in data/processed/tokens_with_gaps/
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            List of exported token names
        """
        gap_tokens = self.identify_tokens_with_gaps(quality_reports)
        
        if not gap_tokens:
            raise ValueError("No tokens with gaps found to export.")
        
        token_list = list(gap_tokens.keys())
        
        try:
            exported = export_parquet_files(token_list, "Tokens with Gaps")
            self.logger.info(f"Exported {len(exported)} tokens with gaps to data/processed/tokens_with_gaps/")
            return exported
        except Exception as e:
            self.logger.error(f"Error exporting tokens with gaps: {e}")
            raise

    def export_all_categories_mutually_exclusive(self, quality_reports: Dict) -> Dict[str, List[str]]:
        """
        Export ALL categories with strict mutual exclusivity enforcement.
        Each token appears in EXACTLY ONE category based on hierarchy: gaps > normal > extremes > dead
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            Dictionary with category names as keys and token lists as values
        """
        print("\n🔄 EXPORTING MUTUALLY EXCLUSIVE CATEGORIES")
        print("=" * 60)
        print("📊 Hierarchy: gaps > normal > extremes > dead")
        
        # Step 1: Categorize all tokens with strict hierarchy
        categorized_tokens = {
            'normal_behavior_tokens': [],
            'tokens_with_extremes': [],
            'dead_tokens': [],
            'tokens_with_gaps': []
        }
        
        overlap_stats = {
            'normal_also_extreme': 0,
            'normal_also_dead': 0,
            'normal_also_gaps': 0,
            'extreme_also_dead': 0,
            'extreme_also_gaps': 0,
            'dead_also_gaps': 0,
            'total_overlaps_resolved': 0
        }
        
        for token, report in quality_reports.items():
            # Check all characteristics
            is_extreme = report.get('is_extreme_token', False)
            is_dead = report.get('is_dead', False)
            
            # UPDATED: Check for significant gaps (many gaps OR large gaps)
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            # Check if token qualifies as normal behavior
            is_normal_behavior = not (is_extreme or is_dead or has_significant_gaps)
            
            # Track overlaps for statistics
            if is_normal_behavior and is_extreme:
                overlap_stats['normal_also_extreme'] += 1
            if is_normal_behavior and is_dead:
                overlap_stats['normal_also_dead'] += 1
            if is_normal_behavior and has_significant_gaps:
                overlap_stats['normal_also_gaps'] += 1
            if is_extreme and is_dead:
                overlap_stats['extreme_also_dead'] += 1
            if is_extreme and has_significant_gaps:
                overlap_stats['extreme_also_gaps'] += 1
            if is_dead and has_significant_gaps:
                overlap_stats['dead_also_gaps'] += 1
            
            # STRICT HIERARCHICAL ASSIGNMENT (each token goes to EXACTLY ONE category)
            # Priority: gaps > normal > extremes > dead
            if has_significant_gaps:
                categorized_tokens['tokens_with_gaps'].append(token)
                if is_normal_behavior or is_extreme or is_dead:
                    overlap_stats['total_overlaps_resolved'] += 1
            elif is_normal_behavior:
                categorized_tokens['normal_behavior_tokens'].append(token)
                if is_extreme or is_dead:
                    overlap_stats['total_overlaps_resolved'] += 1
            elif is_extreme:
                categorized_tokens['tokens_with_extremes'].append(token)
                if is_dead:
                    overlap_stats['total_overlaps_resolved'] += 1
            elif is_dead:
                categorized_tokens['dead_tokens'].append(token)
        
        # Step 2: Display categorization summary
        print(f"\n📈 CATEGORIZATION RESULTS:")
        total_tokens = len(quality_reports)
        for category, tokens in categorized_tokens.items():
            pct = (len(tokens) / total_tokens) * 100 if total_tokens > 0 else 0
            print(f"  {category:25}: {len(tokens):,} tokens ({pct:.1f}%)")
        
        print(f"\n🔍 OVERLAP RESOLUTION:")
        print(f"  Normal tokens that were also extreme:   {overlap_stats['normal_also_extreme']:,}")
        print(f"  Normal tokens that were also dead:      {overlap_stats['normal_also_dead']:,}")
        print(f"  Normal tokens that also had gaps:       {overlap_stats['normal_also_gaps']:,}")
        print(f"  Extreme tokens that were also dead:     {overlap_stats['extreme_also_dead']:,}")
        print(f"  Extreme tokens that also had gaps:      {overlap_stats['extreme_also_gaps']:,}")
        print(f"  Dead tokens that also had gaps:         {overlap_stats['dead_also_gaps']:,}")
        print(f"  Total overlaps resolved:                {overlap_stats['total_overlaps_resolved']:,}")
        
        # Step 3: Export each category
        exported_results = {}
        
        for category, tokens in categorized_tokens.items():
            if tokens:
                try:
                    # Map category names to export group names
                    group_name_map = {
                        'normal_behavior_tokens': 'Normal Behavior Tokens',
                        'tokens_with_extremes': 'Tokens with Extremes',
                        'dead_tokens': 'Dead Tokens',
                        'tokens_with_gaps': 'Tokens with Gaps'
                    }
                    
                    group_name = group_name_map[category]
                    exported = export_parquet_files(tokens, group_name)
                    exported_results[category] = exported
                    
                    print(f"✅ Exported {len(exported):,} tokens to {category}/")
                    
                except Exception as e:
                    print(f"❌ Error exporting {category}: {e}")
                    exported_results[category] = []
            else:
                print(f"⚠️  No tokens found for {category}")
                exported_results[category] = []
        
        print(f"\n✅ EXPORT COMPLETE - All categories are now mutually exclusive!")
        print(f"   Total tokens processed: {total_tokens:,}")
        print(f"   Total tokens exported: {sum(len(tokens) for tokens in exported_results.values()):,}")
        
        return exported_results

    def investigate_tokens_with_gaps(self, quality_reports: Dict) -> Dict:
        """
        Comprehensive investigation of tokens with gaps to help decide whether to keep or remove them
        
        Returns:
            Dictionary with detailed analysis of each token with gaps
        """
        print(f"\n🔍 INVESTIGATING TOKENS WITH GAPS")
        print("=" * 70)
        
        # Get tokens with gaps
        total_gaps = quality_reports.get('gaps', {}).get('total_gaps', 0)
        max_gap = quality_reports.get('gaps', {}).get('max_gap', 0)
        has_many_gaps = total_gaps > 5  # More than 5 gaps
        has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
        
        tokens_with_gaps = {}
        investigation_results = {
            'tokens_analyzed': 0,
            'recommendations': {
                'keep_and_clean': [],
                'remove_completely': [],
                'needs_manual_review': []
            },
            'gap_analysis': {
                'small_gaps_only': [],      # <5 min gaps, easy to fill
                'medium_gaps': [],          # 5-30 min gaps, fillable
                'large_gaps': [],           # >30 min gaps, problematic
                'excessive_gaps': []        # >10 gaps total, very problematic
            },
            'detailed_analysis': {}
        }
        
        for token, report in quality_reports.items():
            gaps_info = report.get('gaps', {})
            total_gaps = gaps_info.get('total_gaps', 0)
            max_gap = gaps_info.get('max_gap', 0)
            
            # Check if token has significant gaps
            has_many_gaps = total_gaps > 5
            has_large_gap = max_gap > 30
            
            if has_many_gaps or has_large_gap:
                investigation_results['tokens_analyzed'] += 1
                
                # Detailed analysis of this token
                token_analysis = self._analyze_gap_token_detailed(token, report)
                investigation_results['detailed_analysis'][token] = token_analysis
                
                # Categorize by gap severity
                if total_gaps > 10:
                    investigation_results['gap_analysis']['excessive_gaps'].append(token)
                elif max_gap > 30:
                    investigation_results['gap_analysis']['large_gaps'].append(token)
                elif max_gap > 5:
                    investigation_results['gap_analysis']['medium_gaps'].append(token)
                else:
                    investigation_results['gap_analysis']['small_gaps_only'].append(token)
                
                # Make recommendation
                recommendation = self._recommend_gap_token_action(token_analysis)
                investigation_results['recommendations'][recommendation].append(token)
        
        # Print summary
        self._print_gap_investigation_summary(investigation_results)
        
        return investigation_results
    
    def _analyze_gap_token_detailed(self, token: str, report: Dict) -> Dict:
        """Detailed analysis of a single token with gaps"""
        gaps_info = report.get('gaps', {})
        
        analysis = {
            'token': token,
            'total_gaps': gaps_info.get('total_gaps', 0),
            'max_gap_minutes': gaps_info.get('max_gap', 0),
            'avg_gap_minutes': gaps_info.get('avg_gap', 0),
            'total_data_points': report.get('total_rows', 0),
            'quality_score': report.get('quality_score', 0),
            'completeness_pct': report.get('completeness_pct', 0),
            'time_span_hours': report.get('time_span_hours', 0),
            'data_density': 0,  # data points per hour
            'gap_severity': 'unknown',
            'data_quality': 'unknown',
            'trading_activity': 'unknown',
            'recommendation_factors': []
        }
        
        # Calculate data density
        if analysis['time_span_hours'] > 0:
            analysis['data_density'] = analysis['total_data_points'] / analysis['time_span_hours']
        
        # Assess gap severity
        if analysis['total_gaps'] > 10:
            analysis['gap_severity'] = 'excessive'
            analysis['recommendation_factors'].append('Too many gaps (>10)')
        elif analysis['max_gap_minutes'] > 60:
            analysis['gap_severity'] = 'severe'
            analysis['recommendation_factors'].append('Very large gaps (>1 hour)')
        elif analysis['max_gap_minutes'] > 30:
            analysis['gap_severity'] = 'moderate'
            analysis['recommendation_factors'].append('Large gaps (>30 min)')
        else:
            analysis['gap_severity'] = 'minor'
            analysis['recommendation_factors'].append('Small gaps (<30 min)')
        
        # Assess overall data quality
        if analysis['quality_score'] > 80:
            analysis['data_quality'] = 'high'
            analysis['recommendation_factors'].append('High quality score')
        elif analysis['quality_score'] > 60:
            analysis['data_quality'] = 'medium'
            analysis['recommendation_factors'].append('Medium quality score')
        else:
            analysis['data_quality'] = 'low'
            analysis['recommendation_factors'].append('Low quality score')
        
        # Assess trading activity
        if analysis['data_density'] > 50:  # >50 data points per hour
            analysis['trading_activity'] = 'high'
            analysis['recommendation_factors'].append('High trading activity')
        elif analysis['data_density'] > 20:
            analysis['trading_activity'] = 'medium'
            analysis['recommendation_factors'].append('Medium trading activity')
        else:
            analysis['trading_activity'] = 'low'
            analysis['recommendation_factors'].append('Low trading activity')
        
        return analysis
    
    def _recommend_gap_token_action(self, analysis: Dict) -> str:
        """Recommend action for a token with gaps based on analysis"""
        
        # Automatic removal criteria
        if (analysis['total_gaps'] > 15 or 
            analysis['max_gap_minutes'] > 59 or 
            analysis['quality_score'] < 40):
            return 'remove_completely'
        
        # Automatic keep and clean criteria
        if (analysis['total_gaps'] <= 3 and 
            analysis['max_gap_minutes'] <= 10 and 
            analysis['quality_score'] > 70):
            return 'keep_and_clean'
        
        # Manual review criteria (borderline cases)
        if (analysis['total_gaps'] <= 8 and 
            analysis['max_gap_minutes'] <= 59 and 
            analysis['quality_score'] > 50):
            return 'needs_manual_review'
        
        # Default to removal for problematic cases
        return 'remove_completely'
    
    def _print_gap_investigation_summary(self, results: Dict):
        """Print comprehensive summary of gap investigation"""
        
        print(f"\n📊 GAP INVESTIGATION SUMMARY")
        print(f"Tokens analyzed: {results['tokens_analyzed']}")
        print(f"=" * 50)
        
        # Recommendations summary
        print(f"\n🎯 RECOMMENDATIONS:")
        for action, tokens in results['recommendations'].items():
            action_name = action.replace('_', ' ').title()
            print(f"  {action_name}: {len(tokens)} tokens")
            if tokens:
                print(f"    Examples: {', '.join(tokens[:3])}")
                if len(tokens) > 3:
                    print(f"    ... and {len(tokens) - 3} more")
        
        # Gap severity breakdown
        print(f"\n🔍 GAP SEVERITY BREAKDOWN:")
        for severity, tokens in results['gap_analysis'].items():
            severity_name = severity.replace('_', ' ').title()
            print(f"  {severity_name}: {len(tokens)} tokens")
        
        # Detailed recommendations
        print(f"\n💡 DETAILED RECOMMENDATIONS:")
        
        keep_tokens = results['recommendations']['keep_and_clean']
        if keep_tokens:
            print(f"\n✅ KEEP AND CLEAN ({len(keep_tokens)} tokens):")
            print(f"   These tokens have minor gaps that can be filled effectively")
            print(f"   → Run data cleaning with 'aggressive' strategy")
            
        remove_tokens = results['recommendations']['remove_completely']
        if remove_tokens:
            print(f"\n❌ REMOVE COMPLETELY ({len(remove_tokens)} tokens):")
            print(f"   These tokens have too many/large gaps for reliable analysis")
            print(f"   → Exclude from training data entirely")
            
        review_tokens = results['recommendations']['needs_manual_review']
        if review_tokens:
            print(f"\n🤔 NEEDS MANUAL REVIEW ({len(review_tokens)} tokens):")
            print(f"   These tokens are borderline cases")
            print(f"   → Examine individual tokens to make final decision")
            
            # Show details for manual review tokens
            for token in review_tokens[:5]:  # Show first 5
                if token in results['detailed_analysis']:
                    analysis = results['detailed_analysis'][token]
                    print(f"\n   📋 {token}:")
                    print(f"      Gaps: {analysis['total_gaps']} (max: {analysis['max_gap_minutes']:.1f} min)")
                    print(f"      Quality: {analysis['quality_score']:.1f}/100")
                    print(f"      Activity: {analysis['data_density']:.1f} points/hour")
                    print(f"      Factors: {', '.join(analysis['recommendation_factors'][:2])}")
        
        print(f"\n🔧 NEXT STEPS:")
        print(f"1. Review the recommendations above")
        print(f"2. For 'keep_and_clean' tokens: Run data cleaning")
        print(f"3. For 'remove_completely' tokens: Exclude from analysis")
        print(f"4. For 'needs_manual_review' tokens: Examine individually")
        print(f"5. Update your token categorization accordingly")

    def _display_outlier_summary(self, quality_reports: Dict):
        """Display outlier detection summary in Streamlit"""
        try:
            # Collect outlier statistics across all tokens
            outlier_stats = {
                'winsorization': {'total_outliers': 0, 'tokens_with_outliers': 0},
                'z_score': {'total_outliers': 0, 'tokens_with_outliers': 0},
                'iqr': {'total_outliers': 0, 'tokens_with_outliers': 0},
                'modified_z_score': {'total_outliers': 0, 'tokens_with_outliers': 0},
                'consensus': {'total_outliers': 0, 'tokens_with_outliers': 0}
            }
            
            total_tokens_analyzed = 0
            
            for token, report in quality_reports.items():
                outlier_analysis = report.get('outlier_analysis', {})
                if outlier_analysis.get('status') == 'success':
                    total_tokens_analyzed += 1
                    summary = outlier_analysis.get('summary', {})
                    outlier_counts = summary.get('outlier_counts', {})
                    
                    # Aggregate statistics
                    for method in ['winsorization', 'z_score', 'iqr', 'modified_z_score']:
                        count = outlier_counts.get(method, 0)
                        if count > 0:
                            outlier_stats[method]['total_outliers'] += count
                            outlier_stats[method]['tokens_with_outliers'] += 1
                    
                    # Consensus outliers
                    consensus_count = summary.get('consensus_outliers', 0)
                    if consensus_count > 0:
                        outlier_stats['consensus']['total_outliers'] += consensus_count
                        outlier_stats['consensus']['tokens_with_outliers'] += 1
            
            if total_tokens_analyzed == 0:
                st.warning("No outlier analysis data available. Outlier detection uses local rolling windows for cryptocurrency data.")
                return
            
            # Display summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Winsorization Outliers", 
                         f"{outlier_stats['winsorization']['tokens_with_outliers']}/{total_tokens_analyzed}",
                         f"{outlier_stats['winsorization']['total_outliers']} total")
            
            with col2:
                st.metric("Z-Score Outliers", 
                         f"{outlier_stats['z_score']['tokens_with_outliers']}/{total_tokens_analyzed}",
                         f"{outlier_stats['z_score']['total_outliers']} total")
            
            with col3:
                st.metric("IQR Outliers", 
                         f"{outlier_stats['iqr']['tokens_with_outliers']}/{total_tokens_analyzed}",
                         f"{outlier_stats['iqr']['total_outliers']} total")
            
            with col4:
                st.metric("Modified Z-Score", 
                         f"{outlier_stats['modified_z_score']['tokens_with_outliers']}/{total_tokens_analyzed}",
                         f"{outlier_stats['modified_z_score']['total_outliers']} total")
            
            with col5:
                st.metric("Consensus Outliers", 
                         f"{outlier_stats['consensus']['tokens_with_outliers']}/{total_tokens_analyzed}",
                         f"{outlier_stats['consensus']['total_outliers']} total")
            
            # Show detailed outlier analysis for tokens with high outlier counts
            st.subheader("Tokens with High Outlier Counts")
            high_outlier_tokens = []
            
            for token, report in quality_reports.items():
                outlier_analysis = report.get('outlier_analysis', {})
                if outlier_analysis.get('status') == 'success':
                    summary = outlier_analysis.get('summary', {})
                    consensus_count = summary.get('consensus_outliers', 0)
                    total_points = summary.get('total_points', 0)
                    
                    if total_points > 0:
                        outlier_percentage = (consensus_count / total_points) * 100
                        if outlier_percentage > 5:  # More than 5% outliers
                            high_outlier_tokens.append({
                                'Token': token,
                                'Consensus Outliers': consensus_count,
                                'Total Points': total_points,
                                'Outlier %': f"{outlier_percentage:.1f}%"
                            })
            
            if high_outlier_tokens:
                # Sort by outlier percentage
                high_outlier_tokens.sort(key=lambda x: float(x['Outlier %'][:-1]), reverse=True)
                outlier_df = pl.DataFrame(high_outlier_tokens[:10])  # Show top 10
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.info("No tokens with significant outlier counts (>5%)")
            
        except Exception as e:
            self.logger.error(f"Error displaying outlier summary: {e}")
            st.error(f"Error displaying outlier summary: {e}")

    # ========================================
    # COMPREHENSIVE OUTLIER DETECTION METHODS
    # ========================================
    
    def detect_outliers_winsorization(self, df: pl.DataFrame, 
                                    price_col: str = 'price',
                                    lower_pct: float = 0.01,
                                    upper_pct: float = 0.99) -> pl.DataFrame:
        """
        Apply winsorization to detect and cap outliers
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            lower_pct: Lower percentile for winsorization (e.g., 0.01 = 1st percentile)
            upper_pct: Upper percentile for winsorization (e.g., 0.99 = 99th percentile)
            
        Returns:
            DataFrame with winsorized prices and outlier flags
        """
        try:
            # Calculate percentiles using Polars
            lower_bound = df.select(pl.col(price_col).quantile(lower_pct)).item()
            upper_bound = df.select(pl.col(price_col).quantile(upper_pct)).item()
            
            # Apply winsorization and flag outliers
            df_winsorized = df.with_columns([
                # Flag outliers before winsorization
                ((pl.col(price_col) < lower_bound) | (pl.col(price_col) > upper_bound)).alias('is_outlier_winsor'),
                
                # Apply winsorization
                pl.when(pl.col(price_col) < lower_bound).then(lower_bound)
                .when(pl.col(price_col) > upper_bound).then(upper_bound)
                .otherwise(pl.col(price_col))
                .alias(f'{price_col}_winsorized'),
                
                # Calculate outlier magnitude
                pl.when(pl.col(price_col) < lower_bound).then((lower_bound - pl.col(price_col)) / lower_bound)
                .when(pl.col(price_col) > upper_bound).then((pl.col(price_col) - upper_bound) / upper_bound)
                .otherwise(0.0)
                .alias('outlier_magnitude_winsor')
            ])
            
            return df_winsorized
            
        except Exception as e:
            self.logger.error(f"Error in winsorization: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_winsor'),
                pl.col(price_col).alias(f'{price_col}_winsorized'),
                pl.lit(0.0).alias('outlier_magnitude_winsor')
            ])
    
    def detect_outliers_z_score(self, df: pl.DataFrame,
                               price_col: str = 'price',
                               threshold: float = 3.0,
                               use_returns: bool = True) -> pl.DataFrame:
        """
        Detect outliers using Z-score method
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            threshold: Z-score threshold (typically 3.0)
            use_returns: Whether to use returns instead of raw prices
            
        Returns:
            DataFrame with Z-score outlier flags
        """
        try:
            if use_returns:
                # Calculate returns first
                df_with_returns = df.with_columns([
                    pl.col(price_col).pct_change().alias('returns')
                ])
                
                # Calculate Z-scores for returns
                df_z_score = df_with_returns.with_columns([
                    ((pl.col('returns') - pl.col('returns').mean()) / pl.col('returns').std()).alias('z_score'),
                ]).with_columns([
                    (pl.col('z_score').abs() > threshold).alias('is_outlier_z_score'),
                    pl.col('z_score').abs().alias('outlier_magnitude_z_score')
                ])
            else:
                # Calculate Z-scores for raw prices
                df_z_score = df.with_columns([
                    ((pl.col(price_col) - pl.col(price_col).mean()) / pl.col(price_col).std()).alias('z_score'),
                ]).with_columns([
                    (pl.col('z_score').abs() > threshold).alias('is_outlier_z_score'),
                    pl.col('z_score').abs().alias('outlier_magnitude_z_score')
                ])
            
            return df_z_score
            
        except Exception as e:
            self.logger.error(f"Error in Z-score detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_z_score'),
                pl.lit(0.0).alias('outlier_magnitude_z_score')
            ])
    
    def detect_outliers_iqr(self, df: pl.DataFrame,
                           price_col: str = 'price',
                           multiplier: float = 1.5,
                           use_returns: bool = True) -> pl.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers)
            use_returns: Whether to use returns instead of raw prices
            
        Returns:
            DataFrame with IQR outlier flags
        """
        try:
            if use_returns:
                # Calculate returns first
                df_with_returns = df.with_columns([
                    pl.col(price_col).pct_change().alias('returns')
                ])
                
                analysis_col = 'returns'
                df_working = df_with_returns
            else:
                analysis_col = price_col
                df_working = df
            
            # Calculate IQR using Polars
            q1 = df_working.select(pl.col(analysis_col).quantile(0.25)).item()
            q3 = df_working.select(pl.col(analysis_col).quantile(0.75)).item()
            
            if q1 is None or q3 is None:
                return df.with_columns([
                    pl.lit(False).alias('is_outlier_iqr'),
                    pl.lit(0.0).alias('outlier_magnitude_iqr')
                ])
            
            iqr = q3 - q1
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            
            # Detect outliers
            df_iqr = df_working.with_columns([
                ((pl.col(analysis_col) < lower_bound) | (pl.col(analysis_col) > upper_bound)).alias('is_outlier_iqr'),
                
                # Calculate outlier magnitude relative to IQR
                pl.when(pl.col(analysis_col) < lower_bound).then((lower_bound - pl.col(analysis_col)) / iqr)
                .when(pl.col(analysis_col) > upper_bound).then((pl.col(analysis_col) - upper_bound) / iqr)
                .otherwise(0.0)
                .alias('outlier_magnitude_iqr')
            ])
            
            return df_iqr
            
        except Exception as e:
            self.logger.error(f"Error in IQR detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_iqr'),
                pl.lit(0.0).alias('outlier_magnitude_iqr')
            ])
    
    def detect_outliers_modified_z_score(self, df: pl.DataFrame,
                                        price_col: str = 'price',
                                        threshold: float = 3.5,
                                        use_returns: bool = True) -> pl.DataFrame:
        """
        Detect outliers using Modified Z-score (using median instead of mean)
        More robust to outliers than standard Z-score
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            threshold: Modified Z-score threshold (typically 3.5)
            use_returns: Whether to use returns instead of raw prices
            
        Returns:
            DataFrame with Modified Z-score outlier flags
        """
        try:
            if use_returns:
                # Calculate returns first
                df_with_returns = df.with_columns([
                    pl.col(price_col).pct_change().alias('returns')
                ])
                
                analysis_col = 'returns'
                df_working = df_with_returns
            else:
                analysis_col = price_col
                df_working = df
            
            # Calculate median and MAD (Median Absolute Deviation)
            median_val = df_working.select(pl.col(analysis_col).median()).item()
            
            if median_val is None:
                return df.with_columns([
                    pl.lit(False).alias('is_outlier_mod_z'),
                    pl.lit(0.0).alias('outlier_magnitude_mod_z')
                ])
            
            # Calculate MAD and modified Z-score
            df_mod_z = df_working.with_columns([
                (pl.col(analysis_col) - median_val).abs().alias('abs_dev_from_median')
            ]).with_columns([
                pl.col('abs_dev_from_median').median().alias('mad')
            ]).with_columns([
                # Modified Z-score = 0.6745 * (x - median) / MAD
                (0.6745 * (pl.col(analysis_col) - median_val) / pl.col('mad')).alias('modified_z_score')
            ]).with_columns([
                (pl.col('modified_z_score').abs() > threshold).alias('is_outlier_mod_z'),
                pl.col('modified_z_score').abs().alias('outlier_magnitude_mod_z')
            ])
            
            return df_mod_z
            
        except Exception as e:
            self.logger.error(f"Error in Modified Z-score detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_mod_z'),
                pl.lit(0.0).alias('outlier_magnitude_mod_z')
            ])
    
    def comprehensive_outlier_analysis(self, df: pl.DataFrame,
                                     price_col: str = 'price',
                                     methods: List[str] = None,
                                     use_local_detection: bool = True,
                                     local_window: int = 60) -> Dict:
        """
        Perform comprehensive outlier analysis using multiple methods
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            methods: List of methods to use ['winsorization', 'z_score', 'iqr', 'modified_z_score']
            use_local_detection: Whether to use local rolling window detection (better for crypto)
            local_window: Size of rolling window for local outlier detection (minutes)
            
        Returns:
            Dictionary with outlier analysis results and processed DataFrame
        """
        if methods is None:
            methods = ['winsorization', 'z_score', 'iqr', 'modified_z_score']
        
        try:
            df_processed = df.clone()
            outlier_summary = {
                'total_points': len(df),
                'methods_used': methods,
                'outlier_counts': {},
                'outlier_percentages': {},
                'consensus_outliers': 0,
                'method_agreement': {}
            }
            
            # Apply each method (global or local detection)
            if 'winsorization' in methods:
                if use_local_detection:
                    df_processed = self.detect_outliers_winsorization_local(df_processed, price_col, local_window)
                else:
                    df_processed = self.detect_outliers_winsorization(df_processed, price_col)
                outlier_count = df_processed['is_outlier_winsor'].sum()
                outlier_summary['outlier_counts']['winsorization'] = outlier_count
                outlier_summary['outlier_percentages']['winsorization'] = (outlier_count / len(df)) * 100
            
            if 'z_score' in methods:
                if use_local_detection:
                    df_processed = self.detect_outliers_z_score_local(df_processed, price_col, local_window)
                else:
                    df_processed = self.detect_outliers_z_score(df_processed, price_col)
                outlier_count = df_processed['is_outlier_z_score'].sum()
                outlier_summary['outlier_counts']['z_score'] = outlier_count
                outlier_summary['outlier_percentages']['z_score'] = (outlier_count / len(df)) * 100
            
            if 'iqr' in methods:
                if use_local_detection:
                    df_processed = self.detect_outliers_iqr_local(df_processed, price_col, local_window)
                else:
                    df_processed = self.detect_outliers_iqr(df_processed, price_col)
                outlier_count = df_processed['is_outlier_iqr'].sum()
                outlier_summary['outlier_counts']['iqr'] = outlier_count
                outlier_summary['outlier_percentages']['iqr'] = (outlier_count / len(df)) * 100
            
            if 'modified_z_score' in methods:
                if use_local_detection:
                    df_processed = self.detect_outliers_modified_z_score_local(df_processed, price_col, local_window)
                else:
                    df_processed = self.detect_outliers_modified_z_score(df_processed, price_col)
                outlier_count = df_processed['is_outlier_mod_z'].sum()
                outlier_summary['outlier_counts']['modified_z_score'] = outlier_count
                outlier_summary['outlier_percentages']['modified_z_score'] = (outlier_count / len(df)) * 100
            
            # Calculate consensus outliers (flagged by multiple methods)
            outlier_cols = [col for col in df_processed.columns if col.startswith('is_outlier_')]
            if len(outlier_cols) > 1:
                # Sum outlier flags across methods
                df_processed = df_processed.with_columns([
                    pl.sum_horizontal(outlier_cols).alias('outlier_method_count')
                ]).with_columns([
                    (pl.col('outlier_method_count') >= 2).alias('is_consensus_outlier')
                ])
                
                consensus_count = df_processed['is_consensus_outlier'].sum()
                outlier_summary['consensus_outliers'] = consensus_count
                outlier_summary['consensus_percentage'] = (consensus_count / len(df)) * 100
            
            # Method agreement analysis
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    col1 = f'is_outlier_{method1.replace("_score", "").replace("ization", "")}'
                    col2 = f'is_outlier_{method2.replace("_score", "").replace("ization", "")}'
                    
                    if col1 in df_processed.columns and col2 in df_processed.columns:
                        agreement = df_processed.filter(
                            pl.col(col1) == pl.col(col2)
                        ).height / len(df_processed)
                        
                        outlier_summary['method_agreement'][f'{method1}_vs_{method2}'] = agreement
            
            return {
                'summary': outlier_summary,
                'processed_dataframe': df_processed,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive outlier analysis: {e}")
            return {
                'summary': {'error': str(e)},
                'processed_dataframe': df,
                'status': 'error'
            }
    
    # ========================================
    # LOCAL OUTLIER DETECTION METHODS (Better for Crypto)
    # ========================================
    
    def detect_outliers_winsorization_local(self, df: pl.DataFrame, 
                                           price_col: str = 'price',
                                           window_size: int = 60) -> pl.DataFrame:
        """
        Apply local winsorization within rolling windows (better for crypto data)
        """
        try:
            # Calculate returns for analysis
            df = df.with_columns([
                pl.col(price_col).pct_change().alias('returns')
            ])
            
            # Get winsorization thresholds
            lower_pct = self.outlier_methods['winsorization']['lower']
            upper_pct = self.outlier_methods['winsorization']['upper']
            
            # Apply rolling winsorization
            df = df.with_columns([
                # Rolling quantiles for winsorization bounds
                pl.col('returns').rolling_quantile(lower_pct, window_size=window_size).alias('rolling_lower'),
                pl.col('returns').rolling_quantile(upper_pct, window_size=window_size).alias('rolling_upper')
            ]).with_columns([
                # Flag local outliers
                ((pl.col('returns') < pl.col('rolling_lower')) | 
                 (pl.col('returns') > pl.col('rolling_upper'))).alias('is_outlier_winsor'),
                
                # Calculate magnitude relative to local bounds
                pl.when(pl.col('returns') < pl.col('rolling_lower'))
                .then((pl.col('rolling_lower') - pl.col('returns')) / pl.col('rolling_lower').abs())
                .when(pl.col('returns') > pl.col('rolling_upper'))
                .then((pl.col('returns') - pl.col('rolling_upper')) / pl.col('rolling_upper').abs())
                .otherwise(0.0)
                .alias('outlier_magnitude_winsor')
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in local winsorization: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_winsor'),
                pl.lit(0.0).alias('outlier_magnitude_winsor')
            ])
    
    def detect_outliers_z_score_local(self, df: pl.DataFrame,
                                     price_col: str = 'price',
                                     window_size: int = 60) -> pl.DataFrame:
        """
        Detect outliers using local Z-score within rolling windows
        """
        try:
            # Calculate returns for analysis
            df = df.with_columns([
                pl.col(price_col).pct_change().alias('returns')
            ])
            
            threshold = self.outlier_methods['z_score']['threshold']
            
            # Apply rolling Z-score
            df = df.with_columns([
                # Rolling mean and std for local Z-score
                pl.col('returns').rolling_mean(window_size).alias('rolling_mean'),
                pl.col('returns').rolling_std(window_size).alias('rolling_std')
            ]).with_columns([
                # Local Z-score
                ((pl.col('returns') - pl.col('rolling_mean')) / pl.col('rolling_std')).alias('local_z_score')
            ]).with_columns([
                # Flag local outliers
                (pl.col('local_z_score').abs() > threshold).alias('is_outlier_z_score'),
                pl.col('local_z_score').abs().alias('outlier_magnitude_z_score')
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in local Z-score detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_z_score'),
                pl.lit(0.0).alias('outlier_magnitude_z_score')
            ])
    
    def detect_outliers_iqr_local(self, df: pl.DataFrame,
                                 price_col: str = 'price',
                                 window_size: int = 60) -> pl.DataFrame:
        """
        Detect outliers using local IQR within rolling windows
        """
        try:
            # Calculate returns for analysis
            df = df.with_columns([
                pl.col(price_col).pct_change().alias('returns')
            ])
            
            multiplier = self.outlier_methods['iqr']['multiplier']
            
            # Apply rolling IQR
            df = df.with_columns([
                # Rolling quartiles
                pl.col('returns').rolling_quantile(0.25, window_size=window_size).alias('rolling_q1'),
                pl.col('returns').rolling_quantile(0.75, window_size=window_size).alias('rolling_q3')
            ]).with_columns([
                # Local IQR and bounds
                (pl.col('rolling_q3') - pl.col('rolling_q1')).alias('rolling_iqr'),
            ]).with_columns([
                # IQR bounds
                (pl.col('rolling_q1') - multiplier * pl.col('rolling_iqr')).alias('iqr_lower'),
                (pl.col('rolling_q3') + multiplier * pl.col('rolling_iqr')).alias('iqr_upper')
            ]).with_columns([
                # Flag local outliers
                ((pl.col('returns') < pl.col('iqr_lower')) | 
                 (pl.col('returns') > pl.col('iqr_upper'))).alias('is_outlier_iqr'),
                
                # Calculate magnitude relative to local IQR
                pl.when(pl.col('returns') < pl.col('iqr_lower'))
                .then((pl.col('iqr_lower') - pl.col('returns')) / pl.col('rolling_iqr'))
                .when(pl.col('returns') > pl.col('iqr_upper'))
                .then((pl.col('returns') - pl.col('iqr_upper')) / pl.col('rolling_iqr'))
                .otherwise(0.0)
                .alias('outlier_magnitude_iqr')
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in local IQR detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_iqr'),
                pl.lit(0.0).alias('outlier_magnitude_iqr')
            ])
    
    def detect_outliers_modified_z_score_local(self, df: pl.DataFrame,
                                              price_col: str = 'price',
                                              window_size: int = 60) -> pl.DataFrame:
        """
        Detect outliers using local Modified Z-score within rolling windows
        """
        try:
            # Calculate returns for analysis
            df = df.with_columns([
                pl.col(price_col).pct_change().alias('returns')
            ])
            
            threshold = self.outlier_methods['modified_z_score']['threshold']
            
            # Apply rolling Modified Z-score
            df = df.with_columns([
                # Rolling median
                pl.col('returns').rolling_median(window_size).alias('rolling_median')
            ]).with_columns([
                # Rolling MAD (Median Absolute Deviation)
                (pl.col('returns') - pl.col('rolling_median')).abs().rolling_median(window_size).alias('rolling_mad')
            ]).with_columns([
                # Local Modified Z-score
                (0.6745 * (pl.col('returns') - pl.col('rolling_median')) / pl.col('rolling_mad')).alias('local_mod_z_score')
            ]).with_columns([
                # Flag local outliers
                (pl.col('local_mod_z_score').abs() > threshold).alias('is_outlier_mod_z'),
                pl.col('local_mod_z_score').abs().alias('outlier_magnitude_mod_z')
            ])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in local Modified Z-score detection: {e}")
            return df.with_columns([
                pl.lit(False).alias('is_outlier_mod_z'),
                pl.lit(0.0).alias('outlier_magnitude_mod_z')
            ])