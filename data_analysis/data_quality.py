"""
Data quality analysis for memecoin data using Polars
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from export_utils import export_parquet_files
from price_analysis import PriceAnalyzer

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
            is_dead = death_duration_hours >= 2
            
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
                df = pl.scan_parquet(pf)
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
        Identify tokens with extreme price movements
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of extreme tokens with their metrics
        """
        extreme_tokens = {}
        
        for token, report in quality_reports.items():
            if report.get('is_extreme_token', False):
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
                    'total_rows': report.get('total_rows', 0)
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
        Identify tokens with normal behavior (not in dead, gaps, or extremes categories)
        
        Args:
            quality_reports: Dictionary of quality reports from analyze_multiple_files
            
        Returns:
            Dictionary of normal behavior tokens with their metrics
        """
        normal_tokens = {}
        
        for token, report in quality_reports.items():
            # Exclude tokens that are:
            # - Dead (constant price for 4+ hours)
            # - Have extreme movements
            # - Have significant gaps
            is_dead = report.get('is_dead', False)
            is_extreme = report.get('is_extreme_token', False)
            has_significant_gaps = report.get('gaps', {}).get('total_gaps', 0) > 5  # More than 5 gaps
            
            # Only include tokens with normal behavior
            if not (is_dead or is_extreme or has_significant_gaps):
                normal_tokens[token] = {
                    'token': token,
                    'quality_score': report.get('quality_score', 0),
                    'total_rows': report.get('total_rows', 0),
                    'gaps': report.get('gaps', {}).get('total_gaps', 0),
                    'is_dead': is_dead,
                    'is_extreme': is_extreme,
                    'has_significant_gaps': has_significant_gaps
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
        
        # Convert to pandas for easier display
        quality_pd = quality_df.to_pandas()
        
        # Display average quality score
        avg_score = quality_pd['quality_score'].mean()
        st.metric("Average Quality Score", f"{avg_score:.1f}/100")
        
        # Quality score distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=quality_pd['quality_score'],
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
        
        # Sort by quality score
        quality_pd = quality_pd.sort_values('quality_score', ascending=False)
        
        # Display metrics for each token
        for _, row in quality_pd.iterrows():
            with st.expander(f"{row['token']} (Score: {row['quality_score']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Rows", row['total_rows'])
                    st.metric("Unique Dates", row['unique_dates'])
                    st.metric("Duplicate Percentage", f"{row['duplicate_pct']:.1f}%")
                
                with col2:
                    if pd.notna(row['avg_gap']):
                        st.metric("Average Time Gap", str(row['avg_gap']))
                    if pd.notna(row['max_gap']):
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
                if pd.notna(row['max_gap']) and isinstance(row['max_gap'], (timedelta, np.timedelta64)) and row['max_gap'] > timedelta(hours=24):
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