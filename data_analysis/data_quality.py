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

            # --- NEW: Compute volatility, total return, price range ---
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
            
            # A token is considered dead if its price is constant for at least 4 hours
            is_dead = death_duration_hours >= 4
            
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
                'has_extreme_range': has_extreme_range
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing data for {token}: {e}")
            return self._empty_quality_report(token, error=str(e))
            
    def _analyze_gaps(self, datetime_series: pl.Series) -> Dict:
        """Analyze gaps in the time series using Polars"""
        if datetime_series.len() <= 1:
            return {'total_gaps': 0, 'gap_details': []}
            
        # Calculate time differences
        time_diff = datetime_series.diff()
        
        # Identify gaps
        gaps = []
        for i, diff in enumerate(time_diff):
            if diff is not None and diff > timedelta(minutes=1):
                gap_size = diff.total_seconds() / 60
                gaps.append({
                    'position': i,
                    'size_minutes': gap_size,
                    'interpolation_type': self._get_interpolation_type(gap_size)
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
        """Calculate a quality score (0-100) for the data"""
        score = 100
        
        # Penalize for missing data
        completeness = (unique_dates / total_rows) * 100 if total_rows > 0 else 0
        score -= (100 - completeness)
        
        # Penalize for gaps
        score -= gaps['total_gaps'] * 2
        
        # Penalize for price anomalies
        score -= (price_anomalies['zero_prices'] + 
                 price_anomalies['negative_prices'] + 
                 price_anomalies['extreme_ups'] + 
                 price_anomalies['extreme_downs']) * 0.5
        
        # Heavily penalize tokens with extreme jumps
        if has_extreme_jump:
            score -= 50

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
            'has_extreme_range': False
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
                    'has_extreme_range': report['has_extreme_range']
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
                # Gap statistics
                tokens_with_gaps = sum(1 for report in quality_reports.values() 
                                     if report['gaps']['total_gaps'] > 0)
                max_gap_all = max((report['gaps']['max_gap'] 
                                 for report in quality_reports.values()), default=0)
                st.metric("Tokens with Gaps", tokens_with_gaps)
                st.metric("Maximum Gap (minutes)", f"{max_gap_all:.1f}")
            
            with col3:
                # Price anomaly statistics
                total_zero = sum(report['price_anomalies']['zero_prices'] 
                               for report in quality_reports.values())
                total_negative = sum(report['price_anomalies']['negative_prices'] 
                                   for report in quality_reports.values())
                st.metric("Total Zero Prices", total_zero)
                st.metric("Total Negative Prices", total_negative)
            
            with col4:
                tokens_with_issues = sum(1 for report in quality_reports.values() if report['has_extreme_jump'])
                st.metric("Tokens with Issues", tokens_with_issues)
            
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
            gap_stats = {
                'tokens_with_gaps': tokens_with_gaps,
                'max_gap_all': max_gap_all,
                'avg_gap_size': quality_df['avg_gap'].mean(),
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
            
            # Display high quality tokens
            if high_quality:
                st.subheader("High Quality Tokens (Score ≥ 80)")
                # Create a table with key metrics
                table_data = []
                for token, report in high_quality.items():
                    table_data.append({
                        'Token': token,
                        'Quality Score': f"{report['quality_score']:.1f}",
                        'Total Rows': report['total_rows'],
                        'Gaps': report['gaps']['total_gaps'],
                        'Zero Prices': report['price_anomalies']['zero_prices'],
                        'Negative Prices': report['price_anomalies']['negative_prices']
                    })
                table_df = pl.DataFrame(table_data)
                # Add pagination
                items_per_page = 10
                total_pages = (len(table_df) + items_per_page - 1) // items_per_page
                page = st.selectbox("Page", range(1, total_pages + 1), index=0)
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(table_df))
                st.dataframe(
                    table_df.slice(start_idx, end_idx - start_idx),
                    height=400,
                    use_container_width=True
                )
                st.write(f"Showing {start_idx + 1}-{end_idx} of {len(table_df)} tokens")
                # Add download button for high quality tokens
                if st.button("Download High Quality Tokens List", key="download_high_quality_csv_inline"):
                    high_quality_tokens = list(high_quality.keys())
                    st.download_button(
                        label="Download as CSV",
                        data="\n".join(high_quality_tokens),
                        file_name="high_quality_tokens.csv",
                        mime="text/csv",
                        key="download_high_quality_csv_file_inline"
                    )
                # Add export button for high quality token parquet files directly below the table
                if st.button("Export High Quality Token Parquet Files to processed/", key="export_high_quality_parquet_inline"):
                    try:
                        export_parquet_files(list(high_quality.keys()), "High Quality Tokens")
                        st.success('Exported high quality token parquet files!')
                    except Exception as e:
                        st.error(f'Export failed: {e}')
            
            # High-quality tokens
            st.subheader("High Quality Tokens")
            high_quality_tokens = quality_df.filter(pl.col('quality_score') > 80)
            st.dataframe(high_quality_tokens.select(['token', 'quality_score']))
            
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