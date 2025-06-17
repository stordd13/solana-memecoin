"""
Price analysis for memecoin data using Polars
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class PriceAnalyzer:
    def __init__(self):
        """Initialize the price analyzer"""
        self.logger = logging.getLogger(__name__)
        self.pattern_thresholds = {
            'pump_threshold': 0.5,      # 50% increase for pump detection
            'dump_threshold': -0.3,     # 30% decrease for dump detection
            'volatility_threshold': 0.2, # 20% standard deviation for volatility
            'trend_threshold': 0.02,    # 2% for trend detection
            'momentum_threshold': 0.01   # 1% for momentum shifts
        }
        
    def analyze_prices(self, df: pl.DataFrame, token: str) -> Dict:
        """
        Analyze price data for a single token using Polars
        
        Args:
            df: DataFrame with token data
            token: Token symbol
            
        Returns:
            Dictionary with price analysis metrics
        """
        try:
            if df is None or df.height == 0:
                self.logger.warning(f"Empty or None DataFrame for token {token}")
                return self._empty_price_report(token, "No data available")
                
            # Validate required columns
            required_cols = ['price', 'datetime']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return self._empty_price_report(token, f"Missing columns: {missing_cols}")
                
            # Validate data types
            if not df['datetime'].dtype == pl.Datetime:
                self.logger.error(f"Invalid datetime type: {df['datetime'].dtype}")
                return self._empty_price_report(token, "Invalid datetime type")
                
            if not df['price'].dtype in [pl.Float64, pl.Float32]:
                self.logger.error(f"Invalid price type: {df['price'].dtype}")
                return self._empty_price_report(token, "Invalid price type")
                
            # Remove any rows with null values
            df = df.drop_nulls(['price', 'datetime'])
            if df.height == 0:
                self.logger.warning(f"No valid data after removing nulls for {token}")
                return self._empty_price_report(token, "No valid data after cleaning")
                
            # Calculate basic price statistics
            price_stats = self._calculate_price_stats(df)
            
            # Calculate temporal features
            temporal_features = self._calculate_temporal_features(df)
            
            # Calculate movement patterns
            movement_patterns = self._calculate_movement_patterns(df)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(df)
            
            # Calculate momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(df)
            
            # Detect patterns
            patterns = self._detect_patterns(df)
            
            # Calculate optimal return
            optimal_return_metrics = self._calculate_optimal_return(df)

            return {
                'token': token,
                'price_stats': price_stats,
                'temporal_features': temporal_features,
                'movement_patterns': movement_patterns,
                'volatility_metrics': volatility_metrics,
                'momentum_metrics': momentum_metrics,
                'patterns': patterns,
                'optimal_return_metrics': optimal_return_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing prices for {token}: {str(e)}")
            return self._empty_price_report(token, error=str(e))
            
    def _calculate_price_stats(self, df: pl.DataFrame) -> Dict:
        """Calculate basic price statistics using Polars"""
        price_col = pl.col('price')
        
        return {
            'min_price': df.select(price_col.min()).item(),
            'max_price': df.select(price_col.max()).item(),
            'mean_price': df.select(price_col.mean()).item(),
            'median_price': df.select(price_col.median()).item(),
            'std_price': df.select(price_col.std()).item(),
            'launch_price': df.select(price_col.first()).item(),
            'current_price': df.select(price_col.last()).item(),
            'total_return': (df.select(price_col.last()).item() / 
                           df.select(price_col.first()).item() - 1)
        }
        
    def _calculate_temporal_features(self, df: pl.DataFrame) -> Dict:
        """Calculate temporal features using Polars"""
        try:
            # Calculate time to peak/bottom
            price_col = pl.col('price')
            max_price = df.select(price_col.max()).item()
            min_price = df.select(price_col.min()).item()
            
            # Get indices of max and min prices
            max_idx = df.filter(price_col == max_price).select(pl.col('datetime').first()).item()
            min_idx = df.filter(price_col == min_price).select(pl.col('datetime').first()).item()
            start_time = df.select(pl.col('datetime').first()).item()
            
            # Handle None values
            if None in (max_idx, min_idx, start_time):
                self.logger.warning("Missing datetime values in temporal features calculation")
                return {
                    'time_to_peak': 0,
                    'time_to_bottom': 0,
                    'time_above_launch': 0
                }
            
            # Calculate minutes from start
            time_to_peak = int((max_idx - start_time).total_seconds() / 60)
            time_to_bottom = int((min_idx - start_time).total_seconds() / 60)
            
            # Calculate time above launch price
            launch_price = df.select(price_col.first()).item()
            if launch_price is None:
                self.logger.warning("Missing launch price in temporal features calculation")
                return {
                    'time_to_peak': time_to_peak,
                    'time_to_bottom': time_to_bottom,
                    'time_above_launch': 0
                }
            
            time_above_launch = df.filter(price_col > launch_price).height
            
            return {
                'time_to_peak': time_to_peak,
                'time_to_bottom': time_to_bottom,
                'time_above_launch': time_above_launch
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal features: {str(e)}")
            return {
                'time_to_peak': 0,
                'time_to_bottom': 0,
                'time_above_launch': 0
            }
        
    def _calculate_movement_patterns(self, df: pl.DataFrame) -> Dict:
        """Calculate price movement patterns using Polars"""
        try:
            # Ensure returns column exists
            if 'returns' not in df.columns:
                df = df.with_columns([
                    pl.col('price').pct_change().fill_null(0).alias('returns')
                ])
            
            # Calculate drawdowns using a simpler approach
            # First get the maximum price up to each point
            df = df.with_columns([
                pl.col('price').max().alias('max_price')
            ])
            
            # Then calculate drawdown as percentage from max
            df = df.with_columns([
                ((pl.col('price') - pl.col('max_price')) / pl.col('max_price')).alias('drawdown')
            ])
            
            # Calculate drawdown statistics
            max_drawdown = df.select(pl.col('drawdown').min()).item()
            avg_drawdown = df.select(pl.col('drawdown').mean()).item()
            
            # Calculate recovery time (time to recover from max drawdown)
            if max_drawdown is not None and max_drawdown < 0:
                # Find the point of maximum drawdown
                max_dd_idx = df.select(pl.col('drawdown').arg_min()).item()
                if max_dd_idx is not None:
                    # Get the price at max drawdown
                    max_dd_price = df.select(pl.col('price').gather(max_dd_idx)).item()
                    if max_dd_price is not None:
                        # Find the first point after max drawdown where price recovers
                        recovery_mask = (pl.col('price') >= max_dd_price) & (pl.col('datetime') > df.select(pl.col('datetime').gather(max_dd_idx)).item())
                        recovery_df = df.filter(recovery_mask)
                        if recovery_df.height > 0:
                            recovery_time = recovery_df.select(pl.col('datetime').first()).item()
                            max_dd_time = df.select(pl.col('datetime').gather(max_dd_idx)).item()
                            if recovery_time is not None and max_dd_time is not None:
                                recovery_minutes = (recovery_time - max_dd_time).total_seconds() / 60
                            else:
                                recovery_minutes = None
                        else:
                            recovery_minutes = None
                    else:
                        recovery_minutes = None
                else:
                    recovery_minutes = None
            else:
                recovery_minutes = None
            
            # Calculate consecutive moves
            df = df.with_columns([
                pl.when(pl.col('returns') > 0).then(1)
                .when(pl.col('returns') < 0).then(-1)
                .otherwise(0).alias('move_direction')
            ])
            
            # Calculate max consecutive moves
            max_consecutive_up = df.filter(pl.col('move_direction') == 1).height
            max_consecutive_down = df.filter(pl.col('move_direction') == -1).height
            
            return {
                'max_drawdown': max_drawdown if max_drawdown is not None else 0,
                'avg_drawdown': avg_drawdown if avg_drawdown is not None else 0,
                'recovery_time_minutes': recovery_minutes if recovery_minutes is not None else 0,
                'max_consecutive_up_moves': max_consecutive_up,
                'max_consecutive_down_moves': max_consecutive_down,
                'total_up_moves': df.filter(pl.col('move_direction') == 1).height,
                'total_down_moves': df.filter(pl.col('move_direction') == -1).height
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating movement patterns: {str(e)}")
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'recovery_time_minutes': 0,
                'max_consecutive_up_moves': 0,
                'max_consecutive_down_moves': 0,
                'total_up_moves': 0,
                'total_down_moves': 0
            }
        
    def _calculate_volatility_clustering(self, df: pl.DataFrame) -> Dict:
        """Calculate volatility clustering metrics using Polars"""
        try:
            # Calculate volatility persistence using lag correlation
            if 'rolling_vol' not in df.columns:
                window_size = min(60, df.height)
                df = df.with_columns([
                    pl.col('returns').rolling_std(window_size).fill_null(0).alias('rolling_vol')
                ])
            
            # Calculate lag-1 correlation manually
            df = df.with_columns([
                pl.col('rolling_vol').shift(1).alias('vol_lag1')
            ])
            
            # Calculate correlation between current and lagged volatility
            vol_persistence = df.select([
                pl.corr('rolling_vol', 'vol_lag1')
            ]).item()
            
            # Calculate high volatility periods
            high_vol_threshold = df.select(pl.col('rolling_vol').mean()).item() * 2
            high_vol_periods = df.filter(pl.col('rolling_vol') > high_vol_threshold).height
            
            return {
                'volatility_persistence': vol_persistence if vol_persistence is not None else 0,
                'high_volatility_periods': high_vol_periods,
                'high_volatility_ratio': high_vol_periods / df.height if df.height > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility clustering: {str(e)}")
            return {
                'volatility_persistence': 0,
                'high_volatility_periods': 0,
                'high_volatility_ratio': 0
            }
        
    def _calculate_volatility_metrics(self, df: pl.DataFrame) -> Dict:
        """Calculate volatility metrics using Polars"""
        try:
            # Ensure returns column exists
            if 'returns' not in df.columns:
                df = df.with_columns([
                    pl.col('price').pct_change().fill_null(0).alias('returns')
                ])
            
            # Calculate rolling volatility
            window_size = min(60, df.height)  # 1 hour or less
            df = df.with_columns([
                pl.col('returns').rolling_std(window_size).fill_null(0).alias('rolling_vol')
            ])
            
            # Calculate volatility clustering
            vol_clustering = self._calculate_volatility_clustering(df)
            
            # Calculate additional volatility metrics
            avg_vol = df.select(pl.col('rolling_vol').mean()).item()
            max_vol = df.select(pl.col('rolling_vol').max()).item()
            
            # Calculate volatility of volatility
            df = df.with_columns([
                pl.col('rolling_vol').pct_change().fill_null(0).alias('vol_change')
            ])
            vol_of_vol = df.select(pl.col('vol_change').std()).item()
            
            return {
                'avg_volatility': avg_vol if avg_vol is not None else 0,
                'max_volatility': max_vol if max_vol is not None else 0,
                'volatility_of_volatility': vol_of_vol if vol_of_vol is not None else 0,
                'volatility_clustering': vol_clustering
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {
                'avg_volatility': 0,
                'max_volatility': 0,
                'volatility_of_volatility': 0,
                'volatility_clustering': {
                    'volatility_persistence': 0,
                    'high_volatility_periods': 0,
                    'high_volatility_ratio': 0
                }
            }
        
    def _calculate_momentum_metrics(self, df: pl.DataFrame) -> Dict:
        """Calculate momentum metrics using Polars"""
        try:
            # Ensure returns column exists
            if 'returns' not in df.columns:
                df = df.with_columns([
                    pl.col('price').pct_change().fill_null(0).alias('returns')
                ])
            
            # Calculate momentum indicators
            df = df.with_columns([
                pl.col('returns').rolling_mean(5).fill_null(0).alias('short_momentum'),
                pl.col('returns').rolling_mean(20).fill_null(0).alias('medium_momentum'),
                pl.col('returns').rolling_mean(60).fill_null(0).alias('long_momentum')
            ])
            
            # Detect trend changes
            trend_changes = self._detect_trend_changes(df)
            
            # Detect momentum shifts
            momentum_shifts = self._detect_momentum_shifts(df)
            
            return {
                'trend_changes': trend_changes,
                'momentum_shifts': momentum_shifts,
                'avg_momentum': df.select(pl.col('short_momentum').mean()).item()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum metrics: {str(e)}")
            return {
                'trend_changes': [],
                'momentum_shifts': [],
                'avg_momentum': 0
            }
        
    def _detect_trend_changes(self, df: pl.DataFrame) -> List[Dict]:
        """Detect significant trend changes using Polars"""
        changes = []
        threshold = self.pattern_thresholds['trend_threshold']
        
        # Calculate trend changes
        df = df.with_columns([
            pl.col('returns').rolling_mean(20).diff().alias('trend_change')
        ])
        
        # Find significant changes
        significant_changes = df.filter(
            pl.col('trend_change').abs() > threshold
        )
        
        for row in significant_changes.iter_rows(named=True):
            changes.append({
                'timestamp': row['datetime'],
                'magnitude': row['trend_change'],
                'type': 'up' if row['trend_change'] > 0 else 'down'
            })
            
        return changes
        
    def _detect_momentum_shifts(self, df: pl.DataFrame) -> List[Dict]:
        """Detect momentum shifts using Polars"""
        shifts = []
        threshold = self.pattern_thresholds['momentum_threshold']
        
        # Calculate momentum changes
        df = df.with_columns([
            (pl.col('short_momentum') - pl.col('medium_momentum')).alias('momentum_diff')
        ])
        
        # Find significant shifts
        significant_shifts = df.filter(
            pl.col('momentum_diff').abs() > threshold
        )
        
        for row in significant_shifts.iter_rows(named=True):
            shifts.append({
                'timestamp': row['datetime'],
                'magnitude': row['momentum_diff'],
                'type': 'acceleration' if row['momentum_diff'] > 0 else 'deceleration'
            })
            
        return shifts
        
    def _calculate_recovery_time(self, df: pl.DataFrame) -> int:
        """Calculate recovery time from maximum drawdown"""
        # Get the maximum drawdown point
        max_drawdown = df.select(pl.col('drawdown').min()).item()
        if max_drawdown >= 0:
            return 0
            
        # Find the time of maximum drawdown
        max_dd_time = df.filter(pl.col('drawdown') == max_drawdown).select(pl.col('datetime').first()).item()
        
        # Find the first time after max drawdown where price recovers
        recovery_time = df.filter(
            (pl.col('datetime') > max_dd_time) & 
            (pl.col('cumulative_returns') >= 0)
        ).select(pl.col('datetime').first())
        
        if recovery_time.height == 0:
            return 0
            
        # Calculate minutes to recovery
        recovery_minutes = int((recovery_time.item() - max_dd_time).total_seconds() / 60)
        return max(0, recovery_minutes)
        
    def _detect_patterns(self, df: pl.DataFrame) -> Dict:
        """Detect price patterns using Polars"""
        try:
            # Calculate key metrics
            price_stats = self._calculate_price_stats(df)
            movement_patterns = self._calculate_movement_patterns(df)
            volatility_metrics = self._calculate_volatility_metrics(df)
            
            # Classify pattern
            pattern = self._classify_pattern(
                price_stats=price_stats,
                movement_patterns=movement_patterns,
                volatility_metrics=volatility_metrics
            )
            
            return {
                'pattern': pattern,
                'max_gain': price_stats['max_price'] / price_stats['launch_price'] - 1,
                'final_return': price_stats['total_return'],
                'volatility': volatility_metrics['avg_volatility']
            }
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {
                'pattern': 'unknown',
                'max_gain': 0,
                'final_return': 0,
                'volatility': 0
            }
        
    def _classify_pattern(self, price_stats: Dict, 
                         movement_patterns: Dict,
                         volatility_metrics: Dict) -> str:
        """Classify the price pattern based on metrics"""
        max_gain = price_stats['max_price'] / price_stats['launch_price'] - 1
        final_return = price_stats['total_return']
        volatility = volatility_metrics['avg_volatility']
        max_drawdown = movement_patterns['max_drawdown']
        
        if max_gain > 5 and final_return > 2:  # 500% max gain, 200% final return
            return "explosive_pump"
        elif max_gain > 2 and final_return > 1:  # 200% max gain, 100% final return
            return "steady_climb"
        elif max_gain > 3 and final_return < 0:  # 300% max gain, negative final return
            return "pump_and_dump"
        elif max_drawdown < -0.5 and final_return < -0.3:  # 50% drawdown, 30% loss
            return "steady_decline"
        elif volatility > 0.5:  # 50% volatility
            return "high_volatility"
        else:
            return "sideways"
            
    def _empty_price_report(self, token: str, error: str = None) -> Dict:
        """Return an empty price report"""
        report = {
            'token': token,
            'price_stats': {
                'min_price': 0,
                'max_price': 0,
                'mean_price': 0,
                'median_price': 0,
                'std_price': 0,
                'launch_price': 0,
                'current_price': 0,
                'total_return': 0
            },
            'temporal_features': {
                'time_to_peak': None,
                'time_to_bottom': None,
                'time_above_launch': 0
            },
            'movement_patterns': {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'recovery_time_minutes': 0,
                'max_consecutive_up_moves': 0,
                'max_consecutive_down_moves': 0,
                'total_up_moves': 0,
                'total_down_moves': 0
            },
            'volatility_metrics': {
                'avg_volatility': 0,
                'max_volatility': 0,
                'volatility_of_volatility': 0,
                'volatility_clustering': {
                    'volatility_persistence': 0,
                    'high_volatility_periods': 0,
                    'high_volatility_ratio': 0
                }
            },
            'momentum_metrics': {
                'trend_changes': [],
                'momentum_shifts': [],
                'avg_momentum': 0
            },
            'patterns': {
                'pattern': 'unknown',
                'max_gain': 0,
                'final_return': 0,
                'volatility': 0
            }
        }
        if error:
            report['error'] = error
        return report

    def _calculate_optimal_return(self, df: pl.DataFrame) -> Dict:
        """Calculates the best possible return from buying and selling once."""
        if df.height < 2:
            return {'optimal_return_pct': 0, 'optimal_entry_minutes': 0, 'optimal_exit_minutes': 0}

        # Get the launch time (first timestamp)
        launch_time = df.select(pl.col('datetime').min()).item()

        # Calculate the max price from each point to the end of the series
        df = df.with_columns(
            pl.col('price').reverse().cum_max().reverse().alias('future_max_price')
        )

        # Calculate potential return if we buy at 'price' and sell at 'future_max_price'
        df = df.with_columns(
            ((pl.col('future_max_price') - pl.col('price')) / pl.col('price')).alias('potential_return')
        )

        # Find the maximum potential return
        max_return = df.select(pl.col('potential_return').max()).item()

        if max_return is None or max_return <= 0:
            return {'optimal_return_pct': 0, 'optimal_entry_minutes': 0, 'optimal_exit_minutes': 0}
        
        # Get the row with the best entry point
        entry_row = df.filter(pl.col('potential_return') == max_return).row(0, named=True)
        
        optimal_entry_time = entry_row['datetime']
        future_max_price = entry_row['future_max_price']

        # Find the first time the future_max_price is reached after the entry time
        exit_row = df.filter(
            (pl.col('datetime') >= optimal_entry_time) &
            (pl.col('price') == future_max_price)
        ).row(0, named=True)

        optimal_exit_time = exit_row['datetime']
        
        # Calculate minutes after launch
        optimal_entry_minutes = int((optimal_entry_time - launch_time).total_seconds() / 60)
        optimal_exit_minutes = int((optimal_exit_time - launch_time).total_seconds() / 60)
        
        return {
            'optimal_return_pct': max_return * 100,
            'optimal_entry_minutes': optimal_entry_minutes,
            'optimal_exit_minutes': optimal_exit_minutes
        }

    def display_price_metrics(self, metrics: Dict):
        """Display price analysis metrics in a more visually appealing way."""
        st.subheader(f"Price Analysis for {metrics.get('token', 'N/A')}")
        
        price_stats = metrics.get('price_stats', {})
        volatility_metrics = metrics.get('volatility_metrics', {})
        movement_patterns = metrics.get('movement_patterns', {})
        optimal_metrics = metrics.get('optimal_return_metrics', {})

        # Key Metrics using st.metric
        col1, col2, col3 = st.columns(3)
        
        total_return_pct = price_stats.get('total_return', 0) * 100
        col1.metric("Total Return", f"{total_return_pct:.2f}%")
        
        avg_volatility_pct = volatility_metrics.get('avg_volatility', 0) * 100
        col2.metric("Avg. Volatility", f"{avg_volatility_pct:.2f}%")

        max_drawdown_pct = movement_patterns.get('max_drawdown', 0) * 100
        col3.metric("Max Drawdown", f"{max_drawdown_pct:.2f}%")

        st.divider()

        # Optimal Trading metrics
        st.subheader("Optimal Trade (Single Buy/Sell)")
        col1_opt, col2_opt, col3_opt = st.columns(3)

        optimal_return = optimal_metrics.get('optimal_return_pct', 0)
        col1_opt.metric("Optimal Return", f"{optimal_return:.2f}%")
        
        entry_minutes = optimal_metrics.get('optimal_entry_minutes', 0)
        exit_minutes = optimal_metrics.get('optimal_exit_minutes', 0)
        
        col2_opt.markdown(f"**Entry Time:**<br>{entry_minutes} min after launch", unsafe_allow_html=True)
        col3_opt.markdown(f"**Exit Time:**<br>{exit_minutes} min after launch", unsafe_allow_html=True)

    def display_patterns(self, metrics: Dict) -> None:
        """Display detected patterns for a single token."""
        try:
            st.subheader("Price Patterns and Trends")
            
            # Get pattern data with safe defaults
            token_patterns = metrics.get('patterns', {})
            pumps = token_patterns.get('pumps', {'datetime': [], 'magnitude': []})
            dumps = token_patterns.get('dumps', {'datetime': [], 'magnitude': []})
            trend_changes = token_patterns.get('trend_changes', {'datetime': [], 'magnitude': []})
            momentum_shifts = token_patterns.get('momentum_shifts', {'datetime': [], 'magnitude': []})
            
            # Display pattern summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pump Events", len(pumps.get('datetime', [])))
                st.metric("Dump Events", len(dumps.get('datetime', [])))
            with col2:
                st.metric("Trend Changes", len(trend_changes.get('datetime', [])))
                st.metric("Momentum Shifts", len(momentum_shifts.get('datetime', [])))
            
            # Display pattern timeline
            if any([pumps.get('datetime'), dumps.get('datetime'), 
                   trend_changes.get('datetime'), momentum_shifts.get('datetime')]):
                st.subheader("Pattern Timeline")
                
                # Create timeline data
                timeline_data = []
                
                # Add pump events
                for dt, mag in zip(pumps.get('datetime', []), pumps.get('magnitude', [])):
                    timeline_data.append({
                        'datetime': dt,
                        'event': 'Pump',
                        'magnitude': mag
                    })
                
                # Add dump events
                for dt, mag in zip(dumps.get('datetime', []), dumps.get('magnitude', [])):
                    timeline_data.append({
                        'datetime': dt,
                        'event': 'Dump',
                        'magnitude': mag
                    })
                
                # Add trend changes
                for dt, mag in zip(trend_changes.get('datetime', []), trend_changes.get('magnitude', [])):
                    timeline_data.append({
                        'datetime': dt,
                        'event': 'Trend Change',
                        'magnitude': mag
                    })
                
                # Add momentum shifts
                for dt, mag in zip(momentum_shifts.get('datetime', []), momentum_shifts.get('magnitude', [])):
                    timeline_data.append({
                        'datetime': dt,
                        'event': 'Momentum Shift',
                        'magnitude': mag
                    })
                
                # Sort by datetime
                timeline_data.sort(key=lambda x: x['datetime'])
                
                # Display timeline
                if timeline_data:
                    df = pd.DataFrame(timeline_data)
                    st.dataframe(df)
                else:
                    st.info("No significant patterns detected")
            else:
                st.info("No significant patterns detected")
                
        except KeyError as e:
            st.error(f"Missing key in pattern data: {str(e)}")
        except Exception as e:
            st.error(f"Error displaying patterns: {str(e)}")
            st.info("No pattern data available")

    def display_aggregated_metrics(self, all_metrics: Dict[str, Dict]):
        """Display aggregated metrics for multiple tokens in a more detailed and visually appealing way."""
        if not all_metrics:
            st.warning("No metrics to display.")
            return

        # Create a list of dictionaries, one for each token, to make it easier to work with
        metrics_list = []
        for token, metrics in all_metrics.items():
            price_stats = metrics.get('price_stats', {})
            optimal_metrics = metrics.get('optimal_return_metrics', {})
            volatility_metrics = metrics.get('volatility_metrics', {})
            
            entry_minutes = optimal_metrics.get('optimal_entry_minutes', 0)
            exit_minutes = optimal_metrics.get('optimal_exit_minutes', 0)

            metrics_list.append({
                'token': token,
                'total_return_%': price_stats.get('total_return', 0) * 100,
                'volatility_%': volatility_metrics.get('avg_volatility', 0) * 100,
                'optimal_return_%': optimal_metrics.get('optimal_return_pct', 0),
                'min_price': price_stats.get('min_price', 0),
                'max_price': price_stats.get('max_price', 0),
                'optimal_entry_min': entry_minutes,
                'optimal_exit_min': exit_minutes,
            })
        
        df = pl.DataFrame(metrics_list)

        st.subheader("Aggregated Average Metrics")
        
        # Calculate averages and medians
        col1, col2, col3, col4 = st.columns(4)
        
        # Returns
        avg_total_return = df.select(pl.mean('total_return_%')).item()
        median_total_return = df.select(pl.median('total_return_%')).item()
        col1.metric("Avg. Total Return", f"{avg_total_return:.2f}%")
        col1.metric("Median Total Return", f"{median_total_return:.2f}%")
        
        # Volatility
        avg_volatility = df.select(pl.mean('volatility_%')).item()
        median_volatility = df.select(pl.median('volatility_%')).item()
        col2.metric("Avg. Volatility", f"{avg_volatility:.2f}%")
        col2.metric("Median Volatility", f"{median_volatility:.2f}%")
        
        # Optimal Return
        avg_optimal_return = df.select(pl.mean('optimal_return_%')).item()
        col3.metric("Avg. Optimal Return", f"{avg_optimal_return:.2f}%")

        # Calculate best universal entry/exit timing
        best_entry_exit = self._calculate_best_universal_timing(all_metrics)
        col4.metric("Best Universal Entry", f"{best_entry_exit['best_entry_min']} min")
        col4.metric("Best Universal Exit", f"{best_entry_exit['best_exit_min']} min")
        col4.metric("Universal Return", f"{best_entry_exit['universal_return']:.2f}%")

        st.divider()
        st.subheader("Detailed Metrics per Token")
        st.dataframe(df, use_container_width=True, height=400)

    def _calculate_best_universal_timing(self, all_metrics: Dict[str, Dict]) -> Dict:
        """Calculate the best entry/exit timing that would work across all tokens."""
        best_return = -float('inf')
        best_entry_min = 0
        best_exit_min = 0
        
        # Try different entry/exit combinations (every 10 minutes for efficiency)
        for entry_min in range(0, 1440, 10):  # 0 to 1440 minutes (24 hours)
            for exit_min in range(entry_min + 10, 1440, 10):  # Must exit after entry
                total_return = 0
                valid_tokens = 0
                
                for token, metrics in all_metrics.items():
                    # Get the original DataFrame to calculate return at specific times
                    # This is a simplified calculation - in reality you'd need the actual price data
                    # For now, we'll estimate based on the optimal return metrics
                    optimal_metrics = metrics.get('optimal_return_metrics', {})
                    price_stats = metrics.get('price_stats', {})
                    
                    # Simple estimation: if our timing is close to optimal, use a fraction of optimal return
                    optimal_entry = optimal_metrics.get('optimal_entry_minutes', 0)
                    optimal_exit = optimal_metrics.get('optimal_exit_minutes', 0)
                    optimal_return = optimal_metrics.get('optimal_return_pct', 0)
                    total_token_return = price_stats.get('total_return', 0) * 100
                    
                    # Estimate return based on how close our timing is to optimal
                    entry_distance = abs(entry_min - optimal_entry) / 1440  # Normalize to 0-1
                    exit_distance = abs(exit_min - optimal_exit) / 1440
                    timing_score = 1 - (entry_distance + exit_distance) / 2
                    
                    # Estimate return (this is a simplified model)
                    estimated_return = max(0, timing_score * optimal_return * 0.7)  # 70% of optimal if perfect timing
                    
                    total_return += estimated_return
                    valid_tokens += 1
                
                if valid_tokens > 0:
                    avg_return = total_return / valid_tokens
                    if avg_return > best_return:
                        best_return = avg_return
                        best_entry_min = entry_min
                        best_exit_min = exit_min
        
        return {
            'best_entry_min': best_entry_min,
            'best_exit_min': best_exit_min,
            'universal_return': best_return
        }


if __name__ == '__main__':
    # Example usage for testing
    pass 