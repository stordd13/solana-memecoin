#!/usr/bin/env python3
"""
Marathon Pump Analysis Dashboard

Analyzes marathon token pump patterns to answer specific questions:
1. What % of high-vol (CV>0.8) marathons pump >50% after minute 5?
2. What's the average time to 1.5x for those that pump?
3. What % of true marathons get misclassified as standard?

Creates interactive HTML visualizations for comprehensive analysis.

Usage:
    python marathon_pump_analyzer.py [--data-dir PATH] [--archetype-results PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class MarathonPumpAnalyzer:
    """Analyzes marathon token pump patterns with interactive visualizations."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("../results")
        self.archetype_data = {}
        self.token_data = {}
        self.marathon_tokens = []
        self.pump_analysis = {}
        self.misclassification_analysis = {}
        
    def load_archetype_results(self, archetype_results_path: Path) -> None:
        """Load archetype characterization results."""
        print(f"📊 Loading archetype results from: {archetype_results_path}")
        
        with open(archetype_results_path, 'r') as f:
            results = json.load(f)
        
        self.archetype_data = results.get('archetype_data', {})
        
        # Extract marathon tokens
        if 'marathon' in self.archetype_data:
            for archetype_name, archetype_info in self.archetype_data['marathon'].items():
                self.marathon_tokens.extend(archetype_info.get('tokens', []))
        
        print(f"📈 Found {len(self.marathon_tokens)} marathon tokens")
        print(f"📊 Categories: {list(self.archetype_data.keys())}")
        
    def load_token_data(self, data_dir: Path) -> None:
        """Load raw token data for analysis."""
        print(f"📁 Loading token data from: {data_dir}")
        
        # Check if files are directly in data_dir (simplified structure)
        direct_files = list(data_dir.glob("*.parquet"))
        
        if direct_files:
            print(f"  📁 Using simplified directory structure")
            print(f"  📁 Loading marathon tokens from: {data_dir}")
            
            loaded_count = 0
            for token_file in direct_files:
                token_name = token_file.stem
                if token_name in self.marathon_tokens:  # Only load marathon tokens
                    try:
                        df = pl.read_parquet(token_file)
                        self.token_data[token_name] = df
                        loaded_count += 1
                    except Exception as e:
                        print(f"    Warning: Failed to load {token_name}: {e}")
                        
            print(f"📊 Loaded {loaded_count} marathon token files")
        else:
            print(f"❌ No token files found in {data_dir}")
    
    def analyze_pump_patterns(self) -> Dict:
        """Analyze pump patterns for marathon tokens."""
        print(f"⚡ Analyzing pump patterns for {len(self.token_data)} marathon tokens...")
        
        pump_data = []
        
        for token_name, df in self.token_data.items():
            try:
                # Convert to pandas for easier manipulation
                pdf = df.to_pandas()
                
                if len(pdf) < 10:  # Need at least 10 minutes of data
                    continue
                
                # Calculate metrics
                prices = pdf['price'].values
                returns = pdf['price'].pct_change().dropna().values
                
                # Basic token metrics
                lifespan_minutes = len(pdf)
                volatility_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                total_return = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
                
                # Check if token has archetype info
                archetype_info = self._get_token_archetype(token_name)
                
                # Pump analysis from minute 5 onwards
                pump_analysis = self._analyze_post_minute5_pumps(prices)
                
                # Time to 1.5x analysis
                time_to_15x = self._calculate_time_to_15x(prices)
                
                pump_data.append({
                    'token_name': token_name,
                    'lifespan_minutes': lifespan_minutes,
                    'volatility_cv': volatility_cv,
                    'total_return': total_return,
                    'category': archetype_info.get('category', 'unknown'),
                    'cluster': archetype_info.get('cluster', -1),
                    'archetype': archetype_info.get('archetype', 'unknown'),
                    
                    # Pump metrics
                    'pumped_50pct_after_min5': pump_analysis['pumped_50pct'],
                    'max_pump_after_min5': pump_analysis['max_pump'],
                    'pump_timing_min5': pump_analysis['pump_timing'],
                    
                    # Time to 1.5x metrics
                    'time_to_15x_minutes': time_to_15x['time_minutes'],
                    'achieved_15x': time_to_15x['achieved'],
                    'max_multiplier': time_to_15x['max_multiplier'],
                    
                    # Volatility classification
                    'high_volatility': volatility_cv > 0.8,
                    'volatility_bucket': self._classify_volatility(volatility_cv)
                })
                
            except Exception as e:
                print(f"    Warning: Failed to analyze {token_name}: {e}")
                continue
        
        self.pump_analysis = pd.DataFrame(pump_data)
        print(f"✅ Analyzed {len(self.pump_analysis)} marathon tokens")
        
        return self._calculate_summary_metrics()
    
    def _get_token_archetype(self, token_name: str) -> Dict:
        """Get archetype information for a token."""
        for category, category_archetypes in self.archetype_data.items():
            for archetype_name, archetype_info in category_archetypes.items():
                if token_name in archetype_info.get('tokens', []):
                    return {
                        'category': category,
                        'cluster': archetype_info.get('cluster_id', -1),
                        'archetype': archetype_name
                    }
        return {'category': 'unknown', 'cluster': -1, 'archetype': 'unknown'}
    
    def _analyze_post_minute5_pumps(self, prices: np.ndarray) -> Dict:
        """Analyze pumps that occur after minute 5."""
        if len(prices) <= 5:
            return {'pumped_50pct': False, 'max_pump': 0.0, 'pump_timing': None}
        
        # Prices from minute 5 onwards
        post_min5_prices = prices[5:]
        baseline_price = prices[4]  # Price at minute 5 as baseline
        
        # Calculate returns from minute 5 baseline
        post_min5_returns = (post_min5_prices - baseline_price) / baseline_price
        
        # Check for 50%+ pump
        pumped_50pct = np.any(post_min5_returns >= 0.5)
        max_pump = np.max(post_min5_returns) if len(post_min5_returns) > 0 else 0.0
        
        # Find timing of first 50%+ pump
        pump_timing = None
        if pumped_50pct:
            pump_indices = np.where(post_min5_returns >= 0.5)[0]
            if len(pump_indices) > 0:
                pump_timing = pump_indices[0] + 5  # Add 5 to account for minute 5 offset
        
        return {
            'pumped_50pct': pumped_50pct,
            'max_pump': max_pump,
            'pump_timing': pump_timing
        }
    
    def _calculate_time_to_15x(self, prices: np.ndarray) -> Dict:
        """Calculate time to reach 1.5x multiplier."""
        if len(prices) < 2:
            return {'time_minutes': None, 'achieved': False, 'max_multiplier': 1.0}
        
        initial_price = prices[0]
        multipliers = prices / initial_price
        
        # Check if 1.5x was achieved
        achieved_15x = np.any(multipliers >= 1.5)
        max_multiplier = np.max(multipliers)
        
        # Find time to 1.5x
        time_to_15x = None
        if achieved_15x:
            indices_15x = np.where(multipliers >= 1.5)[0]
            if len(indices_15x) > 0:
                time_to_15x = indices_15x[0]  # First occurrence
        
        return {
            'time_minutes': time_to_15x,
            'achieved': achieved_15x,
            'max_multiplier': max_multiplier
        }
    
    def _classify_volatility(self, cv: float) -> str:
        """Classify volatility into buckets."""
        if cv <= 0.8:
            return 'Low (≤0.8)'
        elif cv <= 1.5:
            return 'High (0.8-1.5)'
        elif cv <= 2.0:
            return 'Very High (1.5-2.0)'
        else:
            return 'Extreme (>2.0)'
    
    def _calculate_summary_metrics(self) -> Dict:
        """Calculate summary metrics to answer the key questions."""
        if self.pump_analysis.empty:
            return {}
        
        df = self.pump_analysis
        
        # Question 1: What % of high-vol (CV>0.8) marathons pump >50% after minute 5?
        high_vol_tokens = df[df['high_volatility']]
        high_vol_pump_rate = high_vol_tokens['pumped_50pct_after_min5'].mean() * 100 if len(high_vol_tokens) > 0 else 0
        
        # Question 2: What's the average time to 1.5x for those that pump?
        pumping_tokens = df[df['achieved_15x']]
        avg_time_to_15x = pumping_tokens['time_to_15x_minutes'].mean() if len(pumping_tokens) > 0 else 0
        
        # Additional insights
        volatility_breakdown = df.groupby('volatility_bucket').agg({
            'pumped_50pct_after_min5': ['count', 'sum', 'mean'],
            'time_to_15x_minutes': 'mean',
            'max_pump_after_min5': 'mean'
        }).round(3)
        
        summary = {
            'total_marathon_tokens': len(df),
            'high_volatility_tokens': len(high_vol_tokens),
            'high_vol_pump_rate': high_vol_pump_rate,
            'avg_time_to_15x_minutes': avg_time_to_15x,
            'overall_pump_rate': df['pumped_50pct_after_min5'].mean() * 100,
            'overall_15x_rate': df['achieved_15x'].mean() * 100,
            'volatility_breakdown': volatility_breakdown
        }
        
        return summary
    
    def analyze_misclassification(self, classifier_results_path: Optional[Path] = None) -> Dict:
        """Analyze misclassification of true marathons."""
        print(f"🔍 Analyzing marathon misclassification...")
        
        # For now, analyze based on lifespan thresholds since we don't have classifier predictions
        # True marathons should have 1200+ minutes lifespan
        
        if self.pump_analysis.empty:
            return {}
        
        df = self.pump_analysis
        
        # Classify based on actual lifespan
        df['true_category'] = df['lifespan_minutes'].apply(
            lambda x: 'marathon' if x >= 1200 else 'standard'
        )
        
        # Count misclassifications
        true_marathons = df[df['true_category'] == 'marathon']
        classified_as_standard = true_marathons[true_marathons['category'] == 'standard']
        
        misclassification_rate = len(classified_as_standard) / len(true_marathons) * 100 if len(true_marathons) > 0 else 0
        
        # Analyze borderline cases (1150-1250 minutes)
        borderline_tokens = df[(df['lifespan_minutes'] >= 1150) & (df['lifespan_minutes'] <= 1250)]
        
        self.misclassification_analysis = {
            'total_true_marathons': len(true_marathons),
            'misclassified_as_standard': len(classified_as_standard),
            'misclassification_rate': misclassification_rate,
            'borderline_tokens': len(borderline_tokens),
            'borderline_misclassified': len(borderline_tokens[borderline_tokens['category'] == 'standard'])
        }
        
        print(f"📊 True marathons (1200+ min): {len(true_marathons)}")
        print(f"❌ Misclassified as standard: {len(classified_as_standard)} ({misclassification_rate:.1f}%)")
        
        return self.misclassification_analysis
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create interactive HTML visualizations."""
        print(f"📊 Creating marathon pump analysis visualizations...")
        
        if self.pump_analysis.empty:
            print(f"❌ No pump analysis data available")
            return {}
        
        figures = {}
        df = self.pump_analysis
        
        # 1. High-Volatility Pump Analysis
        figures['volatility_pump_analysis'] = self._create_volatility_pump_chart(df)
        
        # 2. Time-to-Pump Distribution
        figures['time_to_pump_distribution'] = self._create_time_to_pump_chart(df)
        
        # 3. Misclassification Analysis
        figures['misclassification_analysis'] = self._create_misclassification_chart(df)
        
        # 4. Comprehensive Pump Dashboard
        figures['pump_success_dashboard'] = self._create_pump_dashboard(df)
        
        # 5. Marathon Deep Dive
        figures['marathon_deep_dive'] = self._create_marathon_deep_dive(df)
        
        return figures
    
    def _create_volatility_pump_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create volatility vs pump success analysis chart."""
        # Group by volatility bucket
        vol_analysis = df.groupby('volatility_bucket').agg({
            'pumped_50pct_after_min5': ['count', 'sum'],
            'high_volatility': 'first'
        }).round(3)
        
        vol_analysis.columns = ['total_tokens', 'pumped_tokens', 'is_high_vol']
        vol_analysis['pump_rate'] = (vol_analysis['pumped_tokens'] / vol_analysis['total_tokens'] * 100).round(1)
        vol_analysis = vol_analysis.reset_index()
        
        fig = go.Figure()
        
        # Bar chart of pump rates
        fig.add_trace(go.Bar(
            x=vol_analysis['volatility_bucket'],
            y=vol_analysis['pump_rate'],
            text=[f"{rate}%<br>({pumped}/{total})" for rate, pumped, total in 
                  zip(vol_analysis['pump_rate'], vol_analysis['pumped_tokens'], vol_analysis['total_tokens'])],
            textposition='auto',
            marker_color=['red' if not high_vol else 'green' for high_vol in vol_analysis['is_high_vol']],
            hovertemplate='<b>%{x}</b><br>Pump Rate: %{y}%<br>Pumped: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>',
            customdata=list(zip(vol_analysis['pumped_tokens'], vol_analysis['total_tokens']))
        ))
        
        fig.update_layout(
            title='Marathon Token Pump Success Rate by Volatility<br><sub>% pumping >50% after minute 5</sub>',
            xaxis_title='Volatility Bucket (CV)',
            yaxis_title='Pump Success Rate (%)',
            yaxis={'range': [0, max(vol_analysis['pump_rate']) * 1.1]},
            annotations=[
                dict(
                    text=f"High volatility (CV>0.8) pump rate: {df[df['high_volatility']]['pumped_50pct_after_min5'].mean()*100:.1f}%",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor='left', yanchor='top',
                    showarrow=False,
                    font=dict(size=12, color="blue"),
                    bgcolor="rgba(255,255,255,0.8)"
                )
            ]
        )
        
        return fig
    
    def _create_time_to_pump_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create time-to-1.5x distribution analysis."""
        # Filter tokens that achieved 1.5x
        pumped_tokens = df[df['achieved_15x']].copy()
        
        if pumped_tokens.empty:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No tokens achieved 1.5x multiplier",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=16)
            )
            fig.update_layout(title="Time to 1.5x Analysis - No Data Available")
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Time to 1.5x Distribution',
                'Time to 1.5x by Volatility',
                'Success Rate Over Time',
                'Average Time by Archetype'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=pumped_tokens['time_to_15x_minutes'],
                nbinsx=20,
                name='Time to 1.5x',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Time by volatility bucket
        time_by_vol = pumped_tokens.groupby('volatility_bucket')['time_to_15x_minutes'].agg(['mean', 'count']).reset_index()
        fig.add_trace(
            go.Bar(
                x=time_by_vol['volatility_bucket'],
                y=time_by_vol['mean'],
                text=[f"{mean:.1f}min<br>(n={count})" for mean, count in zip(time_by_vol['mean'], time_by_vol['count'])],
                textposition='auto',
                name='Avg Time to 1.5x',
                marker_color='orange'
            ),
            row=1, col=2
        )
        
        # 3. Success rate over time (cumulative)
        time_bins = np.arange(0, pumped_tokens['time_to_15x_minutes'].max() + 10, 10)
        success_rates = []
        for time_threshold in time_bins:
            rate = (pumped_tokens['time_to_15x_minutes'] <= time_threshold).mean() * 100
            success_rates.append(rate)
        
        fig.add_trace(
            go.Scatter(
                x=time_bins,
                y=success_rates,
                mode='lines+markers',
                name='Cumulative Success Rate',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
        
        # 4. Time by archetype
        if 'archetype' in pumped_tokens.columns:
            time_by_archetype = pumped_tokens.groupby('archetype')['time_to_15x_minutes'].agg(['mean', 'count']).reset_index()
            fig.add_trace(
                go.Bar(
                    x=time_by_archetype['archetype'],
                    y=time_by_archetype['mean'],
                    text=[f"{mean:.1f}min<br>(n={count})" for mean, count in zip(time_by_archetype['mean'], time_by_archetype['count'])],
                    textposition='auto',
                    name='Time by Archetype',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        avg_time = pumped_tokens['time_to_15x_minutes'].mean()
        median_time = pumped_tokens['time_to_15x_minutes'].median()
        
        fig.update_layout(
            title=f'Time to 1.5x Analysis<br><sub>Average: {avg_time:.1f} minutes | Median: {median_time:.1f} minutes | Success Rate: {len(pumped_tokens)/len(df)*100:.1f}%</sub>',
            showlegend=False,
            height=800
        )
        
        return fig
    
    def _create_misclassification_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create misclassification analysis visualization."""
        # Add true category based on lifespan
        df_copy = df.copy()
        df_copy['true_category'] = df_copy['lifespan_minutes'].apply(
            lambda x: 'marathon' if x >= 1200 else 'standard'
        )
        
        # Create confusion matrix
        confusion_data = pd.crosstab(df_copy['true_category'], df_copy['category'], margins=True)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'True vs Predicted Categories',
                'Misclassification by Lifespan',
                'Borderline Cases (1150-1250 min)',
                'Classification Accuracy Metrics'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Confusion matrix heatmap
        if 'marathon' in confusion_data.index and 'marathon' in confusion_data.columns:
            confusion_matrix = confusion_data.drop('All').drop('All', axis=1)
            
            fig.add_trace(
                go.Heatmap(
                    z=confusion_matrix.values,
                    x=confusion_matrix.columns,
                    y=confusion_matrix.index,
                    text=confusion_matrix.values,
                    texttemplate="%{text}",
                    colorscale='Blues',
                    hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Misclassification by lifespan
        lifespan_bins = pd.cut(df_copy['lifespan_minutes'], bins=20)
        misclass_by_lifespan = df_copy.groupby(lifespan_bins).apply(
            lambda x: (x['true_category'] != x['category']).mean() * 100
        ).reset_index()
        misclass_by_lifespan['lifespan_midpoint'] = misclass_by_lifespan['lifespan_minutes'].apply(lambda x: x.mid)
        
        fig.add_trace(
            go.Scatter(
                x=misclass_by_lifespan['lifespan_midpoint'],
                y=misclass_by_lifespan[0],
                mode='lines+markers',
                name='Misclassification Rate',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Add vertical line at 1200 minutes threshold
        fig.add_vline(x=1200, line_dash="dash", line_color="green", row=1, col=2)
        
        # 3. Borderline cases
        borderline = df_copy[(df_copy['lifespan_minutes'] >= 1150) & (df_copy['lifespan_minutes'] <= 1250)]
        if not borderline.empty:
            borderline_counts = borderline.groupby(['true_category', 'category']).size().reset_index(name='count')
            
            fig.add_trace(
                go.Bar(
                    x=[f"{row['true_category']} → {row['category']}" for _, row in borderline_counts.iterrows()],
                    y=borderline_counts['count'],
                    text=borderline_counts['count'],
                    textposition='auto',
                    name='Borderline Classifications',
                    marker_color='yellow'
                ),
                row=2, col=1
            )
        
        # 4. Accuracy metrics
        true_marathons = df_copy[df_copy['true_category'] == 'marathon']
        if not true_marathons.empty:
            correct_classifications = true_marathons[true_marathons['category'] == 'marathon']
            accuracy = len(correct_classifications) / len(true_marathons) * 100
            
            metrics_text = f"""
            True Marathons: {len(true_marathons)}
            Correctly Classified: {len(correct_classifications)}
            Accuracy: {accuracy:.1f}%
            Misclassification Rate: {100-accuracy:.1f}%
            """
            
            fig.add_annotation(
                text=metrics_text,
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                row=2, col=2
            )
        
        fig.update_layout(
            title='Marathon Token Misclassification Analysis<br><sub>True Category (1200+ min lifespan) vs Predicted Category</sub>',
            height=800
        )
        
        return fig
    
    def _create_pump_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive pump success dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Pump Success by Volatility & Archetype',
                'Time to Pump vs Volatility',
                'Maximum Pump Magnitude',
                'Pump Success Over Token Lifespan',
                'Pump Timing Distribution',
                'Success Rate Summary'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Use different colors for different archetypes
        colors = px.colors.qualitative.Set1
        
        # 1. Pump success by volatility and archetype
        for i, archetype in enumerate(df['archetype'].unique()):
            if archetype == 'unknown':
                continue
                
            archetype_data = df[df['archetype'] == archetype]
            vol_success = archetype_data.groupby('volatility_bucket')['pumped_50pct_after_min5'].mean() * 100
            
            fig.add_trace(
                go.Bar(
                    x=vol_success.index,
                    y=vol_success.values,
                    name=archetype,
                    marker_color=colors[i % len(colors)],
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # 2. Time to pump vs volatility (scatter)
        pumped_tokens = df[df['achieved_15x']]
        if not pumped_tokens.empty:
            fig.add_trace(
                go.Scatter(
                    x=pumped_tokens['volatility_cv'],
                    y=pumped_tokens['time_to_15x_minutes'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=pumped_tokens['max_multiplier'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Max Multiplier")
                    ),
                    text=pumped_tokens['token_name'],
                    name='Time to 1.5x',
                    hovertemplate='<b>%{text}</b><br>Volatility CV: %{x:.3f}<br>Time to 1.5x: %{y} min<br>Max Multiplier: %{marker.color:.2f}x<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Maximum pump magnitude distribution
        fig.add_trace(
            go.Histogram(
                x=df['max_pump_after_min5'] * 100,  # Convert to percentage
                nbinsx=30,
                name='Max Pump %',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # 4. Pump success over token lifespan
        lifespan_bins = pd.cut(df['lifespan_minutes'], bins=15)
        pump_by_lifespan = df.groupby(lifespan_bins)['pumped_50pct_after_min5'].mean() * 100
        lifespan_midpoints = [interval.mid for interval in pump_by_lifespan.index]
        
        fig.add_trace(
            go.Scatter(
                x=lifespan_midpoints,
                y=pump_by_lifespan.values,
                mode='lines+markers',
                name='Pump Success Rate',
                line=dict(color='blue', width=3)
            ),
            row=2, col=2
        )
        
        # 5. Pump timing distribution
        pumped_with_timing = df[df['pump_timing_min5'].notna()]
        if not pumped_with_timing.empty:
            fig.add_trace(
                go.Histogram(
                    x=pumped_with_timing['pump_timing_min5'],
                    nbinsx=20,
                    name='Pump Timing',
                    marker_color='orange'
                ),
                row=3, col=1
            )
        
        # 6. Summary metrics
        summary_metrics = f"""
        Total Marathon Tokens: {len(df):,}
        High Volatility (CV>0.8): {len(df[df['high_volatility']]):,}
        Pumped >50% after min 5: {df['pumped_50pct_after_min5'].sum():,}
        Overall Pump Rate: {df['pumped_50pct_after_min5'].mean()*100:.1f}%
        High-Vol Pump Rate: {df[df['high_volatility']]['pumped_50pct_after_min5'].mean()*100:.1f}%
        Avg Time to 1.5x: {df[df['achieved_15x']]['time_to_15x_minutes'].mean():.1f} min
        """
        
        fig.add_annotation(
            text=summary_metrics,
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            row=3, col=2
        )
        
        fig.update_layout(
            title='Marathon Token Pump Analysis Dashboard',
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def _create_marathon_deep_dive(self, df: pd.DataFrame) -> go.Figure:
        """Create detailed marathon token analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Volatility vs Total Return',
                'Pump Success by Cluster',
                'Risk-Reward Analysis',
                'Token Performance Heatmap'
            ]
        )
        
        # 1. Volatility vs Total Return scatter
        fig.add_trace(
            go.Scatter(
                x=df['volatility_cv'],
                y=df['total_return'] * 100,  # Convert to percentage
                mode='markers',
                marker=dict(
                    size=8,
                    color=['green' if pumped else 'red' for pumped in df['pumped_50pct_after_min5']],
                    opacity=0.6
                ),
                text=df['archetype'],
                name='Marathon Tokens',
                hovertemplate='<b>%{text}</b><br>Volatility CV: %{x:.3f}<br>Total Return: %{y:.1f}%<br>Pumped >50%: %{marker.color}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Pump success by cluster
        cluster_success = df.groupby('cluster').agg({
            'pumped_50pct_after_min5': ['count', 'sum', 'mean'],
            'volatility_cv': 'mean'
        }).round(3)
        
        cluster_success.columns = ['total', 'pumped', 'success_rate', 'avg_volatility']
        cluster_success = cluster_success.reset_index()
        
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {cluster}" for cluster in cluster_success['cluster']],
                y=cluster_success['success_rate'] * 100,
                text=[f"{rate*100:.1f}%<br>({pumped}/{total})" for rate, pumped, total in 
                      zip(cluster_success['success_rate'], cluster_success['pumped'], cluster_success['total'])],
                textposition='auto',
                name='Success Rate by Cluster',
                marker_color='skyblue'
            ),
            row=1, col=2
        )
        
        # 3. Risk-Reward Analysis
        # Create risk-reward buckets
        df_copy = df.copy()
        df_copy['risk_bucket'] = pd.cut(df_copy['volatility_cv'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df_copy['reward_bucket'] = pd.cut(df_copy['total_return'], bins=5, labels=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'])
        
        risk_reward = df_copy.groupby(['risk_bucket', 'reward_bucket']).size().reset_index(name='count')
        
        if not risk_reward.empty:
            # Create a pivot table for heatmap
            heatmap_data = risk_reward.pivot(index='risk_bucket', columns='reward_bucket', values='count')
            heatmap_data = heatmap_data.fillna(0)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    text=heatmap_data.values,
                    texttemplate="%{text}",
                    colorscale='YlOrRd',
                    hovertemplate='Risk: %{y}<br>Reward: %{x}<br>Count: %{z}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Performance metrics summary
        performance_summary = f"""
        📊 MARATHON TOKEN ANALYSIS SUMMARY
        
        🎯 KEY ANSWERS:
        1. High-vol (CV>0.8) pump rate: {df[df['high_volatility']]['pumped_50pct_after_min5'].mean()*100:.1f}%
        2. Avg time to 1.5x: {df[df['achieved_15x']]['time_to_15x_minutes'].mean():.1f} minutes
        3. Misclassification rate: {self.misclassification_analysis.get('misclassification_rate', 0):.1f}%
        
        📈 DETAILED METRICS:
        • Total marathon tokens analyzed: {len(df):,}
        • High volatility tokens (CV>0.8): {len(df[df['high_volatility']]):,}
        • Tokens achieving 1.5x: {len(df[df['achieved_15x']]):,}
        • Overall pump success: {df['pumped_50pct_after_min5'].mean()*100:.1f}%
        • Median volatility CV: {df['volatility_cv'].median():.3f}
        • Max pump observed: {df['max_pump_after_min5'].max()*100:.1f}%
        """
        
        fig.add_annotation(
            text=performance_summary,
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=10, family="monospace"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            row=2, col=2
        )
        
        fig.update_layout(
            title='Marathon Token Deep Dive Analysis<br><sub>Comprehensive performance and risk analysis</sub>',
            height=800
        )
        
        return fig
    
    def save_results(self, output_dir: Path = None) -> str:
        """Save analysis results and visualizations."""
        if output_dir is None:
            output_dir = self.results_dir / "marathon_pump_analysis"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis data
        if not self.pump_analysis.empty:
            analysis_path = output_dir / f"marathon_pump_data_{timestamp}.csv"
            self.pump_analysis.to_csv(analysis_path, index=False)
            print(f"📊 Analysis data saved: {analysis_path}")
        
        # Save summary metrics
        summary_metrics = self._calculate_summary_metrics()
        summary_path = output_dir / f"marathon_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            serializable_metrics = {}
            for key, value in summary_metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                elif hasattr(value, 'to_dict'):  # pandas DataFrame
                    # Skip complex DataFrame structures that can't be JSON serialized
                    serializable_metrics[key] = f"DataFrame with shape {value.shape}"
                elif isinstance(value, (list, tuple, set)):
                    serializable_metrics[key] = list(value)
                else:
                    try:
                        json.dumps(value)  # Test if it's JSON serializable
                        serializable_metrics[key] = value
                    except (TypeError, ValueError):
                        serializable_metrics[key] = str(value)
            
            json.dump(serializable_metrics, f, indent=2, default=str)
        print(f"📋 Summary metrics saved: {summary_path}")
        
        # Create and save visualizations
        figures = self.create_visualizations()
        for fig_name, fig in figures.items():
            fig_path = output_dir / f"{fig_name}_{timestamp}.html"
            fig.write_html(fig_path)
            print(f"📈 Visualization saved: {fig_path}")
        
        print(f"✅ All results saved to: {output_dir}")
        return timestamp
    
    def run_complete_analysis(self, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete marathon pump analysis."""
        print(f"🚀 Starting Marathon Pump Analysis")
        
        # Load data
        self.load_archetype_results(archetype_results_path)
        self.load_token_data(data_dir)
        
        if not self.token_data:
            print(f"❌ No marathon token data loaded")
            return {}
        
        # Analyze pump patterns
        summary_metrics = self.analyze_pump_patterns()
        
        # Analyze misclassification
        misclass_metrics = self.analyze_misclassification()
        
        # Save results
        timestamp = self.save_results()
        
        # Print key answers
        print(f"\n🎯 KEY ANSWERS TO YOUR QUESTIONS:")
        print(f"=" * 60)
        
        if summary_metrics:
            print(f"1. High-vol (CV>0.8) marathon pump rate: {summary_metrics.get('high_vol_pump_rate', 0):.1f}%")
            print(f"2. Average time to 1.5x: {summary_metrics.get('avg_time_to_15x_minutes', 0):.1f} minutes")
        
        if misclass_metrics:
            print(f"3. Marathon misclassification rate: {misclass_metrics.get('misclassification_rate', 0):.1f}%")
        
        print(f"\n📊 ADDITIONAL INSIGHTS:")
        if summary_metrics:
            print(f"• Total marathon tokens analyzed: {summary_metrics.get('total_marathon_tokens', 0):,}")
            print(f"• Overall pump success rate: {summary_metrics.get('overall_pump_rate', 0):.1f}%")
            print(f"• Overall 1.5x achievement rate: {summary_metrics.get('overall_15x_rate', 0):.1f}%")
        
        return {
            'summary_metrics': summary_metrics,
            'misclassification_metrics': misclass_metrics,
            'timestamp': timestamp,
            'total_tokens_analyzed': len(self.pump_analysis)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Marathon Pump Analysis")
    parser.add_argument("--archetype-results", type=Path,
                       help="Path to archetype characterization results JSON")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../data/with_archetypes_fixed"),
                       help="Path to token data directory")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Find latest archetype results if not specified
    if not args.archetype_results:
        results_dir = Path("../results/phase1_day9_10_archetypes")
        if results_dir.exists():
            json_files = list(results_dir.glob("archetype_characterization_*.json"))
            if json_files:
                args.archetype_results = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest archetype results: {args.archetype_results}")
            else:
                print("❌ No archetype results found. Run the Phase 1 pipeline first.")
                return
        else:
            print("❌ Results directory not found. Run the Phase 1 pipeline first.")
            return
    
    # Initialize analyzer
    analyzer = MarathonPumpAnalyzer(args.output_dir)
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(args.archetype_results, args.data_dir)
        
        print(f"\n🎉 Marathon pump analysis complete!")
        print(f"📈 Results saved with timestamp: {results.get('timestamp', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()