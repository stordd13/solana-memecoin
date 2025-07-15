#!/usr/bin/env python3
"""
Archetype Metrics Analysis

Analyzes the discovered behavioral archetypes to understand:
- Volatility patterns per archetype
- Return patterns per archetype  
- Lifecycle characteristics
- Statistical comparisons across archetypes

Usage:
    python archetype_metrics.py --results PATH_TO_ARCHETYPE_RESULTS
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_categorized_tokens


class ArchetypeMetricsAnalyzer:
    """Analyze metrics and characteristics of discovered behavioral archetypes."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("../results")
        self.archetype_data = None
        self.token_metrics = {}
        self.archetype_metrics = {}
        
    def load_archetype_results(self, archetype_results_path: Path) -> Dict:
        """Load archetype characterization results."""
        print(f"📊 Loading archetype results from: {archetype_results_path}")
        
        with open(archetype_results_path, 'r') as f:
            results = json.load(f)
        
        self.archetype_data = results.get('archetype_data', {})
        
        # Count actual archetypes
        total_archetypes = 0
        for category, category_archetypes in self.archetype_data.items():
            total_archetypes += len(category_archetypes)
        
        print(f"📈 Found {total_archetypes} archetypes across {len(self.archetype_data)} categories")
        
        # Print summary
        for category, category_archetypes in self.archetype_data.items():
            print(f"  {category}: {len(category_archetypes)} archetypes")
            for archetype_name, archetype_info in category_archetypes.items():
                token_count = archetype_info.get('size', len(archetype_info.get('tokens', [])))
                print(f"    {archetype_name}: {token_count} tokens")
        
        return results
    
    def load_sprint_tokens(self) -> List[str]:
        """Load sprint tokens from stability results (excluded from archetype analysis)."""
        print(f"📊 Loading sprint tokens from stability results...")
        
        # Try to find the latest stability results file
        stability_dir = Path("../results/phase1_day7_8_stability")
        if not stability_dir.exists():
            print(f"❌ Stability results directory not found: {stability_dir}")
            return []
        
        # Find the latest stability results
        json_files = list(stability_dir.glob("stability_testing_*.json"))
        if not json_files:
            print(f"❌ No stability results found in {stability_dir}")
            return []
            
        latest_stability_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Using stability results: {latest_stability_file}")
        
        try:
            with open(latest_stability_file, 'r') as f:
                stability_results = json.load(f)
            
            # Extract sprint tokens from stability results
            sprint_info = stability_results.get('category_stability_results', {}).get('sprint', {})
            sprint_tokens = sprint_info.get('token_names', [])
            
            print(f"📈 Found {len(sprint_tokens)} sprint tokens")
            return sprint_tokens
            
        except Exception as e:
            print(f"❌ Error loading stability results: {e}")
            return []
    
    def load_token_data(self, data_dir: Path) -> Dict[str, pl.DataFrame]:
        """Load raw token data for metrics calculation."""
        print(f"📁 Loading token data from: {data_dir}")
        
        all_tokens = {}
        
        # Try both directory structures: 
        # 1. New simplified structure (all files in one directory)
        # 2. Old structure (files in subdirectories)
        
        # Check if files are directly in data_dir (new structure - with_archetypes_fixed)
        direct_files = list(data_dir.glob("*.parquet"))
        
        if direct_files:
            print(f"  📁 Using simplified directory structure (with archetype labels)")
            print(f"  📁 Loading tokens directly from: {data_dir}")
            
            for token_file in direct_files:
                token_name = token_file.stem
                try:
                    df = pl.read_parquet(token_file)
                    all_tokens[token_name] = df
                except Exception as e:
                    print(f"    Warning: Failed to load {token_name}: {e}")
        else:
            # Fall back to old directory structure
            print(f"  📁 Using old directory structure with subdirectories")
            
            # Use the actual directory structure: dead_tokens, normal_behavior_tokens, tokens_with_extremes, tokens_with_gaps
            for category_name in ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes', 'tokens_with_gaps']:
                category_path = data_dir / category_name
                if category_path.exists():
                    print(f"  Loading {category_name} tokens...")
                    
                    token_files = list(category_path.glob("*.parquet"))
                    for token_file in token_files:
                        token_name = token_file.stem
                        try:
                            df = pl.read_parquet(token_file)
                            all_tokens[token_name] = df
                        except Exception as e:
                            print(f"    Warning: Failed to load {token_name}: {e}")
        
        print(f"📊 Loaded {len(all_tokens)} tokens total")
        return all_tokens
    
    def calculate_token_metrics(self, token_data: Dict[str, pl.DataFrame]) -> Dict[str, Dict]:
        """Calculate metrics for each token."""
        print(f"⚡ Calculating metrics for {len(token_data)} tokens...")
        
        metrics = {}
        
        for token_name, df in token_data.items():
            try:
                # Convert to pandas for easier calculation
                pdf = df.to_pandas()
                
                if len(pdf) < 2:
                    continue
                
                # Basic metrics
                prices = pdf['price'].values
                returns = pdf['price'].pct_change().dropna().values
                
                # Calculate metrics
                token_metrics = {
                    'token_name': token_name,
                    'lifespan_minutes': len(pdf),
                    'total_return': (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0,
                    'volatility_cv': np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0,
                    'returns_mean': np.mean(returns) if len(returns) > 0 else 0,
                    'returns_std': np.std(returns) if len(returns) > 0 else 0,
                    'returns_skew': stats.skew(returns) if len(returns) > 2 else 0,
                    'returns_kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0,
                    'max_price': np.max(prices),
                    'min_price': np.min(prices),
                    'price_range': (np.max(prices) - np.min(prices)) / np.mean(prices) if np.mean(prices) > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(prices),
                    'time_to_peak': np.argmax(prices) if len(prices) > 0 else 0,
                    'time_to_peak_pct': (np.argmax(prices) / len(prices)) * 100 if len(prices) > 0 else 0
                }
                
                metrics[token_name] = token_metrics
                
            except Exception as e:
                print(f"    Warning: Failed to calculate metrics for {token_name}: {e}")
                continue
        
        print(f"✅ Calculated metrics for {len(metrics)} tokens")
        return metrics
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0
        
        # Calculate cumulative maximum
        cummax = np.maximum.accumulate(prices)
        # Calculate drawdown
        drawdown = (prices - cummax) / cummax
        # Return maximum drawdown (most negative)
        return np.min(drawdown)
    
    def aggregate_archetype_metrics(self) -> Dict[str, Dict]:
        """Aggregate metrics by archetype."""
        print(f"📊 Aggregating metrics by archetype...")
        
        archetype_metrics = {}
        
        for category, category_archetypes in self.archetype_data.items():
            for archetype_name, archetype_info in category_archetypes.items():
                tokens = archetype_info.get('tokens', [])
                
                # Get metrics for tokens in this archetype
                token_metrics_list = []
                for token_name in tokens:
                    if token_name in self.token_metrics:
                        token_metrics_list.append(self.token_metrics[token_name])
                
                if not token_metrics_list:
                    continue
                
                # Convert to DataFrame for easier aggregation
                df = pd.DataFrame(token_metrics_list)
                
                # Calculate aggregated metrics with robust statistics
                aggregated = {
                    'archetype_name': archetype_name,
                    'category': category,
                    'cluster_id': archetype_info.get('cluster_id', 0),
                    'token_count': len(tokens),
                    'valid_metrics_count': len(token_metrics_list),
                    
                    # Lifespan metrics
                    'avg_lifespan_minutes': df['lifespan_minutes'].mean(),
                    'median_lifespan_minutes': df['lifespan_minutes'].median(),
                    'std_lifespan_minutes': df['lifespan_minutes'].std(),
                    'min_lifespan_minutes': df['lifespan_minutes'].min(),
                    'max_lifespan_minutes': df['lifespan_minutes'].max(),
                    
                    # Return metrics - ROBUST STATISTICS (median-based to handle outliers)
                    'avg_total_return': df['total_return'].median(),  # Use median instead of mean
                    'mean_total_return': df['total_return'].mean(),   # Keep mean for comparison
                    'median_total_return': df['total_return'].median(),
                    'std_total_return': df['total_return'].std(),
                    'min_total_return': df['total_return'].min(),
                    'max_total_return': df['total_return'].max(),
                    'return_p25': df['total_return'].quantile(0.25),  # 25th percentile
                    'return_p75': df['total_return'].quantile(0.75),  # 75th percentile
                    'return_p90': df['total_return'].quantile(0.90),  # 90th percentile
                    'return_p95': df['total_return'].quantile(0.95),  # 95th percentile
                    'return_p99': df['total_return'].quantile(0.99),  # 99th percentile
                    'positive_return_rate': (df['total_return'] > 0).mean(),
                    'winsorized_mean_return': df['total_return'].clip(
                        lower=df['total_return'].quantile(0.05), 
                        upper=df['total_return'].quantile(0.95)
                    ).mean(),  # Winsorized mean (5%-95%)
                    
                    # Volatility metrics
                    'avg_volatility_cv': df['volatility_cv'].mean(),
                    'median_volatility_cv': df['volatility_cv'].median(),
                    'std_volatility_cv': df['volatility_cv'].std(),
                    'high_volatility_rate': (df['volatility_cv'] > df['volatility_cv'].quantile(0.75)).mean(),
                    
                    # Returns distribution
                    'avg_returns_mean': df['returns_mean'].mean(),
                    'avg_returns_std': df['returns_std'].mean(),
                    'avg_returns_skew': df['returns_skew'].mean(),
                    'avg_returns_kurtosis': df['returns_kurtosis'].mean(),
                    
                    # Price pattern metrics
                    'avg_price_range': df['price_range'].mean(),
                    'avg_max_drawdown': df['max_drawdown'].mean(),
                    'avg_time_to_peak_pct': df['time_to_peak_pct'].mean(),
                    
                    # Risk metrics
                    'sharpe_ratio': df['returns_mean'].mean() / df['returns_std'].mean() if df['returns_std'].mean() > 0 else 0,
                    'extreme_loss_rate': (df['total_return'] < -0.9).mean(),  # 90%+ loss rate
                    'extreme_gain_rate': (df['total_return'] > 10).mean(),    # 1000%+ gain rate
                }
                
                archetype_metrics[archetype_name] = aggregated
        
        print(f"📈 Aggregated metrics for {len(archetype_metrics)} archetypes")
        return archetype_metrics
    
    def calculate_sprint_metrics(self, sprint_tokens: List[str]) -> Dict:
        """Calculate metrics for sprint tokens (excluded from archetype analysis)."""
        print(f"📊 Calculating sprint metrics for {len(sprint_tokens)} tokens...")
        
        # Get metrics for sprint tokens
        sprint_metrics_list = []
        for token_name in sprint_tokens:
            if token_name in self.token_metrics:
                sprint_metrics_list.append(self.token_metrics[token_name])
        
        if not sprint_metrics_list:
            print(f"❌ No sprint token metrics found")
            return {}
        
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(sprint_metrics_list)
        
        # Calculate aggregated metrics with robust statistics
        sprint_aggregated = {
            'archetype_name': 'sprint_unclustered',
            'category': 'sprint',
            'cluster_id': 'unclustered',
            'token_count': len(sprint_tokens),
            'valid_metrics_count': len(sprint_metrics_list),
            
            # Lifespan metrics
            'avg_lifespan_minutes': df['lifespan_minutes'].mean(),
            'median_lifespan_minutes': df['lifespan_minutes'].median(),
            'std_lifespan_minutes': df['lifespan_minutes'].std(),
            'min_lifespan_minutes': df['lifespan_minutes'].min(),
            'max_lifespan_minutes': df['lifespan_minutes'].max(),
            
            # Return metrics - ROBUST STATISTICS (median-based to handle outliers)
            'avg_total_return': df['total_return'].median(),  # Use median instead of mean
            'mean_total_return': df['total_return'].mean(),   # Keep mean for comparison
            'median_total_return': df['total_return'].median(),
            'std_total_return': df['total_return'].std(),
            'min_total_return': df['total_return'].min(),
            'max_total_return': df['total_return'].max(),
            'return_p25': df['total_return'].quantile(0.25),  # 25th percentile
            'return_p75': df['total_return'].quantile(0.75),  # 75th percentile
            'return_p90': df['total_return'].quantile(0.90),  # 90th percentile
            'return_p95': df['total_return'].quantile(0.95),  # 95th percentile
            'return_p99': df['total_return'].quantile(0.99),  # 99th percentile
            'positive_return_rate': (df['total_return'] > 0).mean(),
            'winsorized_mean_return': df['total_return'].clip(
                lower=df['total_return'].quantile(0.05), 
                upper=df['total_return'].quantile(0.95)
            ).mean(),  # Winsorized mean (5%-95%)
            
            # Volatility metrics
            'avg_volatility_cv': df['volatility_cv'].mean(),
            'median_volatility_cv': df['volatility_cv'].median(),
            'std_volatility_cv': df['volatility_cv'].std(),
            'high_volatility_rate': (df['volatility_cv'] > df['volatility_cv'].quantile(0.75)).mean(),
            
            # Returns distribution
            'avg_returns_mean': df['returns_mean'].mean(),
            'avg_returns_std': df['returns_std'].mean(),
            'avg_returns_skew': df['returns_skew'].mean(),
            'avg_returns_kurtosis': df['returns_kurtosis'].mean(),
            
            # Price pattern metrics
            'avg_price_range': df['price_range'].mean(),
            'avg_max_drawdown': df['max_drawdown'].mean(),
            'avg_time_to_peak_pct': df['time_to_peak_pct'].mean(),
            
            # Risk metrics
            'sharpe_ratio': df['returns_mean'].mean() / df['returns_std'].mean() if df['returns_std'].mean() > 0 else 0,
            'extreme_loss_rate': (df['total_return'] < -0.9).mean(),  # 90%+ loss rate
            'extreme_gain_rate': (df['total_return'] > 10).mean(),    # 1000%+ gain rate
            'mega_pump_rate': (df['total_return'] > 1000).mean(),     # 100,000%+ gain rate (extreme pumps)
            'billion_pump_rate': (df['total_return'] > 1000000).mean(), # 100,000,000%+ gain rate (billion% pumps)
        }
        
        print(f"✅ Sprint metrics calculated for {len(sprint_metrics_list)} tokens")
        print(f"📊 Sprint extreme statistics:")
        print(f"   Median return: {sprint_aggregated['median_total_return']*100:.2f}%")
        print(f"   Mean return: {sprint_aggregated['mean_total_return']*100:.2f}%")
        print(f"   P99 return: {sprint_aggregated['return_p99']*100:.2f}%")
        print(f"   Max return: {sprint_aggregated['max_total_return']*100:.2f}%")
        print(f"   Mega pump rate: {sprint_aggregated['mega_pump_rate']*100:.1f}%")
        print(f"   Billion pump rate: {sprint_aggregated['billion_pump_rate']*100:.1f}%")
        
        return sprint_aggregated
    
    def generate_archetype_comparison(self) -> pd.DataFrame:
        """Generate comparison table of archetype metrics."""
        print(f"📋 Generating archetype comparison table...")
        
        comparison_data = []
        
        for archetype_name, metrics in self.archetype_metrics.items():
            row = {
                'Archetype': archetype_name,
                'Category': metrics['category'],
                'Cluster': metrics['cluster_id'],
                'Token Count': metrics['token_count'],
                'Avg Lifespan (min)': f"{metrics['avg_lifespan_minutes']:.0f}",
                'Median Return (%)': f"{metrics['avg_total_return']*100:.2f}",  # Now using median, fix display
                'Min Return (%)': f"{metrics['min_total_return']*100:.2f}",  # Worst case
                'Max Return (%)': f"{metrics['max_total_return']*100:.2f}",  # Best case (might be extreme)
                'Mean Return (%)': f"{metrics['mean_total_return']*100:.2f}",   # Show mean for comparison
                'Winsorized Return (%)': f"{metrics['winsorized_mean_return']*100:.2f}",  # Outlier-resistant
                'Return P95 (%)': f"{metrics['return_p95']*100:.2f}",  # 95th percentile
                'Return P99 (%)': f"{metrics['return_p99']*100:.2f}",  # 99th percentile
                'Avg Volatility (CV)': f"{metrics['avg_volatility_cv']:.3f}",
                'Positive Return Rate (%)': f"{metrics['positive_return_rate']*100:.1f}",
                'Extreme Loss Rate (%)': f"{metrics['extreme_loss_rate']*100:.1f}",
                'Extreme Gain Rate (%)': f"{metrics['extreme_gain_rate']*100:.1f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Avg Max Drawdown (%)': f"{metrics['avg_max_drawdown']*100:.1f}",
                'Time to Peak (%)': f"{metrics['avg_time_to_peak_pct']:.1f}",
            }
            
            # Add sprint-specific metrics if available
            if metrics['category'] == 'sprint':
                row['Mega Pump Rate (%)'] = f"{metrics['mega_pump_rate']*100:.1f}"
                row['Billion Pump Rate (%)'] = f"{metrics['billion_pump_rate']*100:.1f}"
            else:
                row['Mega Pump Rate (%)'] = "N/A"
                row['Billion Pump Rate (%)'] = "N/A"
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by category and then by median return
        df = df.sort_values(['Category', 'Median Return (%)'], ascending=[True, False])
        
        return df
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create visualizations for archetype analysis."""
        print(f"📊 Creating visualizations...")
        
        figures = {}
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.archetype_metrics).T
        
        # 1. Return vs Volatility Scatter Plot
        fig_scatter = go.Figure()
        
        categories = metrics_df['category'].unique()
        colors = px.colors.qualitative.Set1[:len(categories)]
        
        for i, category in enumerate(categories):
            category_data = metrics_df[metrics_df['category'] == category]
            
            fig_scatter.add_trace(go.Scatter(
                x=category_data['avg_volatility_cv'],
                y=category_data['avg_total_return'],
                mode='markers+text',
                text=category_data['archetype_name'],
                textposition="top center",
                marker=dict(
                    size=category_data['token_count'].astype(float).tolist(),
                    color=colors[i],
                    opacity=0.7,
                    sizemode='diameter',
                    sizeref=max(metrics_df['token_count']) / 100,
                    sizemin=10
                ),
                name=category,
                hovertemplate='<b>%{text}</b><br>' +
                             'Volatility (CV): %{x:.3f}<br>' +
                             'Avg Return: %{y:.3f}<br>' +
                             'Token Count: %{marker.size}<br>' +
                             '<extra></extra>'
            ))
        
        fig_scatter.update_layout(
            title='Archetype Risk-Return Profile',
            xaxis_title='Average Volatility (CV)',
            yaxis_title='Average Total Return',
            showlegend=True,
            hovermode='closest'
        )
        
        figures['risk_return_scatter'] = fig_scatter
        
        # 2. Return Distribution by Category
        fig_returns = go.Figure()
        
        for i, category in enumerate(categories):
            category_data = metrics_df[metrics_df['category'] == category]
            
            fig_returns.add_trace(go.Bar(
                x=category_data['archetype_name'],
                y=category_data['avg_total_return'],
                name=category,
                marker_color=colors[i],
                text=[f"{x:.1%}" for x in category_data['avg_total_return']],
                textposition='auto'
            ))
        
        fig_returns.update_layout(
            title='Average Returns by Archetype',
            xaxis_title='Archetype',
            yaxis_title='Average Total Return',
            showlegend=True,
            xaxis={'categoryorder': 'total descending'}
        )
        
        figures['returns_by_archetype'] = fig_returns
        
        # 3. Volatility Distribution by Category
        fig_volatility = go.Figure()
        
        for i, category in enumerate(categories):
            category_data = metrics_df[metrics_df['category'] == category]
            
            fig_volatility.add_trace(go.Bar(
                x=category_data['archetype_name'],
                y=category_data['avg_volatility_cv'],
                name=category,
                marker_color=colors[i],
                text=[f"{x:.3f}" for x in category_data['avg_volatility_cv']],
                textposition='auto'
            ))
        
        fig_volatility.update_layout(
            title='Average Volatility by Archetype',
            xaxis_title='Archetype',
            yaxis_title='Average Volatility (CV)',
            showlegend=True,
            xaxis={'categoryorder': 'total descending'}
        )
        
        figures['volatility_by_archetype'] = fig_volatility
        
        # 4. Token Count Distribution
        fig_tokens = go.Figure()
        
        for i, category in enumerate(categories):
            category_data = metrics_df[metrics_df['category'] == category]
            
            fig_tokens.add_trace(go.Bar(
                x=category_data['archetype_name'],
                y=category_data['token_count'],
                name=category,
                marker_color=colors[i],
                text=category_data['token_count'],
                textposition='auto'
            ))
        
        fig_tokens.update_layout(
            title='Token Count by Archetype',
            xaxis_title='Archetype',
            yaxis_title='Token Count',
            showlegend=True,
            xaxis={'categoryorder': 'total descending'}
        )
        
        figures['token_count_by_archetype'] = fig_tokens
        
        return figures
    
    def save_results(self, output_dir: Path = None) -> str:
        """Save analysis results."""
        if output_dir is None:
            output_dir = self.results_dir / "archetype_analysis"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison table
        comparison_df = self.generate_archetype_comparison()
        comparison_path = output_dir / f"archetype_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame(self.archetype_metrics).T
        metrics_path = output_dir / f"archetype_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save visualizations
        figures = self.create_visualizations()
        for fig_name, fig in figures.items():
            fig_path = output_dir / f"{fig_name}_{timestamp}.html"
            fig.write_html(fig_path)
        
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Comparison table: {comparison_path}")
        print(f"📈 Detailed metrics: {metrics_path}")
        print(f"📋 Visualizations: {len(figures)} HTML files")
        
        return timestamp
    
    def run_analysis(self, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete archetype metrics analysis including sprint tokens."""
        print(f"🚀 Starting Archetype Metrics Analysis")
        
        # Load data
        archetype_results = self.load_archetype_results(archetype_results_path)
        token_data = self.load_token_data(data_dir)
        
        # Calculate metrics
        self.token_metrics = self.calculate_token_metrics(token_data)
        self.archetype_metrics = self.aggregate_archetype_metrics()
        
        # Load and analyze sprint tokens (excluded from archetype analysis)
        sprint_tokens = self.load_sprint_tokens()
        sprint_metrics = {}
        if sprint_tokens:
            sprint_metrics = self.calculate_sprint_metrics(sprint_tokens)
            if sprint_metrics:
                # Add sprint metrics to archetype metrics
                self.archetype_metrics['sprint_unclustered'] = sprint_metrics
        
        # Generate comparison
        comparison_df = self.generate_archetype_comparison()
        
        # Display results
        print(f"\n📊 ARCHETYPE ANALYSIS SUMMARY")
        print(f"=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Save results
        timestamp = self.save_results()
        
        return {
            'archetype_metrics': self.archetype_metrics,
            'token_metrics': self.token_metrics,
            'sprint_metrics': sprint_metrics,
            'comparison_table': comparison_df,
            'timestamp': timestamp
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Archetype Metrics Analysis")
    parser.add_argument("--results", type=Path, 
                       help="Path to archetype characterization results JSON")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../data/with_archetypes_fixed"),
                       help="Path to token data directory with archetype labels")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Find latest results if not specified
    if not args.results:
        # Try different possible paths for the results directory
        possible_paths = [
            Path("../results/phase1_day9_10_archetypes"),
            Path("../../time_series/results/phase1_day9_10_archetypes"),
            Path("./results/phase1_day9_10_archetypes"),
            Path(__file__).parent.parent / "results" / "phase1_day9_10_archetypes"
        ]
        
        results_dir = None
        for path in possible_paths:
            if path.exists():
                results_dir = path
                print(f"📁 Found results directory: {path}")
                break
        
        if results_dir and results_dir.exists():
            json_files = list(results_dir.glob("archetype_characterization_*.json"))
            if json_files:
                args.results = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest results: {args.results}")
            else:
                print("❌ No archetype results found. Run the Phase 1 pipeline first.")
                return
        else:
            print("❌ Results directory not found. Tried paths:")
            for path in possible_paths:
                print(f"   {path} - {'exists' if path.exists() else 'not found'}")
            print("Run the Phase 1 pipeline first.")
            return
    
    # Initialize analyzer
    analyzer = ArchetypeMetricsAnalyzer(args.output_dir)
    
    try:
        # Run analysis
        results = analyzer.run_analysis(args.results, args.data_dir)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📈 Analyzed {len(results['archetype_metrics'])} archetypes")
        print(f"📊 Processed {len(results['token_metrics'])} tokens")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()