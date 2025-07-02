#!/usr/bin/env python3
"""
Token Variability Analysis Tool
Analyzes price patterns to distinguish real market variations from "straight line" tokens
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from data_cleaning.clean_tokens import CategoryAwareTokenCleaner

class TokenVariabilityAnalyzer:
    def __init__(self):
        self.cleaner = CategoryAwareTokenCleaner()
        self.results = []
        
    def analyze_token_variability(self, df: pl.DataFrame, token_name: str) -> Dict:
        """Comprehensive variability analysis for a single token"""
        prices = df['price'].to_numpy()
        
        if len(prices) < 30:
            return None
            
        # Convert to log prices
        log_prices = np.log(prices + 1e-10)
        
        # Calculate all metrics
        price_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        log_price_cv = np.std(log_prices) / np.abs(np.mean(log_prices)) if np.mean(log_prices) != 0 else 0
        
        # Rolling variability
        window_size = min(60, len(prices) // 4)
        rolling_std = []
        for i in range(0, len(prices) - window_size + 1, window_size // 2):
            window_prices = prices[i:i + window_size]
            rolling_std.append(np.std(window_prices) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0)
        
        # Flat periods
        returns = np.abs(np.diff(prices) / prices[:-1])
        flat_periods = np.sum(returns < 0.001) / len(returns) if len(returns) > 0 else 1.0
        
        # Range efficiency
        meaningful_moves = np.sum(np.abs(np.diff(prices)) > np.mean(prices) * 0.01)
        range_efficiency = meaningful_moves / len(prices) if len(prices) > 0 else 0
        
        # Entropy
        if len(returns) > 10:
            bins = min(10, len(returns) // 5)
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist[hist > 0]
            if len(hist) > 1:
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob))
                max_entropy = np.log2(len(hist))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0
        else:
            normalized_entropy = 0
        
        # Current filtering criteria
        is_low_variability = (
            price_cv < 0.05 and
            log_price_cv < 0.1 and
            flat_periods > 0.8 and
            range_efficiency < 0.1 and
            normalized_entropy < 0.3
        )
        
        return {
            'token_name': token_name,
            'length': len(prices),
            'price_cv': price_cv,
            'log_price_cv': log_price_cv,
            'flat_periods_fraction': flat_periods,
            'range_efficiency': range_efficiency,
            'normalized_entropy': normalized_entropy,
            'is_low_variability': is_low_variability,
            'price_min': np.min(prices),
            'price_max': np.max(prices),
            'price_ratio': np.max(prices) / np.min(prices) if np.min(prices) > 0 else 0,
            'prices': prices,
            'log_prices': log_prices,
            'returns': returns
        }
    
    def analyze_sample_tokens(self, num_samples_per_category: int = 20):
        """Analyze sample tokens from each category"""
        categories = ['normal_behavior_tokens', 'dead_tokens', 'tokens_with_extremes', 'tokens_with_gaps']
        
        for category in categories:
            processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / category
            if not processed_dir.exists():
                print(f"Directory not found: {processed_dir}")
                continue
                
            files = list(processed_dir.glob('*.parquet'))[:num_samples_per_category]
            print(f"Analyzing {len(files)} {category} tokens...")
            
            for file_path in files:
                try:
                    df = pl.read_parquet(file_path)
                    token_name = file_path.stem
                    
                    # Add returns if missing
                    if 'returns' not in df.columns:
                        df = self.cleaner._calculate_returns(df)
                    
                    result = self.analyze_token_variability(df, token_name)
                    if result:
                        result['category'] = category
                        self.results.append(result)
                        
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    def plot_variability_distributions(self):
        """Create comprehensive plots of variability metrics"""
        if not self.results:
            print("No results to plot. Run analyze_sample_tokens() first.")
            return
            
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['prices', 'log_prices', 'returns']} 
                          for r in self.results])
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price CV distribution by category
        plt.subplot(3, 3, 1)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            plt.hist(cat_data['price_cv'], bins=30, alpha=0.7, label=category, density=True)
        plt.axvline(x=0.05, color='red', linestyle='--', label='Filter threshold')
        plt.xlabel('Price Coefficient of Variation')
        plt.ylabel('Density')
        plt.title('Price CV Distribution by Category')
        plt.legend()
        plt.yscale('log')
        
        # 2. Flat periods distribution
        plt.subplot(3, 3, 2)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            plt.hist(cat_data['flat_periods_fraction'], bins=30, alpha=0.7, label=category, density=True)
        plt.axvline(x=0.8, color='red', linestyle='--', label='Filter threshold')
        plt.xlabel('Fraction of Flat Periods')
        plt.ylabel('Density')
        plt.title('Flat Periods Distribution by Category')
        plt.legend()
        
        # 3. Entropy distribution
        plt.subplot(3, 3, 3)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            plt.hist(cat_data['normalized_entropy'], bins=30, alpha=0.7, label=category, density=True)
        plt.axvline(x=0.3, color='red', linestyle='--', label='Filter threshold')
        plt.xlabel('Normalized Entropy')
        plt.ylabel('Density')
        plt.title('Entropy Distribution by Category')
        plt.legend()
        
        # 4. Range efficiency distribution
        plt.subplot(3, 3, 4)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            plt.hist(cat_data['range_efficiency'], bins=30, alpha=0.7, label=category, density=True)
        plt.axvline(x=0.1, color='red', linestyle='--', label='Filter threshold')
        plt.xlabel('Range Efficiency')
        plt.ylabel('Density')
        plt.title('Range Efficiency Distribution by Category')
        plt.legend()
        
        # 5. Filtering effectiveness scatter plot
        plt.subplot(3, 3, 5)
        colors = {'normal_behavior_tokens': 'blue', 'dead_tokens': 'red', 
                 'tokens_with_extremes': 'orange', 'tokens_with_gaps': 'green'}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            filtered = cat_data[cat_data['is_low_variability']]
            passed = cat_data[~cat_data['is_low_variability']]
            
            if len(filtered) > 0:
                plt.scatter(filtered['price_cv'], filtered['flat_periods_fraction'], 
                           c=colors[category], marker='x', s=100, alpha=0.7, 
                           label=f'{category} (filtered)')
            if len(passed) > 0:
                plt.scatter(passed['price_cv'], passed['flat_periods_fraction'], 
                           c=colors[category], marker='o', s=50, alpha=0.7, 
                           label=f'{category} (passed)')
        
        plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Price CV')
        plt.ylabel('Flat Periods Fraction')
        plt.title('Filtering Decision: CV vs Flat Periods')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Price ratio vs CV
        plt.subplot(3, 3, 6)
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            plt.scatter(cat_data['price_ratio'], cat_data['price_cv'], 
                       alpha=0.7, label=category, s=50)
        plt.xlabel('Price Ratio (max/min)')
        plt.ylabel('Price CV')
        plt.title('Price Ratio vs CV by Category')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        
        # 7. Category summary statistics
        plt.subplot(3, 3, 7)
        summary_stats = []
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            filtered_pct = (cat_data['is_low_variability'].sum() / len(cat_data)) * 100
            summary_stats.append({
                'category': category.replace('_', '\n'),
                'total': len(cat_data),
                'filtered_pct': filtered_pct,
                'avg_cv': cat_data['price_cv'].mean(),
                'avg_flat': cat_data['flat_periods_fraction'].mean()
            })
        
        summary_df = pd.DataFrame(summary_stats)
        bars = plt.bar(summary_df['category'], summary_df['filtered_pct'])
        plt.ylabel('Percentage Filtered (%)')
        plt.title('Filtering Rate by Category')
        plt.xticks(rotation=45)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, summary_df['filtered_pct']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # 8 & 9. Example price series plots
        self._plot_example_tokens(plt.subplot(3, 3, 8), plt.subplot(3, 3, 9))
        
        plt.tight_layout()
        
        # Save to utils/results directory
        output_path = Path(__file__).parent.parent / 'results' / 'token_variability_analysis.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Analysis plot saved to: {output_path}")
        
        return summary_df
    
    def _plot_example_tokens(self, ax1, ax2):
        """Plot example tokens showing high vs low variability"""
        if not self.results:
            return
            
        # Find examples
        low_var_example = None
        high_var_example = None
        
        for result in self.results:
            if result['is_low_variability'] and low_var_example is None:
                low_var_example = result
            elif not result['is_low_variability'] and high_var_example is None:
                high_var_example = result
            
            if low_var_example and high_var_example:
                break
        
        # Plot low variability example
        if low_var_example:
            ax1.plot(low_var_example['prices'])
            ax1.set_title(f'Low Variability Example\n{low_var_example["token_name"][:20]}...\n'
                         f'CV: {low_var_example["price_cv"]:.4f}, '
                         f'Flat: {low_var_example["flat_periods_fraction"]:.2f}')
            ax1.set_ylabel('Price')
            ax1.set_xlabel('Time (minutes)')
        
        # Plot high variability example
        if high_var_example:
            ax2.plot(high_var_example['prices'])
            ax2.set_title(f'High Variability Example\n{high_var_example["token_name"][:20]}...\n'
                         f'CV: {high_var_example["price_cv"]:.4f}, '
                         f'Flat: {high_var_example["flat_periods_fraction"]:.2f}')
            ax2.set_ylabel('Price')
            ax2.set_xlabel('Time (minutes)')
    
    def print_filtering_summary(self):
        """Print detailed filtering statistics"""
        if not self.results:
            print("No results available.")
            return
            
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['prices', 'log_prices', 'returns']} 
                          for r in self.results])
        
        print("\n" + "="*80)
        print("TOKEN VARIABILITY ANALYSIS SUMMARY")
        print("="*80)
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            filtered = cat_data[cat_data['is_low_variability']]
            passed = cat_data[~cat_data['is_low_variability']]
            
            print(f"\n📊 {category.upper().replace('_', ' ')}:")
            print(f"   Total analyzed: {len(cat_data)}")
            print(f"   Filtered (low variability): {len(filtered)} ({len(filtered)/len(cat_data)*100:.1f}%)")
            print(f"   Passed: {len(passed)} ({len(passed)/len(cat_data)*100:.1f}%)")
            
            if len(filtered) > 0:
                print(f"   Filtered tokens - Avg CV: {filtered['price_cv'].mean():.4f}, "
                      f"Avg Flat: {filtered['flat_periods_fraction'].mean():.3f}")
            if len(passed) > 0:
                print(f"   Passed tokens - Avg CV: {passed['price_cv'].mean():.4f}, "
                      f"Avg Flat: {passed['flat_periods_fraction'].mean():.3f}")
        
        print(f"\n🎯 CURRENT FILTER THRESHOLDS:")
        print(f"   Price CV < 0.05")
        print(f"   Log Price CV < 0.1")  
        print(f"   Flat periods > 0.8 (80%)")
        print(f"   Range efficiency < 0.1")
        print(f"   Normalized entropy < 0.3")
        print(f"   (ALL conditions must be met)")

# Main execution
if __name__ == "__main__":
    print("🔬 Starting Token Variability Analysis...")
    
    analyzer = TokenVariabilityAnalyzer()
    
    # Analyze samples from each category
    analyzer.analyze_sample_tokens(num_samples_per_category=30)
    
    # Print summary
    analyzer.print_filtering_summary()
    
    # Generate plots
    print("\n📈 Generating variability analysis plots...")
    summary_df = analyzer.plot_variability_distributions()
    
    print(f"\n✅ Analysis complete! Check 'utils/results/token_variability_analysis.png' for detailed plots.")
    print(f"📊 Analyzed {len(analyzer.results)} tokens total.")