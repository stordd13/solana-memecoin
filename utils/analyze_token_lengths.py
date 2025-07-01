#!/usr/bin/env python3
"""
Script to analyze token length distribution from feature files
to inform walk-forward validation strategy design.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_parquet_files_by_category(features_dir):
    """Get parquet files organized by category."""
    categories = {}
    features_path = Path(features_dir)
    
    for category_dir in features_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            parquet_files = list(category_dir.glob("*.parquet"))
            categories[category_name] = parquet_files
            print(f"Found {len(parquet_files)} files in {category_name}")
    
    return categories

def sample_files_from_category(files, sample_size=None):
    """Sample files from a category for analysis. If sample_size is None, return all files."""
    if sample_size is None or len(files) <= sample_size:
        return files
    return random.sample(files, sample_size)

def analyze_token_length(file_path):
    """Analyze length of a single token file."""
    try:
        df = pd.read_parquet(file_path)
        return len(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def calculate_percentiles(lengths):
    """Calculate key percentiles for token lengths."""
    if not lengths:
        return {}
    
    return {
        'count': len(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'percentile_10': np.percentile(lengths, 10),
        'percentile_25': np.percentile(lengths, 25),
        'percentile_75': np.percentile(lengths, 75),
        'percentile_90': np.percentile(lengths, 90),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }

def main():
    features_dir = "../data/features"
    
    print("=== Token Length Distribution Analysis ===")
    print("This analysis will inform walk-forward validation design\n")
    
    # Get files by category
    categories = get_parquet_files_by_category(features_dir)
    
    if not categories:
        print("No categories found!")
        return
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Results storage
    all_results = {}
    overall_lengths = []
    
    for category_name, files in categories.items():
        print(f"\n--- Analyzing {category_name} ---")
        
        # Sample files for analysis (None = all files)
        sample_files = sample_files_from_category(files, sample_size=None)
        print(f"Analyzing {len(sample_files)} files from {category_name}")
        
        # Get lengths for sampled files
        lengths = []
        for file_path in sample_files:
            length = analyze_token_length(file_path)
            if length is not None:
                lengths.append(length)
        
        if lengths:
            # Calculate statistics
            stats = calculate_percentiles(lengths)
            all_results[category_name] = stats
            overall_lengths.extend(lengths)
            
            # Print results
            print(f"Sample size: {stats['count']}")
            print(f"Min length: {stats['min']}")
            print(f"Max length: {stats['max']}")
            print(f"Mean length: {stats['mean']:.1f}")
            print(f"Median length: {stats['median']:.1f}")
            print(f"Standard deviation: {stats['std']:.1f}")
            print(f"10th percentile: {stats['percentile_10']:.1f}")
            print(f"25th percentile: {stats['percentile_25']:.1f}")
            print(f"75th percentile: {stats['percentile_75']:.1f}")
            print(f"90th percentile: {stats['percentile_90']:.1f}")
            print(f"95th percentile: {stats['percentile_95']:.1f}")
            print(f"99th percentile: {stats['percentile_99']:.1f}")
        else:
            print(f"No valid files found in {category_name}")
    
    # Overall statistics
    if overall_lengths:
        print(f"\n=== OVERALL STATISTICS (all categories combined) ===")
        overall_stats = calculate_percentiles(overall_lengths)
        
        print(f"Total tokens analyzed: {overall_stats['count']}")
        print(f"Overall min length: {overall_stats['min']}")
        print(f"Overall max length: {overall_stats['max']}")
        print(f"Overall mean length: {overall_stats['mean']:.1f}")
        print(f"Overall median length: {overall_stats['median']:.1f}")
        print(f"Overall standard deviation: {overall_stats['std']:.1f}")
        print(f"Overall 10th percentile: {overall_stats['percentile_10']:.1f}")
        print(f"Overall 25th percentile: {overall_stats['percentile_25']:.1f}")
        print(f"Overall 75th percentile: {overall_stats['percentile_75']:.1f}")
        print(f"Overall 90th percentile: {overall_stats['percentile_90']:.1f}")
        print(f"Overall 95th percentile: {overall_stats['percentile_95']:.1f}")
        print(f"Overall 99th percentile: {overall_stats['percentile_99']:.1f}")
        
        # Walk-forward validation recommendations
        print(f"\n=== WALK-FORWARD VALIDATION RECOMMENDATIONS ===")
        
        min_training_window = int(overall_stats['percentile_75'])  # Use 75th percentile as min
        recommended_min_window = int(overall_stats['percentile_90'])  # Conservative choice
        step_size_small = max(10, int(overall_stats['percentile_25'] * 0.1))  # 10% of 25th percentile
        step_size_medium = max(20, int(overall_stats['median'] * 0.1))  # 10% of median
        step_size_large = max(50, int(overall_stats['percentile_75'] * 0.1))  # 10% of 75th percentile
        
        print(f"Minimum training window (75th percentile): {min_training_window} time steps")
        print(f"Recommended minimum training window (90th percentile): {recommended_min_window} time steps")
        print(f"Small step size (fine-grained): {step_size_small} time steps")
        print(f"Medium step size (balanced): {step_size_medium} time steps")
        print(f"Large step size (coarse-grained): {step_size_large} time steps")
        
        # Estimate number of validation folds possible
        print(f"\nEstimated validation folds for different strategies:")
        for percentile_name, percentile_value in [
            ("median token", overall_stats['median']),
            ("75th percentile token", overall_stats['percentile_75']),
            ("90th percentile token", overall_stats['percentile_90'])
        ]:
            available_length = int(percentile_value - recommended_min_window)
            if available_length > 0:
                folds_small = available_length // step_size_small
                folds_medium = available_length // step_size_medium
                folds_large = available_length // step_size_large
                print(f"  {percentile_name} ({int(percentile_value)} steps):")
                print(f"    Small steps: ~{folds_small} folds")
                print(f"    Medium steps: ~{folds_medium} folds")
                print(f"    Large steps: ~{folds_large} folds")
            else:
                print(f"  {percentile_name}: Not enough data for validation")
        
        # Save results to JSON
        all_results['overall'] = overall_stats
        output_file = "token_length_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_file}")
        
        # Create distribution plots
        create_distribution_plots(all_results, overall_lengths, categories)
    
    else:
        print("No data found to analyze!")

def create_distribution_plots(all_results, overall_lengths, categories):
    """Create comprehensive distribution plots."""
    print(f"\n=== CREATING DISTRIBUTION PLOTS ===")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Token Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall histogram
    ax1 = axes[0, 0]
    ax1.hist(overall_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.median(overall_lengths), color='red', linestyle='--', label=f'Median: {np.median(overall_lengths):.0f}')
    ax1.axvline(np.mean(overall_lengths), color='orange', linestyle='--', label=f'Mean: {np.mean(overall_lengths):.0f}')
    ax1.set_xlabel('Token Length (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Token Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot by category
    ax2 = axes[0, 1]
    category_data = []
    category_labels = []
    category_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (category_name, files) in enumerate(categories.items()):
        if len(files) > 0:
            # Sample and get lengths for this category (None = all files)
            sample_files = sample_files_from_category(files, sample_size=None)
            lengths = []
            for file_path in sample_files:
                length = analyze_token_length(file_path)
                if length is not None:
                    lengths.append(length)
            
            if lengths:
                category_data.append(lengths)
                category_labels.append(category_name.replace('_', ' ').title())
    
    if category_data:
        box_plot = ax2.boxplot(category_data, tick_labels=category_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], category_colors[:len(category_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_xlabel('Token Category')
    ax2.set_ylabel('Token Length (minutes)')
    ax2.set_title('Token Length by Category')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_lengths = np.sort(overall_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax3.plot(sorted_lengths, cumulative, linewidth=2, color='purple')
    ax3.axhline(0.25, color='red', linestyle=':', alpha=0.7, label='25th percentile')
    ax3.axhline(0.50, color='orange', linestyle=':', alpha=0.7, label='50th percentile')
    ax3.axhline(0.75, color='green', linestyle=':', alpha=0.7, label='75th percentile')
    ax3.axhline(0.90, color='blue', linestyle=':', alpha=0.7, label='90th percentile')
    ax3.set_xlabel('Token Length (minutes)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics table
    stats_data = []
    for category_name in ['overall'] + list(categories.keys()):
        if category_name in all_results:
            stats = all_results[category_name]
            stats_data.append([
                category_name.replace('_', ' ').title(),
                f"{stats['count']}",
                f"{stats['min']:.0f}",
                f"{stats['median']:.0f}",
                f"{stats['mean']:.0f}",
                f"{stats['max']:.0f}",
                f"{stats['std']:.0f}"
            ])
    
    if stats_data:
        table = ax4.table(cellText=stats_data,
                         colLabels=['Category', 'Count', 'Min', 'Median', 'Mean', 'Max', 'Std'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color the header
        try:
            for i in range(7):  # Number of columns
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        except:
            pass  # Skip if table structure is different
        
        # Alternate row colors
        for i in range(1, len(stats_data) + 1):
            for j in range(len(stats_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
    
    ax4.set_title('Summary Statistics by Category', pad=20)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    plot_file = output_dir / "token_length_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to {plot_file}")
    
    # Also save as PDF for better quality
    pdf_file = output_dir / "token_length_distribution.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"High-quality PDF saved to {pdf_file}")
    
    plt.show()
    
    # Create additional detailed histogram for each category
    if len(categories) > 1:
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('Token Length Distribution by Category (Detailed View)', fontsize=16, fontweight='bold')
        
        axes2 = axes2.flatten()
        
        for i, (category_name, files) in enumerate(categories.items()):
            if i < 4 and len(files) > 0:  # Only plot first 4 categories
                ax = axes2[i]
                
                # Get lengths for this category (None = all files)
                sample_files = sample_files_from_category(files, sample_size=None)
                lengths = []
                for file_path in sample_files:
                    length = analyze_token_length(file_path)
                    if length is not None:
                        lengths.append(length)
                
                if lengths:
                    ax.hist(lengths, bins=30, alpha=0.7, color=category_colors[i], edgecolor='black')
                    ax.axvline(np.median(lengths), color='red', linestyle='--', 
                              label=f'Median: {np.median(lengths):.0f}')
                    ax.axvline(np.mean(lengths), color='orange', linestyle='--', 
                              label=f'Mean: {np.mean(lengths):.0f}')
                    ax.set_xlabel('Token Length (minutes)')
                    ax.set_ylabel('Frequency')
                    ax.set_title(category_name.replace('_', ' ').title())
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(categories), 4):
            axes2[i].axis('off')
        
        plt.tight_layout()
        
        # Save detailed plots
        detailed_plot_file = output_dir / "token_length_by_category_detailed.png"
        plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
        print(f"Detailed category plots saved to {detailed_plot_file}")
        
        plt.show()

if __name__ == "__main__":
    main()