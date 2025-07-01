#!/usr/bin/env python3
"""
Simple script to plot token length distribution from existing analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def create_simple_distribution_plot():
    """Create simple distribution plot from existing analysis."""
    
    # Load existing analysis
    try:
        with open('../token_length_analysis.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Run analyze_token_lengths.py first to generate data")
        return
    
    # Extract overall statistics
    overall_stats = results.get('overall', {})
    if not overall_stats:
        print("No overall statistics found")
        return
    
    print("Creating distribution plot...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Token Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Load a sample of actual data for histogram
    features_dir = Path("../data/features")
    category_lengths = {}
    
    for category_dir in features_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            parquet_files = list(category_dir.glob("*.parquet"))[:50]  # Sample 50 files
            lengths = []
            
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    lengths.append(len(df))
                except:
                    continue
            
            if lengths:
                category_lengths[category_name] = lengths
    
    # Plot 1: Overall histogram
    all_lengths = []
    for lengths in category_lengths.values():
        all_lengths.extend(lengths)
    
    if all_lengths:
        ax1.hist(all_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.median(all_lengths), color='red', linestyle='--', 
                   label=f'Median: {np.median(all_lengths):.0f}')
        ax1.axvline(np.mean(all_lengths), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(all_lengths):.0f}')
        ax1.set_xlabel('Token Length (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Token Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by category
    if category_lengths:
        category_data = []
        category_labels = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (cat_name, lengths) in enumerate(category_lengths.items()):
            category_data.append(lengths)
            category_labels.append(cat_name.replace('_', ' ').title())
        
        box_plot = ax2.boxplot(category_data, tick_labels=category_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(category_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Token Category')
        ax2.set_ylabel('Token Length (minutes)')
        ax2.set_title('Token Length by Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    plot_file = output_dir / "token_length_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {plot_file}")
    
    # Show statistics
    print(f"\n=== DISTRIBUTION SUMMARY ===")
    if all_lengths:
        print(f"Total tokens sampled: {len(all_lengths)}")
        print(f"Min length: {min(all_lengths)}")
        print(f"Max length: {max(all_lengths)}")
        print(f"Mean length: {np.mean(all_lengths):.1f}")
        print(f"Median length: {np.median(all_lengths):.1f}")
        print(f"25th percentile: {np.percentile(all_lengths, 25):.1f}")
        print(f"75th percentile: {np.percentile(all_lengths, 75):.1f}")
        print(f"90th percentile: {np.percentile(all_lengths, 90):.1f}")
    
    plt.show()

if __name__ == "__main__":
    create_simple_distribution_plot()