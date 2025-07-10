#!/usr/bin/env python3
"""
Comprehensive token length analysis for walk-forward validation design.
This analysis examines time series data sampled at 1-minute intervals.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
import json

def analyze_comprehensive_lengths(features_dir, sample_size_per_category=None):
    """Comprehensive analysis of token lengths across ALL categories (no sampling if None)."""
    
    print("=== COMPREHENSIVE TOKEN LENGTH ANALYSIS ===")
    print("Each token contains minute-level time series data (24 features)")
    print("Key insight: 1 row = 1 minute of data")
    print()
    
    categories = {}
    features_path = Path(features_dir)
    
    # Get all categories and files
    for category_dir in features_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            parquet_files = list(category_dir.glob("*.parquet"))
            categories[category_name] = parquet_files
            print(f"Found {len(parquet_files)} files in {category_name}")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Analysis results
    results = {}
    all_lengths = []
    detailed_stats = {}
    
    for category_name, files in categories.items():
        print(f"\n--- Analyzing {category_name} ---")
        
        # Sample files (or use all if sample_size_per_category is None)
        if sample_size_per_category is None:
            sample_files = files
            print(f"Analyzing ALL {len(sample_files)} files from {category_name}")
        else:
            sample_files = files if len(files) <= sample_size_per_category else random.sample(files, sample_size_per_category)
            print(f"Analyzing {len(sample_files)} files from {category_name} (sampled from {len(files)} total)")
        
        lengths = []
        time_spans = []
        
        for file_path in sample_files:
            try:
                df = pd.read_parquet(file_path)
                length_minutes = len(df)
                lengths.append(length_minutes)
                all_lengths.append(length_minutes)
                
                # Calculate time span if datetime column exists
                if 'datetime' in df.columns:
                    time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600  # hours
                    time_spans.append(time_span)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if lengths:
            # Calculate statistics
            stats = {
                'count': len(lengths),
                'min_minutes': np.min(lengths),
                'max_minutes': np.max(lengths),
                'mean_minutes': np.mean(lengths),
                'median_minutes': np.median(lengths),
                'std_minutes': np.std(lengths),
                'percentile_10': np.percentile(lengths, 10),
                'percentile_25': np.percentile(lengths, 25),
                'percentile_50': np.percentile(lengths, 50),
                'percentile_75': np.percentile(lengths, 75),
                'percentile_90': np.percentile(lengths, 90),
                'percentile_95': np.percentile(lengths, 95),
                'percentile_99': np.percentile(lengths, 99)
            }
            
            # Convert to hours for easier interpretation
            stats_hours = {k.replace('minutes', 'hours'): v/60 if 'minutes' in k else v 
                          for k, v in stats.items()}
            
            results[category_name] = stats
            detailed_stats[category_name] = {
                'minutes': stats,
                'hours': stats_hours,
                'time_spans_hours': time_spans if time_spans else []
            }
            
            # Print results
            print(f"Sample size: {stats['count']}")
            print(f"Length range: {stats['min_minutes']:.0f} - {stats['max_minutes']:.0f} minutes")
            print(f"Length range: {stats['min_minutes']/60:.1f} - {stats['max_minutes']/60:.1f} hours")
            print(f"Mean length: {stats['mean_minutes']:.1f} minutes ({stats['mean_minutes']/60:.1f} hours)")
            print(f"Median length: {stats['median_minutes']:.1f} minutes ({stats['median_minutes']/60:.1f} hours)")
            print(f"Key percentiles (minutes): 25th={stats['percentile_25']:.0f}, 75th={stats['percentile_75']:.0f}, 90th={stats['percentile_90']:.0f}")
            print(f"Key percentiles (hours): 25th={stats['percentile_25']/60:.1f}, 75th={stats['percentile_75']/60:.1f}, 90th={stats['percentile_90']/60:.1f}")
            
            if time_spans:
                avg_time_span = np.mean(time_spans)
                print(f"Average actual time span: {avg_time_span:.1f} hours")
    
    # Overall analysis
    if all_lengths:
        print(f"\n=== OVERALL STATISTICS (ALL CATEGORIES) ===")
        overall_stats = {
            'count': len(all_lengths),
            'min_minutes': np.min(all_lengths),
            'max_minutes': np.max(all_lengths),
            'mean_minutes': np.mean(all_lengths),
            'median_minutes': np.median(all_lengths),
            'std_minutes': np.std(all_lengths),
            'percentile_10': np.percentile(all_lengths, 10),
            'percentile_25': np.percentile(all_lengths, 25),
            'percentile_50': np.percentile(all_lengths, 50),
            'percentile_75': np.percentile(all_lengths, 75),
            'percentile_90': np.percentile(all_lengths, 90),
            'percentile_95': np.percentile(all_lengths, 95),
            'percentile_99': np.percentile(all_lengths, 99)
        }
        
        print(f"Total tokens analyzed: {overall_stats['count']}")
        print(f"Overall range: {overall_stats['min_minutes']:.0f} - {overall_stats['max_minutes']:.0f} minutes")
        print(f"Overall range: {overall_stats['min_minutes']/60:.1f} - {overall_stats['max_minutes']/60:.1f} hours")
        print(f"Overall mean: {overall_stats['mean_minutes']:.1f} minutes ({overall_stats['mean_minutes']/60:.1f} hours)")
        print(f"Overall median: {overall_stats['median_minutes']:.1f} minutes ({overall_stats['median_minutes']/60:.1f} hours)")
        print(f"Overall percentiles (minutes): 25th={overall_stats['percentile_25']:.0f}, 75th={overall_stats['percentile_75']:.0f}, 90th={overall_stats['percentile_90']:.0f}")
        print(f"Overall percentiles (hours): 25th={overall_stats['percentile_25']/60:.1f}, 75th={overall_stats['percentile_75']/60:.1f}, 90th={overall_stats['percentile_90']/60:.1f}")
        
        # LSTM and Walk-Forward Validation Recommendations
        print(f"\n=== LSTM TRAINING & WALK-FORWARD VALIDATION RECOMMENDATIONS ===")
        
        # Calculate recommendations based on data distribution
        min_sequence_length = 60  # 1 hour minimum for LSTM
        conservative_min = max(120, int(overall_stats['percentile_25']))  # 2 hours or 25th percentile
        recommended_min = max(180, int(overall_stats['percentile_50']))   # 3 hours or median
        safe_min = max(240, int(overall_stats['percentile_75']))          # 4 hours or 75th percentile
        
        # Step sizes for walk-forward validation
        step_small = 15   # 15 minutes
        step_medium = 30  # 30 minutes  
        step_large = 60   # 1 hour
        step_xlarge = 120 # 2 hours
        
        print(f"\nMINIMUM SEQUENCE LENGTHS FOR LSTM:")
        print(f"  Absolute minimum: {min_sequence_length} minutes (1 hour)")
        print(f"  Conservative minimum: {conservative_min} minutes ({conservative_min/60:.1f} hours)")
        print(f"  Recommended minimum: {recommended_min} minutes ({recommended_min/60:.1f} hours)")
        print(f"  Safe minimum: {safe_min} minutes ({safe_min/60:.1f} hours)")
        
        print(f"\nRECOMMENDED WALK-FORWARD STEP SIZES:")
        print(f"  Fine-grained: {step_small} minutes (good for short-term patterns)")
        print(f"  Balanced: {step_medium} minutes (good balance of granularity/speed)")
        print(f"  Coarse: {step_large} minutes (faster validation, longer-term focus)")
        print(f"  Very coarse: {step_xlarge} minutes (fastest, macro trend focus)")
        
        # Calculate number of possible validation folds
        print(f"\nVALIDATION FOLD ESTIMATES:")
        print(f"Using recommended minimum training window ({recommended_min} minutes):")
        
        for percentile_name, percentile_val in [
            ("25th percentile token", overall_stats['percentile_25']),
            ("Median token", overall_stats['percentile_50']),
            ("75th percentile token", overall_stats['percentile_75']),
            ("90th percentile token", overall_stats['percentile_90'])
        ]:
            available_for_validation = int(percentile_val - recommended_min)
            if available_for_validation > 0:
                folds_small = available_for_validation // step_small
                folds_medium = available_for_validation // step_medium
                folds_large = available_for_validation // step_large
                folds_xlarge = available_for_validation // step_xlarge
                
                print(f"\n  {percentile_name} ({int(percentile_val)} min, {percentile_val/60:.1f}h):")
                print(f"    15-min steps: {folds_small} folds")
                print(f"    30-min steps: {folds_medium} folds") 
                print(f"    60-min steps: {folds_large} folds")
                print(f"    120-min steps: {folds_xlarge} folds")
            else:
                print(f"\n  {percentile_name}: Insufficient data for validation")
        
        # Category-specific recommendations
        print(f"\n=== CATEGORY-SPECIFIC INSIGHTS ===")
        for category, stats in results.items():
            median_hours = stats['median_minutes'] / 60
            if median_hours >= 20:
                recommendation = "Excellent for walk-forward validation"
            elif median_hours >= 12:
                recommendation = "Good for walk-forward validation"
            elif median_hours >= 6:
                recommendation = "Moderate for walk-forward validation"
            elif median_hours >= 3:
                recommendation = "Limited walk-forward validation"
            else:
                recommendation = "Not suitable for walk-forward validation"
            
            print(f"  {category}: Median {median_hours:.1f}h -> {recommendation}")
        
        # Save detailed results
        results['overall'] = overall_stats
        output_file = "comprehensive_token_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary_stats': results,
                'detailed_stats': detailed_stats,
                'recommendations': {
                    'min_sequence_length': min_sequence_length,
                    'conservative_min': conservative_min,
                    'recommended_min': recommended_min,
                    'safe_min': safe_min,
                    'step_sizes': {
                        'fine': step_small,
                        'balanced': step_medium,
                        'coarse': step_large,
                        'very_coarse': step_xlarge
                    }
                }
            }, f, indent=2, default=str)
        
        print(f"\nDetailed analysis saved to {output_file}")
        
        return results, overall_stats
    
    else:
        print("No data found!")
        return {}, {}

if __name__ == "__main__":
    features_dir = "../data/features"
    print("🔥 ANALYZING ALL 30K+ TOKENS (NO SAMPLING)")
    analyze_comprehensive_lengths(features_dir, sample_size_per_category=None)