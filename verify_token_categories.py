#!/usr/bin/env python3
"""
Quick script to verify token distribution across lifespan categories
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_token_distribution():
    """Analyze how tokens are distributed across categories"""
    
    print("=" * 80)
    print("TOKEN CATEGORY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load from processed categories
    processed_dir = project_root / 'data' / 'processed'
    categories = ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes', 'tokens_with_gaps']
    
    all_tokens = {}
    original_category_counts = {}
    
    # Load tokens (sample for speed)
    SAMPLE_SIZE = 1000  # Sample 1000 tokens per category for speed
    
    for category in categories:
        category_path = processed_dir / category
        if not category_path.exists():
            print(f"Warning: Category directory not found: {category_path}")
            continue
            
        parquet_files = list(category_path.glob("*.parquet"))
        original_category_counts[category] = len(parquet_files)
        
        # Sample for faster analysis
        sample_files = parquet_files[:SAMPLE_SIZE] if len(parquet_files) > SAMPLE_SIZE else parquet_files
        
        print(f"Sampling {len(sample_files)} tokens from {category} (total: {len(parquet_files)})...")
        
        for file_path in sample_files:
            try:
                df = pl.read_parquet(file_path)
                token_name = file_path.stem
                
                if 'datetime' in df.columns and 'price' in df.columns:
                    df = df.with_columns([pl.lit(category).alias('original_category')])
                    all_tokens[token_name] = df
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    print(f"\n📊 ORIGINAL CATEGORY DISTRIBUTION:")
    total_original = 0
    for category, count in original_category_counts.items():
        print(f"  {category}: {count:,} tokens")
        total_original += count
    print(f"  TOTAL: {total_original:,} tokens")
    
    # Now analyze lifespans and death detection
    print(f"\n🔍 ANALYZING LIFESPANS AND DEATH DETECTION...")
    
    from time_series.archetype_utils import detect_token_death
    
    lifespan_stats = {
        'Sprint': [],      # 0-400 active minutes
        'Standard': [],    # 400-1200 active minutes  
        'Marathon': []     # 1200+ active minutes
    }
    
    death_stats = {'dead': 0, 'alive': 0}
    actual_lifespans = []
    
    for token_name, df in all_tokens.items():
        prices = df['price'].to_numpy()
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
        
        # Detect death
        death_minute = detect_token_death(prices, returns, window=30)
        
        # Calculate actual lifespan
        actual_lifespan = len(df)  # Total data points
        active_lifespan = death_minute if death_minute is not None else len(df)
        
        actual_lifespans.append(actual_lifespan)
        
        if death_minute is not None:
            death_stats['dead'] += 1
        else:
            death_stats['alive'] += 1
        
        # Categorize by ACTIVE lifespan (death-aware)
        if 0 <= active_lifespan <= 400:
            lifespan_stats['Sprint'].append({
                'token': token_name,
                'actual_lifespan': actual_lifespan,
                'active_lifespan': active_lifespan,
                'death_minute': death_minute,
                'original_category': df['original_category'][0]
            })
        elif 400 < active_lifespan <= 1200:
            lifespan_stats['Standard'].append({
                'token': token_name,
                'actual_lifespan': actual_lifespan,
                'active_lifespan': active_lifespan,
                'death_minute': death_minute,
                'original_category': df['original_category'][0]
            })
        elif active_lifespan > 1200:
            lifespan_stats['Marathon'].append({
                'token': token_name,
                'actual_lifespan': actual_lifespan,
                'active_lifespan': active_lifespan,
                'death_minute': death_minute,
                'original_category': df['original_category'][0]
            })
    
    print(f"\n📈 LIFESPAN CATEGORY DISTRIBUTION (DEATH-AWARE):")
    for category, tokens in lifespan_stats.items():
        print(f"  {category}: {len(tokens):,} tokens")
        
        if len(tokens) > 0:
            active_lifespans = [t['active_lifespan'] for t in tokens]
            actual_lifespans_cat = [t['actual_lifespan'] for t in tokens]
            print(f"    Active lifespan range: {min(active_lifespans)} - {max(active_lifespans)} minutes")
            print(f"    Actual lifespan range: {min(actual_lifespans_cat)} - {max(actual_lifespans_cat)} minutes")
            
            # Show a few examples
            print(f"    Examples:")
            for i, token in enumerate(tokens[:3]):
                print(f"      {token['token']}: actual={token['actual_lifespan']}min, active={token['active_lifespan']}min, death={token['death_minute']}")
    
    print(f"\n💀 DEATH DETECTION SUMMARY:")
    print(f"  Dead tokens: {death_stats['dead']:,}")
    print(f"  Alive tokens: {death_stats['alive']:,}")
    print(f"  Death rate: {death_stats['dead']/(death_stats['dead']+death_stats['alive'])*100:.1f}%")
    
    print(f"\n📊 ACTUAL LIFESPAN STATISTICS:")
    actual_lifespans = np.array(actual_lifespans)
    print(f"  Min lifespan: {np.min(actual_lifespans)} minutes")
    print(f"  Max lifespan: {np.max(actual_lifespans)} minutes")
    print(f"  Mean lifespan: {np.mean(actual_lifespans):.1f} minutes")
    print(f"  Median lifespan: {np.median(actual_lifespans):.1f} minutes")
    
    # Check for tokens with 2000+ minutes of actual data
    long_tokens = actual_lifespans[actual_lifespans >= 2000]
    print(f"  Tokens with 2000+ minutes: {len(long_tokens):,}")
    if len(long_tokens) > 0:
        print(f"  Longest token: {np.max(long_tokens)} minutes")
    
    # Find discrepancy - tokens with long actual lifespan but not in Marathon
    print(f"\n🔍 POTENTIAL ISSUES:")
    long_actual_not_marathon = 0
    for token_name, df in all_tokens.items():
        actual_lifespan = len(df)
        if actual_lifespan >= 1200:  # Should be Marathon
            # Check if it's actually in Marathon
            in_marathon = any(t['token'] == token_name for t in lifespan_stats['Marathon'])
            if not in_marathon:
                long_actual_not_marathon += 1
                if long_actual_not_marathon <= 5:  # Show first 5 examples
                    prices = df['price'].to_numpy()
                    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
                    death_minute = detect_token_death(prices, returns, window=30)
                    active_lifespan = death_minute if death_minute is not None else len(df)
                    print(f"  {token_name}: actual={actual_lifespan}min, active={active_lifespan}min, death={death_minute}")
    
    if long_actual_not_marathon > 0:
        print(f"  FOUND {long_actual_not_marathon} tokens with 1200+ actual minutes NOT in Marathon category!")
        print(f"  This explains why Marathon only has {len(lifespan_stats['Marathon'])} tokens!")
    else:
        print(f"  No discrepancies found. All long tokens are properly categorized.")

if __name__ == "__main__":
    analyze_token_distribution()