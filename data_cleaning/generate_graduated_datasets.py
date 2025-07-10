#!/usr/bin/env python3
"""
Generate Graduated Datasets for Different Time Horizons

This script creates multiple cleaned datasets using different cleaning strategies:
- short_term: For 15m-60m predictions (very lenient cleaning)
- medium_term: For 120m-360m predictions (balanced cleaning)
- long_term: For 720m+ predictions (aggressive cleaning)

USAGE:
    python data_cleaning/generate_graduated_datasets.py --limit 100

This will create:
- data/cleaned_tokens_short_term/
- data/cleaned_tokens_medium_term/
- data/cleaned_tokens_long_term/
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional, Dict, List
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from clean_tokens import CategoryAwareTokenCleaner


def generate_graduated_datasets(limit: Optional[int] = None, 
                               strategies: List[str] = None) -> Dict[str, Dict]:
    """
    Generate datasets with different cleaning strategies
    
    Args:
        limit: Maximum number of tokens to process per strategy
        strategies: List of strategies to generate ['short_term', 'medium_term', 'long_term']
    
    Returns:
        Dictionary with results for each strategy
    """
    if strategies is None:
        strategies = ['short_term', 'medium_term', 'long_term']
    
    # Initialize cleaner
    cleaner = CategoryAwareTokenCleaner()
    
    # Get all token files
    data_dir = Path('data/tokens')
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    token_files = list(data_dir.glob('*.parquet'))
    if limit:
        token_files = token_files[:limit]
    
    print(f"🎯 Generating graduated datasets for {len(token_files)} tokens")
    print(f"📊 Strategies: {', '.join(strategies)}")
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"🔄 Processing strategy: {strategy.upper()}")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = Path(f'data/cleaned_tokens_{strategy}')
        output_dir.mkdir(exist_ok=True)
        
        # Process tokens
        strategy_results = {
            'cleaned_successfully': 0,
            'excluded_tokens': 0,
            'total_processed': 0,
            'exclusion_reasons': {},
            'cleaning_stats': {}
        }
        
        for token_file in tqdm(token_files, desc=f"Processing {strategy}"):
            try:
                token_name = token_file.stem
                
                # Load token data
                df = pl.read_parquet(token_file)
                
                # Apply the specific cleaning strategy
                if strategy == 'short_term':
                    cleaned_df, modifications = cleaner._short_term_cleaning(df, token_name)
                elif strategy == 'medium_term':
                    cleaned_df, modifications = cleaner._medium_term_cleaning(df, token_name)
                elif strategy == 'long_term':
                    cleaned_df, modifications = cleaner._long_term_cleaning(df, token_name)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                strategy_results['total_processed'] += 1
                
                if cleaned_df is not None and len(cleaned_df) >= 60:
                    # Save cleaned token
                    output_file = output_dir / f"{token_name}.parquet"
                    cleaned_df.write_parquet(output_file)
                    
                    strategy_results['cleaned_successfully'] += 1
                    
                    # Track cleaning stats
                    original_length = len(df)
                    final_length = len(cleaned_df)
                    retention_rate = final_length / original_length
                    
                    if 'retention_rates' not in strategy_results['cleaning_stats']:
                        strategy_results['cleaning_stats']['retention_rates'] = []
                    strategy_results['cleaning_stats']['retention_rates'].append(retention_rate)
                    
                else:
                    # Token was excluded
                    strategy_results['excluded_tokens'] += 1
                    
                    # Track exclusion reasons
                    for mod in modifications:
                        if mod['type'].startswith('excluded_'):
                            reason = mod['type']
                            if reason not in strategy_results['exclusion_reasons']:
                                strategy_results['exclusion_reasons'][reason] = 0
                            strategy_results['exclusion_reasons'][reason] += 1
                
            except Exception as e:
                print(f"❌ Error processing {token_file.name}: {e}")
                continue
        
        # Calculate summary statistics
        if strategy_results['cleaning_stats'].get('retention_rates'):
            retention_rates = strategy_results['cleaning_stats']['retention_rates']
            strategy_results['cleaning_stats']['avg_retention_rate'] = sum(retention_rates) / len(retention_rates)
            strategy_results['cleaning_stats']['min_retention_rate'] = min(retention_rates)
            strategy_results['cleaning_stats']['max_retention_rate'] = max(retention_rates)
        
        strategy_results['success_rate'] = (
            strategy_results['cleaned_successfully'] / strategy_results['total_processed'] 
            if strategy_results['total_processed'] > 0 else 0
        )
        
        results[strategy] = strategy_results
        
        # Print strategy summary
        print(f"\n📊 {strategy.upper()} STRATEGY RESULTS:")
        print(f"   ✅ Successfully cleaned: {strategy_results['cleaned_successfully']:,}")
        print(f"   ❌ Excluded tokens: {strategy_results['excluded_tokens']:,}")
        print(f"   📈 Success rate: {strategy_results['success_rate']:.1%}")
        if 'avg_retention_rate' in strategy_results['cleaning_stats']:
            print(f"   📏 Avg retention rate: {strategy_results['cleaning_stats']['avg_retention_rate']:.1%}")
        
        # Show top exclusion reasons
        if strategy_results['exclusion_reasons']:
            print(f"   🚫 Top exclusion reasons:")
            for reason, count in sorted(strategy_results['exclusion_reasons'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                print(f"      • {reason}: {count:,}")
        
        # Save strategy results
        results_file = output_dir / 'cleaning_results.json'
        with open(results_file, 'w') as f:
            json.dump(strategy_results, f, indent=2)
    
    return results


def compare_strategies(results: Dict[str, Dict]) -> None:
    """Compare results across different strategies"""
    
    print(f"\n{'='*60}")
    print("📊 STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    # Create comparison table
    print(f"{'Strategy':<15} {'Success Rate':<12} {'Avg Retention':<15} {'Tokens Cleaned':<15}")
    print(f"{'-'*60}")
    
    for strategy, result in results.items():
        success_rate = f"{result['success_rate']:.1%}"
        avg_retention = f"{result['cleaning_stats'].get('avg_retention_rate', 0):.1%}"
        tokens_cleaned = f"{result['cleaned_successfully']:,}"
        
        print(f"{strategy:<15} {success_rate:<12} {avg_retention:<15} {tokens_cleaned:<15}")
    
    print(f"\n💡 EXPECTED BEHAVIOR:")
    print(f"   🟢 Short-term: Highest success rate, highest retention (preserves micro-patterns)")
    print(f"   🟡 Medium-term: Moderate success rate, moderate retention (balanced)")
    print(f"   🔴 Long-term: Lowest success rate, lowest retention (aggressive cleaning)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Train models on each dataset:")
    print(f"      • Short-term models (15m-60m) → use data/cleaned_tokens_short_term/")
    print(f"      • Medium-term models (120m-360m) → use data/cleaned_tokens_medium_term/")
    print(f"      • Long-term models (720m+) → use data/cleaned_tokens_long_term/")
    print(f"   2. Compare performance improvements")
    print(f"   3. Implement ensemble approach combining best-performing models")


def main():
    parser = argparse.ArgumentParser(description='Generate graduated datasets for different time horizons')
    parser.add_argument('--limit', type=int, help='Limit number of tokens to process')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['short_term', 'medium_term', 'long_term'],
                       default=['short_term', 'medium_term', 'long_term'],
                       help='Cleaning strategies to generate')
    
    args = parser.parse_args()
    
    try:
        results = generate_graduated_datasets(
            limit=args.limit,
            strategies=args.strategies
        )
        
        compare_strategies(results)
        
        print(f"\n🎉 Graduated dataset generation complete!")
        print(f"📁 Check output directories:")
        for strategy in args.strategies:
            print(f"   • data/cleaned_tokens_{strategy}/")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 