"""
Test the refactored cleaning process to see how many tokens we preserve
"""

import os
import sys
from pathlib import Path
import polars as pl
from clean_tokens import clean_all_categories

def analyze_cleaning_results():
    """Run cleaning and analyze the results"""
    
    print("="*60)
    print("TESTING REFACTORED DATA CLEANING")
    print("="*60)
    
    # Run cleaning on a sample (limit to speed up testing)
    print("\nRunning cleaning on sample tokens...")
    summary = clean_all_categories(limit_per_category=100)  # Test with 100 tokens per category
    
    # Analyze results
    print("\n" + "="*60)
    print("CLEANING RESULTS SUMMARY")
    print("="*60)
    
    total_processed = summary['total_files_processed']
    total_cleaned = summary['total_successfully_cleaned']
    total_errors = summary['total_errors']
    success_rate = summary['overall_success_rate']
    
    print(f"\nOverall Statistics:")
    print(f"  Total tokens processed: {total_processed}")
    print(f"  Successfully cleaned: {total_cleaned}")
    print(f"  Errors/Excluded: {total_errors}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Retention rate: {(total_cleaned/total_processed)*100:.1f}%")
    
    print("\nPer-Category Breakdown:")
    for category, result in summary['category_results'].items():
        if 'total_files_processed' in result:
            cat_total = result['total_files_processed']
            cat_success = result['successfully_cleaned']
            cat_rate = (cat_success/cat_total)*100 if cat_total > 0 else 0
            print(f"\n  {category}:")
            print(f"    Processed: {cat_total}")
            print(f"    Cleaned: {cat_success}")
            print(f"    Retention rate: {cat_rate:.1f}%")
    
    # If we want to see more details, load the cleaning logs
    cleaned_base = Path("data/cleaned")
    if cleaned_base.exists():
        print("\n" + "="*60)
        print("EXCLUSION REASONS ANALYSIS")
        print("="*60)
        
        for category in ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens", "tokens_with_gaps"]:
            log_path = cleaned_base / f"{category}_cleaning_log.json"
            if log_path.exists():
                log_df = pl.read_json(log_path)
                
                # Count exclusion reasons
                excluded_df = log_df.filter(pl.col('status') != 'cleaned_successfully')
                if len(excluded_df) > 0:
                    print(f"\n{category} - Exclusion Reasons:")
                    
                    # Group by status
                    status_counts = excluded_df.group_by('status').agg(
                        pl.count().alias('count')
                    ).sort('count', descending=True)
                    
                    for row in status_counts.iter_rows(named=True):
                        print(f"  {row['status']}: {row['count']}")
    
    return summary

def compare_token_counts():
    """Compare token counts before and after cleaning"""
    
    print("\n" + "="*60)
    print("TOKEN COUNT COMPARISON")
    print("="*60)
    
    processed_base = Path("data/processed")
    cleaned_base = Path("data/cleaned")
    
    for category in ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens", "tokens_with_gaps"]:
        processed_dir = processed_base / category
        cleaned_dir = cleaned_base / category
        
        if processed_dir.exists() and cleaned_dir.exists():
            processed_count = len(list(processed_dir.glob("*.parquet")))
            cleaned_count = len(list(cleaned_dir.glob("*.parquet")))
            retention = (cleaned_count/processed_count)*100 if processed_count > 0 else 0
            
            print(f"\n{category}:")
            print(f"  Original: {processed_count} tokens")
            print(f"  After cleaning: {cleaned_count} tokens")
            print(f"  Retention: {retention:.1f}%")

if __name__ == "__main__":
    # Test the refactored cleaning
    summary = analyze_cleaning_results()
    
    # Compare counts
    compare_token_counts()
    
    print("\n✅ Test complete!")
    print("\nExpected improvements with ENHANCED staircase detection:")
    print("- Higher retention rate (targeting >80% vs previous ~25%)")
    print("- Preserved extreme pumps (1,000,000x+ gains)")
    print("- Enhanced staircase detection catches BOTH patterns:")
    print("  • Type A: jump→flat (vertical then horizontal)")
    print("  • Type B: flat→jump (horizontal then vertical)")
    print("- Removed only true staircase artifacts (checked pre+post jump)")
    print("- More lenient on price variations")
    print("- Smarter cut points based on staircase type") 