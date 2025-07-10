#!/usr/bin/env python3
"""
Gap Investigation and Data Cleaning Workflow

This script provides a complete workflow to:
1. Investigate tokens with gaps
2. Get recommendations (keep/remove)
3. Automatically clean tokens based on recommendations

Usage:
    python investigate_and_clean_gaps.py [--limit N] [--auto-clean]
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from ..data_analysis.data_quality import DataQualityAnalyzer
from ..data_analysis.data_loader import DataLoader
from ..data_cleaning.clean_tokens import clean_all_categories_with_investigation


def main():
    parser = argparse.ArgumentParser(description="Investigate tokens with gaps and optionally clean them")
    parser.add_argument('--limit', type=int, help='Limit number of tokens to analyze')
    parser.add_argument('--auto-clean', action='store_true', help='Automatically proceed with cleaning based on recommendations')
    parser.add_argument('--save-results', type=str, help='Save investigation results to JSON file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GAP INVESTIGATION AND CLEANING WORKFLOW")
    print("="*70)
    
    # Step 1: Load data and run quality analysis
    print("\n🔍 STEP 1: Loading data and analyzing quality...")
    
    data_loader = DataLoader()
    available_tokens = data_loader.get_available_tokens()
    
    if args.limit:
        available_tokens = available_tokens[:args.limit]
        print(f"Limited analysis to {len(available_tokens)} tokens")
    
    print(f"Analyzing {len(available_tokens)} tokens...")
    
    # Get parquet files for analysis
    parquet_files = []
    for token_info in available_tokens:
        file_path = Path(token_info['file'])
        if file_path.exists():
            parquet_files.append(file_path)
    
    if not parquet_files:
        print("❌ No valid token files found!")
        return
    
    # Run quality analysis
    quality_analyzer = DataQualityAnalyzer()
    quality_df = quality_analyzer.analyze_multiple_files(parquet_files, limit=args.limit)
    
    # Convert to dictionary format for gap investigation
    quality_reports = {}
    for row in quality_df.iter_rows(named=True):
        quality_reports[row['token']] = row
    
    # Step 2: Investigate tokens with gaps
    print("\n🔬 STEP 2: Investigating tokens with gaps...")
    
    investigation_results = quality_analyzer.investigate_tokens_with_gaps(quality_reports)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(investigation_results, f, indent=2, default=str)
        print(f"Investigation results saved to: {args.save_results}")
    
    # Step 3: Ask user about cleaning or auto-clean
    if investigation_results['tokens_analyzed'] == 0:
        print("\n✅ No tokens with significant gaps found. No cleaning needed!")
        return
    
    print(f"\n🎯 INVESTIGATION COMPLETE")
    print(f"Found {investigation_results['tokens_analyzed']} tokens with gaps:")
    
    for action, tokens in investigation_results['recommendations'].items():
        if tokens:
            action_name = action.replace('_', ' ').title()
            print(f"  {action_name}: {len(tokens)} tokens")
    
    # Decide whether to proceed with cleaning
    proceed_with_cleaning = args.auto_clean
    
    if not proceed_with_cleaning:
        print(f"\n🤔 Do you want to proceed with data cleaning based on these recommendations?")
        print(f"   - Tokens marked 'remove_completely' will be excluded from cleaning")
        print(f"   - Tokens marked 'keep_and_clean' will be cleaned with aggressive gap filling")
        print(f"   - Tokens marked 'needs_manual_review' will be included (you can review them later)")
        
        response = input("\nProceed with cleaning? (y/n): ").lower().strip()
        proceed_with_cleaning = response in ['y', 'yes']
    
    if not proceed_with_cleaning:
        print("\n⏸️ Cleaning skipped. You can run data cleaning manually later.")
        print("To use these results in cleaning:")
        print("1. Save the investigation results to a JSON file")
        print("2. Load them in your cleaning script")
        print("3. Use clean_all_categories_with_investigation() function")
        return
    
    # Step 3: Clean data based on recommendations
    print(f"\n🧹 STEP 3: Cleaning data based on recommendations...")
    
    cleaning_results = clean_all_categories_with_investigation(
        investigation_results=investigation_results,
        limit_per_category=args.limit
    )
    
    # Final summary
    print(f"\n✅ WORKFLOW COMPLETE!")
    print(f"Investigation and cleaning finished successfully.")
    
    total_excluded = sum(r.get('tokens_excluded_by_investigation', 0) for r in cleaning_results.values())
    total_cleaned = sum(r.get('successful_cleanings', 0) for r in cleaning_results.values())
    
    print(f"\nFinal Results:")
    print(f"  Tokens excluded based on gap analysis: {total_excluded}")
    print(f"  Tokens successfully cleaned: {total_cleaned}")
    print(f"  Cleaned data available in: data/cleaned/")


if __name__ == "__main__":
    main() 