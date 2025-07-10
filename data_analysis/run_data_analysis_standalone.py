"""
Standalone Data Analysis Runner for Memecoin Data
Runs comprehensive data analysis including quality analysis, price analysis, pattern detection,
and category export to processed/ folders without needing the Streamlit app
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (go up one level from data_analysis/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules (since we're inside the data_analysis package)
from data_analysis.data_loader import DataLoader
from data_analysis.data_quality import DataQualityAnalyzer
from data_analysis.price_analysis import PriceAnalyzer
from data_analysis.analyze_tokens import TokenAnalyzer
from data_analysis.export_utils import export_parquet_files


def run_data_quality_analysis(limit: int = None):
    """Run comprehensive data quality analysis"""
    print("\n" + "="*60)
    print("STEP 1: DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Initialize data loader and analyzer (uses data/raw/dataset)
    data_loader = DataLoader(subfolder="raw/dataset")
    analyzer = DataQualityAnalyzer()
    
    # Get all available tokens
    available_tokens = data_loader.get_available_tokens()
    print(f"Found {len(available_tokens)} tokens in raw data")
    
    if limit:
        available_tokens = available_tokens[:limit]
        print(f"Processing limited to {limit} tokens")
    
    # Analyze each token
    quality_reports = {}
    print("\nAnalyzing data quality for all tokens...")
    
    total_tokens = len(available_tokens)
    for i, token_info in enumerate(available_tokens):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total_tokens} tokens...")
        
        try:
            # Load token data
            df = data_loader.get_token_data(token_info['symbol'])
            if df is not None and not df.is_empty():
                # Analyze quality with outlier detection
                report = analyzer.analyze_single_file(df, token_info['symbol'])
                quality_reports[token_info['symbol']] = report
        except Exception as e:
            logger.error(f"Error analyzing {token_info['symbol']}: {e}")
            continue
    
    print(f"\nSuccessfully analyzed {len(quality_reports)} tokens")
    
    # Display summary statistics
    print_quality_summary(quality_reports, analyzer)
    
    return quality_reports


def print_quality_summary(quality_reports, analyzer):
    """Print comprehensive quality analysis summary"""
    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY")
    print("="*60)
    
    # Basic statistics
    total_tokens = len(quality_reports)
    avg_quality_score = sum(report['quality_score'] for report in quality_reports.values()) / total_tokens if total_tokens > 0 else 0
    
    print(f"📊 BASIC STATISTICS:")
    print(f"  Total tokens analyzed: {total_tokens:,}")
    print(f"  Average quality score: {avg_quality_score:.1f}/100")
    
    # Categorize tokens
    gap_tokens = analyzer.identify_tokens_with_gaps(quality_reports)
    normal_tokens = analyzer.identify_normal_behavior_tokens(quality_reports)
    extreme_tokens = analyzer.identify_extreme_tokens(quality_reports)
    dead_tokens = analyzer.identify_dead_tokens(quality_reports)
    
    print(f"\n🏷️  TOKEN CATEGORIZATION:")
    print(f"  Tokens with gaps (exclude from training): {len(gap_tokens):,}")
    print(f"  Normal behavior tokens (best for training): {len(normal_tokens):,}")
    print(f"  Extreme tokens (valuable patterns): {len(extreme_tokens):,}")
    print(f"  Dead tokens (completion data): {len(dead_tokens):,}")
    
    # Outlier analysis summary
    print_outlier_summary(quality_reports)
    
    # Quality distribution
    quality_distribution = {
        'excellent': len([r for r in quality_reports.values() if r['quality_score'] >= 90]),
        'good': len([r for r in quality_reports.values() if 70 <= r['quality_score'] < 90]),
        'fair': len([r for r in quality_reports.values() if 50 <= r['quality_score'] < 70]),
        'poor': len([r for r in quality_reports.values() if r['quality_score'] < 50])
    }
    
    print(f"\n📈 QUALITY DISTRIBUTION:")
    for category, count in quality_distribution.items():
        pct = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  {category.capitalize()}: {count:,} tokens ({pct:.1f}%)")
    
    # Gap analysis
    total_gaps = sum(report.get('gaps', {}).get('total_gaps', 0) for report in quality_reports.values())
    tokens_with_gaps = len([r for r in quality_reports.values() if r.get('gaps', {}).get('total_gaps', 0) > 0])
    
    print(f"\n🕳️  GAP ANALYSIS:")
    print(f"  Tokens with gaps: {tokens_with_gaps:,}")
    print(f"  Total gaps found: {total_gaps:,}")
    
    # Extreme movement analysis
    tokens_with_extremes = len([r for r in quality_reports.values() if r.get('is_extreme_token', False)])
    print(f"\n🚀 EXTREME MOVEMENT ANALYSIS:")
    print(f"  Tokens with extreme movements: {tokens_with_extremes:,}")


def print_outlier_summary(quality_reports):
    """Print outlier detection summary"""
    print(f"\n🎯 OUTLIER DETECTION SUMMARY:")
    
    # Collect outlier statistics
    outlier_stats = {
        'winsorization': {'total_outliers': 0, 'tokens_with_outliers': 0},
        'z_score': {'total_outliers': 0, 'tokens_with_outliers': 0},
        'iqr': {'total_outliers': 0, 'tokens_with_outliers': 0},
        'modified_z_score': {'total_outliers': 0, 'tokens_with_outliers': 0},
        'consensus': {'total_outliers': 0, 'tokens_with_outliers': 0}
    }
    
    total_tokens_analyzed = 0
    
    for token, report in quality_reports.items():
        outlier_analysis = report.get('outlier_analysis', {})
        if outlier_analysis.get('status') == 'success':
            total_tokens_analyzed += 1
            summary = outlier_analysis.get('summary', {})
            outlier_counts = summary.get('outlier_counts', {})
            
            # Aggregate statistics
            for method in ['winsorization', 'z_score', 'iqr', 'modified_z_score']:
                count = outlier_counts.get(method, 0)
                if count > 0:
                    outlier_stats[method]['total_outliers'] += count
                    outlier_stats[method]['tokens_with_outliers'] += 1
            
            # Consensus outliers
            consensus_count = summary.get('consensus_outliers', 0)
            if consensus_count > 0:
                outlier_stats['consensus']['total_outliers'] += consensus_count
                outlier_stats['consensus']['tokens_with_outliers'] += 1
    
    if total_tokens_analyzed > 0:
        print(f"  Tokens with outlier analysis: {total_tokens_analyzed:,}")
        for method, stats in outlier_stats.items():
            method_name = method.replace('_', ' ').title()
            print(f"  {method_name}: {stats['tokens_with_outliers']:,} tokens, {stats['total_outliers']:,} outliers")


def run_price_analysis(quality_reports, limit: int = None):
    """Run comprehensive price analysis"""
    print("\n" + "="*60)
    print("STEP 2: PRICE ANALYSIS")
    print("="*60)
    
    # Initialize data loader and price analyzer
    data_loader = DataLoader(subfolder="raw/dataset")
    price_analyzer = PriceAnalyzer()
    
    # Get tokens for analysis (prefer normal behavior tokens)
    analyzer = DataQualityAnalyzer()
    normal_tokens = analyzer.identify_normal_behavior_tokens(quality_reports)
    
    # If we have normal tokens, use those; otherwise use all tokens
    if normal_tokens:
        tokens_to_analyze = list(normal_tokens.keys())
        print(f"Analyzing {len(tokens_to_analyze)} normal behavior tokens for price patterns")
    else:
        tokens_to_analyze = list(quality_reports.keys())
        print(f"Analyzing all {len(tokens_to_analyze)} tokens for price patterns")
    
    if limit:
        tokens_to_analyze = tokens_to_analyze[:limit]
        print(f"Analysis limited to {limit} tokens")
    
    # Analyze prices
    price_metrics = {}
    print("\nAnalyzing price patterns...")
    
    total_tokens = len(tokens_to_analyze)
    for i, token in enumerate(tokens_to_analyze):
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{total_tokens} tokens...")
        
        try:
            # Load token data
            df = data_loader.get_token_data(token)
            if df is not None and not df.is_empty():
                # Analyze prices
                metrics = price_analyzer.analyze_prices(df, token)
                price_metrics[token] = metrics
        except Exception as e:
            logger.error(f"Error analyzing prices for {token}: {e}")
            continue
    
    print(f"\nSuccessfully analyzed prices for {len(price_metrics)} tokens")
    
    # Display price analysis summary
    print_price_summary(price_metrics)
    
    return price_metrics


def print_price_summary(price_metrics):
    """Print price analysis summary"""
    print("\n" + "="*60)
    print("PRICE ANALYSIS SUMMARY")
    print("="*60)
    
    if not price_metrics:
        print("No price metrics available")
        return
    
    # Collect statistics
    total_returns = []
    optimal_returns = []
    volatilities = []
    max_gains = []
    pattern_distribution = {}
    
    for token, metrics in price_metrics.items():
        price_stats = metrics.get('price_stats', {})
        patterns = metrics.get('patterns', {})
        volatility_metrics = metrics.get('volatility_metrics', {})
        optimal_metrics = metrics.get('optimal_return_metrics', {})
        
        # Collect data
        total_return = price_stats.get('total_return', 0) * 100  # Convert to percentage
        total_returns.append(total_return)
        
        optimal_return = optimal_metrics.get('optimal_return_pct', 0)
        optimal_returns.append(optimal_return)
        
        volatility = volatility_metrics.get('avg_volatility', 0) * 100
        volatilities.append(volatility)
        
        max_gain = patterns.get('max_gain', 0) * 100
        max_gains.append(max_gain)
        
        pattern = patterns.get('pattern', 'unknown')
        pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1
    
    # Calculate summary statistics
    import numpy as np
    
    print(f"📊 RETURN STATISTICS:")
    print(f"  Average total return: {np.mean(total_returns):.2f}%")
    print(f"  Median total return: {np.median(total_returns):.2f}%")
    print(f"  Best total return: {np.max(total_returns):.2f}%")
    print(f"  Worst total return: {np.min(total_returns):.2f}%")
    
    print(f"\n🎯 OPTIMAL TRADING STATISTICS:")
    print(f"  Average optimal return: {np.mean(optimal_returns):.2f}%")
    print(f"  Median optimal return: {np.median(optimal_returns):.2f}%")
    print(f"  Best optimal return: {np.max(optimal_returns):.2f}%")
    
    print(f"\n📈 VOLATILITY STATISTICS:")
    print(f"  Average volatility: {np.mean(volatilities):.2f}%")
    print(f"  Median volatility: {np.median(volatilities):.2f}%")
    print(f"  Highest volatility: {np.max(volatilities):.2f}%")
    
    print(f"\n🚀 MAX GAIN STATISTICS:")
    print(f"  Average max gain: {np.mean(max_gains):.2f}%")
    print(f"  Median max gain: {np.median(max_gains):.2f}%")
    print(f"  Highest max gain: {np.max(max_gains):.2f}%")
    
    print(f"\n🔍 PATTERN DISTRIBUTION:")
    total_patterns = len(price_metrics)
    for pattern, count in sorted(pattern_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_patterns) * 100
        print(f"  {pattern.replace('_', ' ').title()}: {count} tokens ({pct:.1f}%)")


def export_mutually_exclusive_categories(quality_reports):
    """
    Export all categories with strict mutual exclusivity enforcement.
    Each token appears in EXACTLY ONE category based on hierarchy: gaps > normal > extremes > dead
    
    Args:
        quality_reports: Dictionary of quality reports
        
    Returns:
        Dictionary with category names as keys and token lists as values
    """
    print("\n" + "="*60)
    print("STEP 4: CATEGORY EXPORT (MUTUALLY EXCLUSIVE)")
    print("="*60)
    print("🔄 Exporting categories with strict hierarchy: gaps > normal > extremes > dead")
    
    # Step 1: Categorize all tokens with strict hierarchy
    categorized_tokens = {
        'normal_behavior_tokens': [],
        'tokens_with_extremes': [],
        'dead_tokens': [],
        'tokens_with_gaps': []
    }
    
    overlap_stats = {
        'normal_also_extreme': 0,
        'normal_also_dead': 0,
        'normal_also_gaps': 0,
        'extreme_also_dead': 0,
        'extreme_also_gaps': 0,
        'dead_also_gaps': 0,
        'total_overlaps_resolved': 0
    }
    
    for token, report in quality_reports.items():
        # Check all characteristics
        is_extreme = report.get('is_extreme_token', False)
        is_dead = report.get('is_dead', False)
        
        # Check for significant gaps (many gaps OR large gaps)
        total_gaps = report.get('gaps', {}).get('total_gaps', 0)
        max_gap = report.get('gaps', {}).get('max_gap', 0)
        has_many_gaps = total_gaps > 5  # More than 5 gaps
        has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
        has_significant_gaps = has_many_gaps or has_large_gap
        
        # Check if token qualifies as normal behavior
        is_normal_behavior = not (is_extreme or is_dead or has_significant_gaps)
        
        # Track overlaps for statistics
        if is_normal_behavior and is_extreme:
            overlap_stats['normal_also_extreme'] += 1
        if is_normal_behavior and is_dead:
            overlap_stats['normal_also_dead'] += 1
        if is_normal_behavior and has_significant_gaps:
            overlap_stats['normal_also_gaps'] += 1
        if is_extreme and is_dead:
            overlap_stats['extreme_also_dead'] += 1
        if is_extreme and has_significant_gaps:
            overlap_stats['extreme_also_gaps'] += 1
        if is_dead and has_significant_gaps:
            overlap_stats['dead_also_gaps'] += 1
        
        # STRICT HIERARCHICAL ASSIGNMENT (each token goes to EXACTLY ONE category)
        # Priority: gaps > normal > extremes > dead
        if has_significant_gaps:
            categorized_tokens['tokens_with_gaps'].append(token)
            if is_normal_behavior or is_extreme or is_dead:
                overlap_stats['total_overlaps_resolved'] += 1
        elif is_normal_behavior:
            categorized_tokens['normal_behavior_tokens'].append(token)
            if is_extreme or is_dead:
                overlap_stats['total_overlaps_resolved'] += 1
        elif is_extreme:
            categorized_tokens['tokens_with_extremes'].append(token)
            if is_dead:
                overlap_stats['total_overlaps_resolved'] += 1
        elif is_dead:
            categorized_tokens['dead_tokens'].append(token)
    
    # Step 2: Display categorization summary
    print(f"\n📈 CATEGORIZATION RESULTS:")
    total_tokens = len(quality_reports)
    for category, tokens in categorized_tokens.items():
        pct = (len(tokens) / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  {category:25}: {len(tokens):,} tokens ({pct:.1f}%)")
    
    print(f"\n🔍 OVERLAP RESOLUTION:")
    print(f"  Normal tokens that were also extreme:   {overlap_stats['normal_also_extreme']:,}")
    print(f"  Normal tokens that were also dead:      {overlap_stats['normal_also_dead']:,}")
    print(f"  Normal tokens that also had gaps:       {overlap_stats['normal_also_gaps']:,}")
    print(f"  Extreme tokens that were also dead:     {overlap_stats['extreme_also_dead']:,}")
    print(f"  Extreme tokens that also had gaps:      {overlap_stats['extreme_also_gaps']:,}")
    print(f"  Dead tokens that also had gaps:         {overlap_stats['dead_also_gaps']:,}")
    print(f"  Total overlaps resolved:                {overlap_stats['total_overlaps_resolved']:,}")
    
    # Step 3: Export each category
    exported_results = {}
    
    for category, tokens in categorized_tokens.items():
        if tokens:
            try:
                # Map category names to export group names
                group_name_map = {
                    'normal_behavior_tokens': 'Normal Behavior Tokens',
                    'tokens_with_extremes': 'Tokens with Extremes',
                    'dead_tokens': 'Dead Tokens',
                    'tokens_with_gaps': 'Tokens with Gaps'
                }
                
                group_name = group_name_map[category]
                exported = export_parquet_files(tokens, group_name)
                exported_results[category] = exported
                
                print(f"✅ Exported {len(exported):,} tokens to data/processed/{category}/")
                
            except Exception as e:
                print(f"❌ Error exporting {category}: {e}")
                logger.error(f"Error exporting {category}: {e}")
                exported_results[category] = []
        else:
            print(f"⚠️  No tokens found for {category}")
            exported_results[category] = []
    
    print(f"\n✅ EXPORT COMPLETE - All categories are now mutually exclusive!")
    print(f"   Total tokens processed: {total_tokens:,}")
    total_exported = sum(len(tokens) for tokens in exported_results.values())
    print(f"   Total tokens exported: {total_exported:,}")
    
    # Display file paths for each category
    print(f"\n📁 EXPORTED TO:")
    for category, exported_tokens in exported_results.items():
        if exported_tokens:
            print(f"  data/processed/{category}/: {len(exported_tokens):,} tokens")
    
    return exported_results


def run_token_analysis(limit: int = None):
    """Run detailed token analysis"""
    print("\n" + "="*60)
    print("STEP 3: DETAILED TOKEN ANALYSIS")
    print("="*60)
    
    # Initialize token analyzer (uses data/raw)
    token_analyzer = TokenAnalyzer(data_subfolder="raw")
    
    # Run analysis
    results = token_analyzer.analyze_multiple_tokens(limit=limit)
    
    if results:
        # Generate summary report
        summary_df = token_analyzer.generate_summary_report(results)
        print(f"\nGenerated summary report with {len(summary_df)} tokens")
        
        # Save results
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save summary to CSV
        summary_df.write_csv(output_dir / "token_summary.csv")
        print(f"Saved summary to {output_dir / 'token_summary.csv'}")
        
        # Generate plots
        print("Generating analysis plots...")
        token_analyzer.plot_token_analysis(results, output_dir)
        print(f"Saved plots to {output_dir}/")
        
        return results
    else:
        print("No token analysis results generated")
        return {}


def save_results_to_files(quality_reports, price_metrics, token_results):
    """Save all analysis results to files"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save quality reports
    if quality_reports:
        with open(output_dir / "quality_reports.json", 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_reports = {}
            for token, report in quality_reports.items():
                serializable_report = {}
                for key, value in report.items():
                    if key == 'gaps' and isinstance(value, dict):
                        # Handle gap details with datetime objects
                        gap_details = value.get('gap_details', [])
                        serialized_gaps = []
                        for gap in gap_details:
                            serialized_gap = gap.copy()
                            if 'start_time' in serialized_gap:
                                serialized_gap['start_time'] = str(serialized_gap['start_time'])
                            serialized_gaps.append(serialized_gap)
                        serializable_report[key] = {**value, 'gap_details': serialized_gaps}
                    elif key == 'outlier_analysis' and isinstance(value, dict):
                        # Skip the processed_dataframe as it's not serializable
                        outlier_summary = value.get('summary', {})
                        serializable_report[key] = {
                            'summary': outlier_summary,
                            'status': value.get('status', 'unknown')
                        }
                    else:
                        serializable_report[key] = value
                serializable_reports[token] = serializable_report
            
            json.dump(serializable_reports, f, indent=2, default=str)
        print(f"Saved quality reports to {output_dir / 'quality_reports.json'}")
    
    # Save price metrics
    if price_metrics:
        with open(output_dir / "price_metrics.json", 'w') as f:
            json.dump(price_metrics, f, indent=2, default=str)
        print(f"Saved price metrics to {output_dir / 'price_metrics.json'}")
    
    # Save token analysis results
    if token_results:
        with open(output_dir / "token_analysis.json", 'w') as f:
            # Convert Polars DataFrames to dicts for serialization
            serializable_results = {}
            for token, result in token_results.items():
                serializable_result = result.copy()
                if 'data' in serializable_result and serializable_result['data'] is not None:
                    try:
                        serializable_result['data'] = serializable_result['data'].to_pandas().to_dict()
                    except:
                        serializable_result['data'] = None
                serializable_results[token] = serializable_result
            
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"Saved token analysis to {output_dir / 'token_analysis.json'}")
    
    print(f"\nAll results saved to {output_dir}/")


def main():
    """Run the complete standalone data analysis"""
    start_time = datetime.now()
    
    print("\n" + "🔬"*20)
    print("MEMECOIN DATA ANALYSIS RUNNER")
    print("🔬"*20)
    print(f"\nStarting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize data loader to get the path
    data_loader = DataLoader(subfolder="raw/dataset")
    print(f"Raw data path: {data_loader.base_path}")
    
    # Configuration - adjust these limits for testing
    LIMIT_QUALITY = None      # None for all tokens, or set a number like 100 for testing
    LIMIT_PRICE = None        # None for all tokens (removed 50-token limit)
    LIMIT_TOKEN = 20          # Limit detailed token analysis to 20 tokens
    
    # Track results
    quality_reports = {}
    price_metrics = {}
    token_results = {}
    export_results = {}
    
    try:
        # Step 1: Data Quality Analysis
        quality_reports = run_data_quality_analysis(limit=LIMIT_QUALITY)
        
        # Step 2: Price Analysis
        if quality_reports:
            price_metrics = run_price_analysis(quality_reports, limit=LIMIT_PRICE)
        
        # Step 3: Detailed Token Analysis
        token_results = run_token_analysis(limit=LIMIT_TOKEN)
        
        # Step 4: Export Categories to processed/ folders
        if quality_reports:
            export_choice = input("\n🗂️  Export categories to processed/ folders? (y/n): ").lower().strip()
            if export_choice in ['y', 'yes']:
                export_results = export_mutually_exclusive_categories(quality_reports)
            else:
                print("📝 Category export skipped (skipped by user choice)")
        
        # Step 5: Save all results (with user confirmation)
        if quality_reports or price_metrics or token_results:
            save_choice = input("\n💾 Save analysis results to files? (y/n): ").lower().strip()
            if save_choice in ['y', 'yes']:
                save_results_to_files(quality_reports, price_metrics, token_results)
            else:
                print("📝 Analysis results not saved (skipped by user choice)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed with error: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Duration: {duration}")
    print(f"Quality reports: {len(quality_reports)} tokens")
    print(f"Price metrics: {len(price_metrics)} tokens")
    print(f"Token analysis: {len(token_results)} tokens")
    
    # Summary of exported categories
    if export_results:
        total_exported = sum(len(tokens) for tokens in export_results.values())
        print(f"Categories exported: {total_exported} tokens")
        for category, tokens in export_results.items():
            if tokens:
                print(f"  {category}: {len(tokens)} tokens")
    
    if quality_reports or price_metrics or token_results:
        print("\n✅ Analysis completed successfully!")
        print("\n📁 Output files:")
        print("  - analysis_results/quality_reports.json")
        print("  - analysis_results/price_metrics.json")
        print("  - analysis_results/token_analysis.json")
        print("  - analysis_results/token_summary.csv")
        print("  - analysis_results/*.png (plots)")
        
        # Show processed categories if exported
        if export_results:
            print("\n📁 Processed categories:")
            for category, tokens in export_results.items():
                if tokens:
                    print(f"  - data/processed/{category}/: {len(tokens)} tokens")
        
        print("\n🚀 Next steps:")
        print("  1. Review the analysis results")
        if export_results:
            print("  2. Categories ready for ML training in data/processed/")
            print("  3. Run data cleaning on exported categories")
            print("  4. Proceed with feature engineering")
            print("  5. Train ML models on clean data")
        else:
            print("  2. Export categories to processed/ folders")
            print("  3. Run data cleaning on identified categories")
            print("  4. Proceed with feature engineering")
            print("  5. Train ML models on clean data")
    else:
        print("\n⚠️ No results generated. Please check the data path and try again.")


if __name__ == "__main__":
    main() 