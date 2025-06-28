#!/usr/bin/env python3
"""
Token Overlap Analyzer for Processed Data Folders

This script analyzes the overlap between different categories of processed tokens
and provides detailed statistics about token distribution across folders.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import json
from collections import defaultdict, Counter

class TokenOverlapAnalyzer:
    def __init__(self, processed_base_path: str = None):
        """Initialize the analyzer with the processed data path"""
        if processed_base_path is None:
            self.processed_base = Path(__file__).parent.parent / "data" / "processed"
        else:
            self.processed_base = Path(processed_base_path)
        
        self.folders = [
            "normal_behavior_tokens",
            "dead_tokens", 
            "tokens_with_gaps",
            "tokens_with_extremes"
        ]
        
        # Cache for token sets
        self._token_sets_cache = {}
    
    def get_tokens_in_folder(self, folder_name: str) -> Set[str]:
        """Get set of token names (without .parquet extension) in a folder"""
        if folder_name in self._token_sets_cache:
            return self._token_sets_cache[folder_name]
        
        folder_path = self.processed_base / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return set()
        
        tokens = set()
        for file_path in folder_path.glob("*.parquet"):
            token_name = file_path.stem  # Remove .parquet extension
            tokens.add(token_name)
        
        self._token_sets_cache[folder_name] = tokens
        return tokens
    
    def get_all_folder_stats(self) -> Dict[str, int]:
        """Get count of tokens in each folder"""
        stats = {}
        for folder in self.folders:
            tokens = self.get_tokens_in_folder(folder)
            stats[folder] = len(tokens)
        return stats
    
    def quick_overlap_check(self, folder1: str, folder2: str = None) -> Dict:
        """Quick overlap check between two folders or one folder against all others"""
        if folder2:
            # Compare two specific folders
            return self._compare_two_folders(folder1, folder2)
        else:
            # Compare folder1 with all other folders
            return self._compare_folder_with_all(folder1)
    
    def _compare_two_folders(self, folder1: str, folder2: str) -> Dict:
        """Compare two specific folders"""
        tokens1 = self.get_tokens_in_folder(folder1)
        tokens2 = self.get_tokens_in_folder(folder2)
        
        if not tokens1 and not tokens2:
            return {'error': f"Both folders '{folder1}' and '{folder2}' are empty or don't exist"}
        
        overlap = tokens1.intersection(tokens2)
        
        result = {
            'folder1': folder1,
            'folder2': folder2,
            'folder1_total': len(tokens1),
            'folder2_total': len(tokens2),
            'overlap_count': len(overlap),
            'overlap_pct_folder1': (len(overlap) / len(tokens1) * 100) if tokens1 else 0,
            'overlap_pct_folder2': (len(overlap) / len(tokens2) * 100) if tokens2 else 0,
            'unique_to_folder1': len(tokens1 - tokens2),
            'unique_to_folder2': len(tokens2 - tokens1),
            'sample_overlap': list(overlap)[:5]  # First 5 for display
        }
        
        return result
    
    def _compare_folder_with_all(self, folder1: str) -> Dict:
        """Compare folder1 with all other folders"""
        tokens1 = self.get_tokens_in_folder(folder1)
        
        if not tokens1:
            return {'error': f"Folder '{folder1}' is empty or doesn't exist"}
        
        results = {
            'base_folder': folder1,
            'base_folder_total': len(tokens1),
            'comparisons': {}
        }
        
        for folder in self.folders:
            if folder == folder1:
                continue
            
            comparison = self._compare_two_folders(folder1, folder)
            results['comparisons'][folder] = comparison
        
        return results
    
    def print_quick_overlap_results(self, results: Dict):
        """Print quick overlap results in a formatted way"""
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        if 'comparisons' in results:
            # Multiple comparisons
            print("🔍 QUICK TOKEN OVERLAP CHECKER")
            print("=" * 50)
            print(f"📁 Processed data path: {self.processed_base}")
            print(f"\n🎯 COMPARING '{results['base_folder'].upper()}' WITH ALL OTHER FOLDERS")
            print("=" * 70)
            
            for folder, comparison in results['comparisons'].items():
                self._print_single_comparison(comparison)
            
            # Summary table
            print(f"\n📋 SUMMARY TABLE")
            print("=" * 70)
            print(f"{'Folder':<25} {'Total':<8} {'Overlap':<8} {'% Overlap':<10} {'Unique':<8}")
            print("-" * 70)
            
            for folder, comparison in results['comparisons'].items():
                print(f"{folder:<25} {comparison['folder2_total']:<8,} {comparison['overlap_count']:<8,} "
                      f"{comparison['overlap_pct_folder2']:<10.1f} {comparison['unique_to_folder2']:<8,}")
        else:
            # Single comparison
            self._print_single_comparison(results)
    
    def _print_single_comparison(self, result: Dict):
        """Print a single folder comparison"""
        print(f"\n📊 OVERLAP ANALYSIS: {result['folder1'].upper()} vs {result['folder2'].upper()}")
        print("=" * 60)
        print(f"{result['folder1']:20}: {result['folder1_total']:,} tokens")
        print(f"{result['folder2']:20}: {result['folder2_total']:,} tokens")
        print(f"{'Overlap':20}: {result['overlap_count']:,} tokens")
        
        if result['folder1_total'] > 0:
            print(f"{'% of ' + result['folder1']:20}: {result['overlap_pct_folder1']:.1f}%")
        if result['folder2_total'] > 0:
            print(f"{'% of ' + result['folder2']:20}: {result['overlap_pct_folder2']:.1f}%")
        
        print(f"{'Unique to ' + result['folder1']:20}: {result['unique_to_folder1']:,} tokens")
        print(f"{'Unique to ' + result['folder2']:20}: {result['unique_to_folder2']:,} tokens")
        
        if result['sample_overlap']:
            print(f"{'Sample overlap':20}: {', '.join(result['sample_overlap'][:3])}")

    def analyze_overlap_with_normal_behavior(self) -> Dict[str, Dict]:
        """Analyze overlap of each folder with normal_behavior_tokens"""
        normal_behavior_tokens = self.get_tokens_in_folder("normal_behavior_tokens")
        
        results = {}
        for folder in self.folders:
            if folder == "normal_behavior_tokens":
                continue
                
            folder_tokens = self.get_tokens_in_folder(folder)
            overlap = normal_behavior_tokens.intersection(folder_tokens)
            
            results[folder] = {
                'total_tokens': len(folder_tokens),
                'overlap_with_normal_behavior': len(overlap),
                'overlap_percentage': (len(overlap) / len(folder_tokens) * 100) if folder_tokens else 0,
                'unique_to_folder': len(folder_tokens - normal_behavior_tokens),
                'overlap_tokens': list(overlap)[:10]  # First 10 for display
            }
        
        return results
    
    def analyze_all_overlaps(self) -> Dict[str, Dict[str, Dict]]:
        """Analyze overlap between all pairs of folders"""
        results = {}
        
        for i, folder1 in enumerate(self.folders):
            results[folder1] = {}
            tokens1 = self.get_tokens_in_folder(folder1)
            
            for j, folder2 in enumerate(self.folders):
                if i >= j:  # Skip diagonal and already computed pairs
                    continue
                    
                tokens2 = self.get_tokens_in_folder(folder2)
                overlap = tokens1.intersection(tokens2)
                
                results[folder1][folder2] = {
                    'overlap_count': len(overlap),
                    'folder1_total': len(tokens1),
                    'folder2_total': len(tokens2),
                    'overlap_pct_folder1': (len(overlap) / len(tokens1) * 100) if tokens1 else 0,
                    'overlap_pct_folder2': (len(overlap) / len(tokens2) * 100) if tokens2 else 0,
                    'sample_tokens': list(overlap)[:5]  # First 5 for display
                }
        
        return results
    
    def find_tokens_in_multiple_categories(self) -> Dict[str, List[str]]:
        """Find tokens that appear in multiple categories"""
        token_to_folders = defaultdict(list)
        
        for folder in self.folders:
            tokens = self.get_tokens_in_folder(folder)
            for token in tokens:
                token_to_folders[token].append(folder)
        
        # Group by number of categories
        multi_category_tokens = {}
        for token, folders in token_to_folders.items():
            if len(folders) > 1:
                count = len(folders)
                if count not in multi_category_tokens:
                    multi_category_tokens[count] = []
                multi_category_tokens[count].append({
                    'token': token,
                    'categories': folders
                })
        
        return multi_category_tokens
    
    def get_unique_tokens_per_folder(self) -> Dict[str, Dict]:
        """Find tokens that are unique to each folder"""
        all_tokens_by_folder = {folder: self.get_tokens_in_folder(folder) for folder in self.folders}
        
        results = {}
        for folder in self.folders:
            folder_tokens = all_tokens_by_folder[folder]
            other_tokens = set()
            
            for other_folder in self.folders:
                if other_folder != folder:
                    other_tokens.update(all_tokens_by_folder[other_folder])
            
            unique_tokens = folder_tokens - other_tokens
            results[folder] = {
                'unique_count': len(unique_tokens),
                'total_count': len(folder_tokens),
                'unique_percentage': (len(unique_tokens) / len(folder_tokens) * 100) if folder_tokens else 0,
                'sample_unique_tokens': list(unique_tokens)[:10]
            }
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive overlap analysis report"""
        report = []
        report.append("=" * 80)
        report.append("TOKEN OVERLAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # 1. Folder Statistics
        report.append("1. FOLDER STATISTICS")
        report.append("-" * 40)
        folder_stats = self.get_all_folder_stats()
        for folder, count in folder_stats.items():
            report.append(f"{folder:25}: {count:,} tokens")
        report.append(f"{'TOTAL UNIQUE TOKENS':25}: {len(set().union(*[self.get_tokens_in_folder(f) for f in self.folders])):,}")
        report.append("")
        
        # 2. Overlap with Normal Behavior Tokens
        report.append("2. OVERLAP WITH NORMAL BEHAVIOR TOKENS")
        report.append("-" * 40)
        nb_overlaps = self.analyze_overlap_with_normal_behavior()
        for folder, data in nb_overlaps.items():
            report.append(f"\n{folder.upper()}:")
            report.append(f"  Total tokens: {data['total_tokens']:,}")
            report.append(f"  Overlap with normal behavior: {data['overlap_with_normal_behavior']:,} ({data['overlap_percentage']:.1f}%)")
            report.append(f"  Unique to {folder}: {data['unique_to_folder']:,}")
            if data['overlap_tokens']:
                report.append(f"  Sample overlapping tokens: {', '.join(data['overlap_tokens'][:3])}")
        report.append("")
        
        # 3. All Pairwise Overlaps
        report.append("3. PAIRWISE OVERLAPS BETWEEN ALL FOLDERS")
        report.append("-" * 40)
        all_overlaps = self.analyze_all_overlaps()
        for folder1, folder2_data in all_overlaps.items():
            for folder2, overlap_data in folder2_data.items():
                report.append(f"\n{folder1.upper()} ↔ {folder2.upper()}:")
                report.append(f"  Overlap: {overlap_data['overlap_count']:,} tokens")
                report.append(f"  {overlap_data['overlap_pct_folder1']:.1f}% of {folder1}, {overlap_data['overlap_pct_folder2']:.1f}% of {folder2}")
        report.append("")
        
        # 4. Multi-Category Tokens
        report.append("4. TOKENS IN MULTIPLE CATEGORIES")
        report.append("-" * 40)
        multi_tokens = self.find_tokens_in_multiple_categories()
        for count in sorted(multi_tokens.keys(), reverse=True):
            tokens_data = multi_tokens[count]
            report.append(f"\nTokens in {count} categories: {len(tokens_data)}")
            for token_data in tokens_data[:5]:  # Show first 5
                report.append(f"  {token_data['token']}: {', '.join(token_data['categories'])}")
            if len(tokens_data) > 5:
                report.append(f"  ... and {len(tokens_data) - 5} more")
        report.append("")
        
        # 5. Unique Tokens
        report.append("5. TOKENS UNIQUE TO EACH FOLDER")
        report.append("-" * 40)
        unique_tokens = self.get_unique_tokens_per_folder()
        for folder, data in unique_tokens.items():
            report.append(f"\n{folder.upper()}:")
            report.append(f"  Unique tokens: {data['unique_count']:,} ({data['unique_percentage']:.1f}% of folder)")
        report.append("")
        
        # 6. Data Quality Insights
        report.append("6. DATA QUALITY INSIGHTS")
        report.append("-" * 40)
        
        # Normal behavior vs other categories
        normal_tokens = self.get_tokens_in_folder("normal_behavior_tokens")
        dead_tokens = self.get_tokens_in_folder("dead_tokens")
        extreme_tokens = self.get_tokens_in_folder("tokens_with_extremes")
        
        if normal_tokens:
            normal_dead_overlap = normal_tokens.intersection(dead_tokens)
            normal_extreme_overlap = normal_tokens.intersection(extreme_tokens)
            
            report.append(f"Normal behavior tokens that are also dead: {len(normal_dead_overlap):,}")
            report.append(f"Normal behavior tokens with extremes: {len(normal_extreme_overlap):,}")
            report.append(f"  This suggests {len(normal_dead_overlap)/len(normal_tokens)*100:.1f}% of normal tokens became dead" if normal_tokens else "")
        
        # Extreme tokens analysis
        if extreme_tokens and dead_tokens:
            dead_extreme_overlap = dead_tokens.intersection(extreme_tokens)
            report.append(f"Dead tokens with extremes: {len(dead_extreme_overlap):,} ({len(dead_extreme_overlap)/len(extreme_tokens)*100:.1f}% of extreme tokens)")
            report.append(f"Extreme tokens that died: {len(dead_extreme_overlap):,} ({len(dead_extreme_overlap)/len(dead_tokens)*100:.1f}% of dead tokens)")
        
        return "\n".join(report)
    
    def save_detailed_overlap_data(self, output_path: str = None):
        """Save detailed overlap data to JSON file"""
        if output_path is None:
            output_path = self.processed_base / "token_overlap_analysis.json"
        
        data = {
            'folder_stats': self.get_all_folder_stats(),
            'normal_behavior_overlaps': self.analyze_overlap_with_normal_behavior(),
            'all_overlaps': self.analyze_all_overlaps(),
            'multi_category_tokens': self.find_tokens_in_multiple_categories(),
            'unique_tokens': self.get_unique_tokens_per_folder()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Detailed overlap data saved to: {output_path}")

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze token overlaps between processed folders')
    parser.add_argument('--mode', choices=['quick', 'comprehensive'], default='comprehensive',
                        help='Analysis mode: quick for fast comparisons, comprehensive for detailed analysis')
    parser.add_argument('--folder1', default='normal_behavior_tokens', 
                        help='First folder to compare (for quick mode)')
    parser.add_argument('--folder2', help='Second folder to compare (if not specified, compares with all folders)')
    parser.add_argument('--processed-path', default=None, help='Path to processed data folder')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TokenOverlapAnalyzer(args.processed_path)
    
    if args.mode == 'quick':
        # Quick overlap check mode
        available_folders = [f for f in analyzer.folders if (analyzer.processed_base / f).exists()]
        
        print("🔍 QUICK TOKEN OVERLAP CHECKER")
        print("=" * 50)
        print(f"📁 Processed data path: {analyzer.processed_base}")
        print(f"📂 Available folders: {', '.join(available_folders)}")
        
        if args.folder1 not in available_folders:
            print(f"❌ Error: Folder '{args.folder1}' does not exist")
            return
        
        if args.folder2 and args.folder2 not in available_folders:
            print(f"❌ Error: Folder '{args.folder2}' does not exist")
            return
        
        # Perform quick overlap check
        results = analyzer.quick_overlap_check(args.folder1, args.folder2)
        analyzer.print_quick_overlap_results(results)
        
    else:
        # Comprehensive analysis mode
        print("Analyzing token overlaps between processed folders...")
        print("This may take a few moments...\n")
        
        # Generate and print the report
        report = analyzer.generate_comprehensive_report()
        print(report)
        
        # Save detailed data
        analyzer.save_detailed_overlap_data()
        
        print("\n" + "=" * 80)
        print("Analysis complete! Check token_overlap_analysis.json for detailed data.")

if __name__ == "__main__":
    main() 