#!/usr/bin/env python3
"""
Add Archetype Columns to Token Data

Adds 'category' and 'cluster' columns to the original token data files
based on the archetype characterization results.

Usage:
    python add_archetype_columns.py --archetype-results PATH_TO_RESULTS --data-dir PATH_TO_DATA --output-dir PATH_TO_OUTPUT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import polars as pl
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class ArchetypeColumnAdder:
    """Add archetype columns to token data."""
    
    def __init__(self):
        self.archetype_data = {}
        self.token_labels = {}
        
    def load_archetype_results(self, archetype_results_path: Path) -> None:
        """Load archetype characterization results."""
        print(f"📊 Loading archetype results from: {archetype_results_path}")
        
        with open(archetype_results_path, 'r') as f:
            results = json.load(f)
        
        self.archetype_data = results.get('archetype_data', {})
        
        # Create token-to-label mapping
        self.token_labels = {}
        for category, category_archetypes in self.archetype_data.items():
            for archetype_name, archetype_info in category_archetypes.items():
                cluster_id = archetype_info.get('cluster_id', 0)
                tokens = archetype_info.get('tokens', [])
                
                for token in tokens:
                    self.token_labels[token] = {
                        'category': category,
                        'cluster': cluster_id,
                        'archetype': archetype_name
                    }
        
        print(f"📈 Loaded labels for {len(self.token_labels)} tokens")
        print(f"📊 Categories: {list(self.archetype_data.keys())}")
        
        # Print category distribution
        category_counts = {}
        cluster_counts = {}
        
        for token, labels in self.token_labels.items():
            category = labels['category']
            cluster = labels['cluster']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            cluster_key = f"{category}_cluster_{cluster}"
            cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        
        print(f"📊 Category distribution: {category_counts}")
        print(f"📊 Cluster distribution: {dict(sorted(cluster_counts.items()))}")
        
    def process_token_files(self, data_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Process all token files and add archetype columns."""
        print(f"📁 Processing token files from: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create single output directory (no subdirectories)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'labeled_files': 0,
            'unlabeled_files': 0,
            'errors': 0
        }
        
        # Collect all token files from all subdirectories
        all_token_files = []
        
        # Process each data category directory
        for category_name in ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes', 'tokens_with_gaps']:
            category_path = data_dir / category_name
            if not category_path.exists():
                continue
                
            print(f"\n  Collecting {category_name}...")
            
            # Get all parquet files
            token_files = list(category_path.glob("*.parquet"))
            all_token_files.extend(token_files)
            stats['total_files'] += len(token_files)
            print(f"    Found {len(token_files)} files")
        
        print(f"\n📊 Processing {len(all_token_files)} total files...")
        
        # Process all files with progress bar
        for token_file in tqdm(all_token_files, desc="  Processing tokens"):
            token_name = token_file.stem
            output_file = output_dir / token_file.name
            
            try:
                # Load the token data
                df = pl.read_parquet(token_file)
                
                # Add archetype columns
                if token_name in self.token_labels:
                    # Token has archetype labels
                    labels = self.token_labels[token_name]
                    df = df.with_columns([
                        pl.lit(labels['category']).alias('category'),
                        pl.lit(labels['cluster']).alias('cluster'),
                        pl.lit(labels['archetype']).alias('archetype')
                    ])
                    stats['labeled_files'] += 1
                else:
                    # Token doesn't have archetype labels
                    df = df.with_columns([
                        pl.lit(None).alias('category'),
                        pl.lit(None).alias('cluster'),
                        pl.lit(None).alias('archetype')
                    ])
                    stats['unlabeled_files'] += 1
                
                # Save the updated data to single directory
                df.write_parquet(output_file)
                stats['processed_files'] += 1
                
            except Exception as e:
                print(f"    ❌ Error processing {token_name}: {e}")
                stats['errors'] += 1
                continue
        
        return stats
    
    def create_summary_report(self, stats: Dict[str, int], output_dir: Path) -> None:
        """Create a summary report of the processing."""
        print(f"\n📊 Creating summary report...")
        
        # Calculate percentages
        total_files = stats['total_files']
        labeled_pct = (stats['labeled_files'] / total_files * 100) if total_files > 0 else 0
        unlabeled_pct = (stats['unlabeled_files'] / total_files * 100) if total_files > 0 else 0
        
        report = f"""# Archetype Column Addition Summary Report
        
## Overview
- **Total Files**: {stats['total_files']:,}
- **Processed Files**: {stats['processed_files']:,}
- **Labeled Files**: {stats['labeled_files']:,} ({labeled_pct:.1f}%)
- **Unlabeled Files**: {stats['unlabeled_files']:,} ({unlabeled_pct:.1f}%)
- **Errors**: {stats['errors']:,}

## Archetype Distribution
"""
        
        # Add archetype distribution
        category_counts = {}
        cluster_counts = {}
        
        for token, labels in self.token_labels.items():
            category = labels['category']
            cluster = labels['cluster']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            cluster_key = f"{category}_cluster_{cluster}"
            cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        
        report += "\n### Categories\n"
        for category, count in sorted(category_counts.items()):
            pct = (count / len(self.token_labels) * 100) if len(self.token_labels) > 0 else 0
            report += f"- **{category}**: {count:,} tokens ({pct:.1f}%)\n"
        
        report += "\n### Clusters\n"
        for cluster, count in sorted(cluster_counts.items()):
            pct = (count / len(self.token_labels) * 100) if len(self.token_labels) > 0 else 0
            report += f"- **{cluster}**: {count:,} tokens ({pct:.1f}%)\n"
        
        report += f"""
        
## Data Quality
- **Coverage**: {labeled_pct:.1f}% of tokens have archetype labels
- **Success Rate**: {(stats['processed_files'] / stats['total_files'] * 100):.1f}% of files processed successfully
- **Error Rate**: {(stats['errors'] / stats['total_files'] * 100):.1f}% of files had errors

## Usage
The updated token files now include three new columns:
- `category`: The temporal category (standard, marathon, etc.)
- `cluster`: The cluster ID within the category (0, 1, 2, etc.)
- `archetype`: The full archetype name (e.g., "standard_cluster_0")

Files without archetype labels have `null` values in these columns.
"""
        
        # Save report
        report_path = output_dir / "archetype_addition_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Summary report saved to: {report_path}")
    
    def run_processing(self, archetype_results_path: Path, data_dir: Path, output_dir: Path) -> None:
        """Run the complete processing pipeline."""
        print(f"🚀 Starting Archetype Column Addition")
        
        # Load archetype results
        self.load_archetype_results(archetype_results_path)
        
        # Process token files
        stats = self.process_token_files(data_dir, output_dir)
        
        # Create summary report
        self.create_summary_report(stats, output_dir)
        
        # Print final summary
        print(f"\n🎉 Processing complete!")
        print(f"📊 SUMMARY")
        print(f"=" * 60)
        print(f"Total files: {stats['total_files']:,}")
        print(f"Processed files: {stats['processed_files']:,}")
        print(f"Labeled files: {stats['labeled_files']:,}")
        print(f"Unlabeled files: {stats['unlabeled_files']:,}")
        print(f"Errors: {stats['errors']:,}")
        print(f"📁 Updated files saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add Archetype Columns to Token Data")
    parser.add_argument("--archetype-results", type=Path,
                       help="Path to archetype characterization results JSON")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../data/processed"),
                       help="Path to processed token data directory")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("../../data/with_archetypes"),
                       help="Output directory for updated token files")
    
    args = parser.parse_args()
    
    # Find latest results if not specified
    if not args.archetype_results:
        # Try different possible paths for the results directory
        possible_paths = [
            Path("../results/phase1_day9_10_archetypes"),
            Path("../../time_series/results/phase1_day9_10_archetypes"),
            Path("./results/phase1_day9_10_archetypes"),
            Path(__file__).parent.parent / "results" / "phase1_day9_10_archetypes"
        ]
        
        results_dir = None
        for path in possible_paths:
            if path.exists():
                results_dir = path
                print(f"📁 Found results directory: {path}")
                break
        
        if results_dir and results_dir.exists():
            json_files = list(results_dir.glob("archetype_characterization_*.json"))
            if json_files:
                args.archetype_results = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest results: {args.archetype_results}")
            else:
                print("❌ No archetype results found. Run the Phase 1 pipeline first.")
                return
        else:
            print("❌ Results directory not found. Tried paths:")
            for path in possible_paths:
                print(f"   {path} - {'exists' if path.exists() else 'not found'}")
            print("Run the Phase 1 pipeline first.")
            return
    
    # Validate input directory
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Initialize processor
    processor = ArchetypeColumnAdder()
    
    try:
        # Run processing
        processor.run_processing(
            args.archetype_results,
            args.data_dir,
            args.output_dir
        )
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()