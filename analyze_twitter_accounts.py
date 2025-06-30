"""
Twitter Account Analysis for Memecoin Dataset
Analyzes Twitter account usage patterns across tokens to identify shared accounts
"""

import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import json

def find_parquet_files(directories: List[Path]) -> List[Path]:
    """Find all parquet files in the given directories"""
    all_files = []
    
    for directory in directories:
        if directory.exists():
            files = list(directory.glob("*.parquet"))
            print(f"📁 {directory}: {len(files)} parquet files")
            all_files.extend(files)
        else:
            print(f"⚠️  Directory not found: {directory}")
    
    return all_files

def extract_twitter_data(file_path: Path, sample_rows: Optional[int] = None) -> Dict:
    """Extract Twitter account information from a parquet file"""
    try:
        # Load the parquet file
        if sample_rows:
            df = pl.read_parquet(file_path).head(sample_rows)
        else:
            df = pl.read_parquet(file_path)
        
        if df.is_empty():
            return None
        
        # Look for Twitter-related columns
        twitter_columns = [col for col in df.columns if 'twitter' in col.lower() or 'uri' in col.lower()]
        
        if not twitter_columns:
            return None
        
        # Extract unique Twitter accounts from all Twitter columns
        twitter_accounts = set()
        for col in twitter_columns:
            accounts = df[col].drop_nulls().unique().to_list()
            # Filter out empty strings and common non-Twitter values
            filtered_accounts = [
                acc for acc in accounts 
                if acc and str(acc).strip() and str(acc).strip() != 'null' and len(str(acc).strip()) > 1
            ]
            twitter_accounts.update(filtered_accounts)
        
        return {
            'token_file': file_path.stem,
            'twitter_columns': twitter_columns,
            'total_rows': df.height,
            'twitter_accounts': list(twitter_accounts),
            'num_twitter_accounts': len(twitter_accounts),
            'sample_data': df[twitter_columns].head(5).to_dicts() if twitter_columns else []
        }
        
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return None

def analyze_twitter_patterns(twitter_data: List[Dict]) -> Dict:
    """Analyze patterns in Twitter account usage"""
    
    # Count all Twitter accounts across all tokens
    all_accounts = []
    token_to_accounts = {}
    accounts_to_tokens = {}
    
    for data in twitter_data:
        if data and data['twitter_accounts']:
            token_name = data['token_file']
            accounts = data['twitter_accounts']
            
            token_to_accounts[token_name] = accounts
            all_accounts.extend(accounts)
            
            # Track which tokens use each account
            for account in accounts:
                if account not in accounts_to_tokens:
                    accounts_to_tokens[account] = []
                accounts_to_tokens[account].append(token_name)
    
    # Count frequency of each account
    account_counts = Counter(all_accounts)
    
    # Find shared accounts (used by multiple tokens)
    shared_accounts = {acc: tokens for acc, tokens in accounts_to_tokens.items() if len(tokens) > 1}
    
    # Statistics
    stats = {
        'total_tokens_with_twitter': len([d for d in twitter_data if d and d['twitter_accounts']]),
        'total_unique_accounts': len(set(all_accounts)),
        'total_account_usages': len(all_accounts),
        'shared_accounts_count': len(shared_accounts),
        'most_used_accounts': account_counts.most_common(10),
        'shared_accounts': shared_accounts,
        'tokens_per_account_distribution': list(Counter([len(tokens) for tokens in accounts_to_tokens.values()]).items()),
        'accounts_per_token_distribution': list(Counter([len(accounts) for accounts in token_to_accounts.values()]).items())
    }
    
    return stats

def create_visualizations(stats: Dict) -> List[go.Figure]:
    """Create visualizations for Twitter account analysis"""
    figures = []
    
    # 1. Distribution of tokens per Twitter account
    if stats['tokens_per_account_distribution']:
        tokens_per_acc, counts = zip(*stats['tokens_per_account_distribution'])
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=[f"{n} token{'s' if n > 1 else ''}" for n in tokens_per_acc],
            y=counts,
            text=counts,
            textposition='auto',
            marker_color='lightblue'
        ))
        fig1.update_layout(
            title="Distribution: How Many Tokens Use Each Twitter Account",
            xaxis_title="Number of Tokens Using Same Account",
            yaxis_title="Number of Twitter Accounts",
            showlegend=False
        )
        figures.append(fig1)
    
    # 2. Distribution of Twitter accounts per token  
    if stats['accounts_per_token_distribution']:
        accounts_per_token, token_counts = zip(*stats['accounts_per_token_distribution'])
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[f"{n} account{'s' if n > 1 else ''}" for n in accounts_per_token],
            y=token_counts,
            text=token_counts,
            textposition='auto',
            marker_color='lightgreen'
        ))
        fig2.update_layout(
            title="Distribution: How Many Twitter Accounts Each Token Has",
            xaxis_title="Number of Twitter Accounts per Token",
            yaxis_title="Number of Tokens",
            showlegend=False
        )
        figures.append(fig2)
    
    # 3. Most used Twitter accounts
    if stats['most_used_accounts']:
        accounts, usage_counts = zip(*stats['most_used_accounts'][:15])  # Top 15
        
        # Truncate long account names for display
        display_accounts = [acc[:50] + '...' if len(str(acc)) > 50 else str(acc) for acc in accounts]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=display_accounts,
            x=usage_counts,
            orientation='h',
            text=usage_counts,
            textposition='auto',
            marker_color='salmon'
        ))
        fig3.update_layout(
            title="Most Frequently Used Twitter Accounts (Top 15)",
            xaxis_title="Number of Tokens Using This Account",
            yaxis_title="Twitter Account",
            height=600
        )
        figures.append(fig3)
    
    # 4. Summary statistics pie chart
    shared_count = stats['shared_accounts_count']
    unique_count = stats['total_unique_accounts'] - shared_count
    
    if shared_count > 0:
        fig4 = go.Figure()
        fig4.add_trace(go.Pie(
            labels=['Unique to One Token', 'Shared Between Tokens'],
            values=[unique_count, shared_count],
            hole=0.3,
            marker_colors=['lightcoral', 'gold']
        ))
        fig4.update_layout(
            title="Twitter Accounts: Unique vs Shared",
            annotations=[dict(text='Twitter<br>Accounts', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        figures.append(fig4)
    
    return figures

def print_summary(stats: Dict):
    """Print a summary of the Twitter account analysis"""
    print("\n" + "="*60)
    print("🐦 TWITTER ACCOUNT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"📊 OVERVIEW:")
    print(f"   Total tokens with Twitter data: {stats['total_tokens_with_twitter']:,}")
    print(f"   Total unique Twitter accounts: {stats['total_unique_accounts']:,}")
    print(f"   Total account usages: {stats['total_account_usages']:,}")
    print(f"   Average accounts per token: {stats['total_account_usages'] / max(stats['total_tokens_with_twitter'], 1):.1f}")
    
    print(f"\n🔄 SHARING PATTERNS:")
    print(f"   Accounts shared between tokens: {stats['shared_accounts_count']:,}")
    print(f"   Accounts unique to one token: {stats['total_unique_accounts'] - stats['shared_accounts_count']:,}")
    print(f"   Sharing rate: {stats['shared_accounts_count'] / max(stats['total_unique_accounts'], 1) * 100:.1f}%")
    
    if stats['most_used_accounts']:
        print(f"\n🏆 TOP SHARED ACCOUNTS:")
        for i, (account, count) in enumerate(stats['most_used_accounts'][:5], 1):
            account_display = str(account)[:60] + '...' if len(str(account)) > 60 else str(account)
            print(f"   {i}. {account_display} ({count} tokens)")
    
    # Show some interesting shared accounts
    if stats['shared_accounts']:
        print(f"\n🔍 EXAMPLE SHARED ACCOUNTS:")
        shared_items = list(stats['shared_accounts'].items())[:3]
        for account, tokens in shared_items:
            account_display = str(account)[:50] + '...' if len(str(account)) > 50 else str(account)
            token_list = ', '.join(tokens[:3]) + ('...' if len(tokens) > 3 else '')
            print(f"   '{account_display}' → {len(tokens)} tokens: {token_list}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Twitter account usage in memecoin dataset')
    parser.add_argument('--sample-tokens', type=int, default=None, 
                       help='Number of tokens to randomly sample (default: all)')
    parser.add_argument('--sample-rows', type=int, default=None,
                       help='Number of rows to sample from each file (default: all)')
    parser.add_argument('--output-dir', type=str, default='twitter_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Define data directories
    data_dirs = [
        Path("data/raw/dataset"),
        Path("data/raw/dataset-fresh")
    ]
    
    print("🐦 Twitter Account Analysis for Memecoin Dataset")
    print("="*50)
    
    # Find all parquet files
    all_files = find_parquet_files(data_dirs)
    
    if not all_files:
        print("❌ No parquet files found in the specified directories!")
        return
    
    print(f"\n📊 Found {len(all_files):,} total parquet files")
    
    # Sample tokens if requested
    if args.sample_tokens and len(all_files) > args.sample_tokens:
        import random
        random.seed(42)
        all_files = random.sample(all_files, args.sample_tokens)
        print(f"🎲 Randomly sampled {len(all_files):,} tokens for analysis")
    
    # Extract Twitter data from each file
    print(f"\n🔍 Extracting Twitter data from files...")
    twitter_data = []
    
    for file_path in tqdm(all_files, desc="Processing files"):
        data = extract_twitter_data(file_path, args.sample_rows)
        if data:
            twitter_data.append(data)
    
    if not twitter_data:
        print("❌ No Twitter data found in any files!")
        return
    
    print(f"✅ Successfully extracted Twitter data from {len(twitter_data):,} files")
    
    # Analyze patterns
    print("\n📈 Analyzing Twitter account patterns...")
    stats = analyze_twitter_patterns(twitter_data)
    
    # Print summary
    print_summary(stats)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    figures = create_visualizations(stats)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save visualizations
    for i, fig in enumerate(figures):
        filename = output_dir / f"twitter_analysis_{i+1}.html"
        fig.write_html(filename)
        print(f"   📈 Saved visualization: {filename}")
    
    # Save detailed data
    detailed_output = {
        'summary_stats': stats,
        'sample_files_analyzed': len(twitter_data),
        'total_files_found': len(all_files),
        'analysis_parameters': {
            'sample_tokens': args.sample_tokens,
            'sample_rows': args.sample_rows
        }
    }
    
    # Save shared accounts details (top 50 to avoid huge files)
    if stats['shared_accounts']:
        shared_accounts_top = dict(list(stats['shared_accounts'].items())[:50])
        detailed_output['top_shared_accounts_detail'] = shared_accounts_top
    
    json_path = output_dir / "twitter_analysis_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_output, f, indent=2, default=str)
    print(f"   💾 Saved detailed analysis: {json_path}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"🔍 Key Finding: {stats['shared_accounts_count']:,} Twitter accounts are shared between multiple tokens")
    
    # Quick recommendation
    if stats['shared_accounts_count'] > 0:
        sharing_rate = stats['shared_accounts_count'] / stats['total_unique_accounts'] * 100
        if sharing_rate > 20:
            print(f"⚠️  High sharing rate ({sharing_rate:.1f}%) - consider analyzing account creation patterns")
        elif sharing_rate > 5:
            print(f"📝 Moderate sharing rate ({sharing_rate:.1f}%) - normal for related projects")
        else:
            print(f"✅ Low sharing rate ({sharing_rate:.1f}%) - mostly unique accounts")

if __name__ == "__main__":
    main() 