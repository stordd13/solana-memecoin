"""
Quick Analysis: Positive vs Negative Returns Distribution
Analyzes how many tokens have positive/negative returns over different time horizons
"""

import os
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_returns_distribution():
    """Analyze positive vs negative returns across all tokens"""
    
    # Configuration
    # ---------------------------------------------------------------------------
    # Base directory selection
    # You can override with environment variable:
    #     BASE_DIR=data/features python analyze_returns_distribution.py
    # ---------------------------------------------------------------------------

    base_dir = Path(os.environ.get("BASE_DIR", "data/cleaned"))
    categories = [
        "normal_behavior_tokens",
        "tokens_with_gaps", 
        "tokens_with_extremes",
        "dead_tokens"
    ]
    horizons = [15, 30, 60, 120, 240, 360, 720]  # 15m to 12h
    
    print("="*60)
    print("MEMECOIN RETURNS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Collect all token files
    all_files = []
    for category in categories:
        cat_dir = base_dir / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*.parquet"))
            all_files.extend(files)
            print(f"Found {len(files)} files in {category}")
    
    print(f"\nTotal files to analyze: {len(all_files)}")
    
    # Results storage
    results = {h: {'positive': 0, 'negative': 0, 'total_valid': 0} for h in horizons}
    category_results = {}
    
    # Analyze each token
    for file_path in tqdm(all_files, desc="Analyzing tokens"):
        try:
            df = pl.read_parquet(file_path)
            
            # Require a price column but do NOT impose a global length threshold.
            # Length is now checked per-horizon further below.
            if 'price' not in df.columns:
                continue
            
            prices = df['price'].to_numpy()
            
            # Skip tokens with NaN values
            if np.isnan(prices).any():
                continue
            
            category = file_path.parent.name
            if category not in category_results:
                category_results[category] = {h: {'positive': 0, 'negative': 0, 'total_valid': 0} for h in horizons}
            
            # Calculate returns for each horizon
            for horizon in horizons:
                if len(prices) < horizon + 1:
                    continue
                
                # Calculate return from start to horizon minutes later
                initial_price = prices[0]
                future_price = prices[horizon]
                
                if initial_price > 0:  # Avoid division by zero
                    return_pct = (future_price - initial_price) / initial_price * 100
                    
                    # Count positive vs negative
                    if return_pct > 0:
                        results[horizon]['positive'] += 1
                        category_results[category][horizon]['positive'] += 1
                    else:
                        results[horizon]['negative'] += 1
                        category_results[category][horizon]['negative'] += 1
                    
                    results[horizon]['total_valid'] += 1
                    category_results[category][horizon]['total_valid'] += 1
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Print overall results
    print(f"\n{'='*60}")
    print("OVERALL RESULTS BY TIME HORIZON")
    print(f"{'='*60}")
    
    for horizon in horizons:
        pos = results[horizon]['positive']
        neg = results[horizon]['negative']
        total = results[horizon]['total_valid']
        
        if total > 0:
            pos_pct = pos / total * 100
            neg_pct = neg / total * 100
            
            if horizon >= 60:
                time_str = f"{horizon//60}h"
            else:
                time_str = f"{horizon}m"
            
            print(f"{time_str:>4}: {pos:>6,} positive ({pos_pct:5.1f}%) | {neg:>6,} negative ({neg_pct:5.1f}%) | Total: {total:>6,}")
    
    # Print category breakdown
    print(f"\n{'='*60}")
    print("BREAKDOWN BY CATEGORY (15min horizon)")
    print(f"{'='*60}")
    
    for category, cat_data in category_results.items():
        pos = cat_data[15]['positive']
        neg = cat_data[15]['negative']
        total = cat_data[15]['total_valid']
        
        if total > 0:
            pos_pct = pos / total * 100
            print(f"{category:<25}: {pos:>5,} positive ({pos_pct:5.1f}%) | {neg:>5,} negative | Total: {total:>5,}")
    
    # Create visualization
    fig = create_visualization(results, category_results)
    
    # Save plot
    output_path = Path("ML/directional_models/returns_distribution_analysis.html")
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    
    # Calculate overall trend
    pos_15m = results[15]['positive']
    total_15m = results[15]['total_valid']
    pos_12h = results[720]['positive']
    total_12h = results[720]['total_valid']
    
    if total_15m > 0 and total_12h > 0:
        pos_pct_15m = pos_15m / total_15m * 100
        pos_pct_12h = pos_12h / total_12h * 100
        
        print(f"• Short-term (15m): {pos_pct_15m:.1f}% of tokens have positive returns")
        print(f"• Long-term (12h): {pos_pct_12h:.1f}% of tokens have positive returns")
        print(f"• Trend: {'Improving' if pos_pct_12h > pos_pct_15m else 'Declining'} over time")
        
        if pos_pct_15m < 50:
            print(f"• Most tokens decline even in the first 15 minutes - typical memecoin behavior")
        
        # Class imbalance note
        print(f"• This explains why ML models achieve high accuracy by predicting DOWN")
        print(f"• The severe class imbalance ({100-pos_pct_15m:.1f}% DOWN vs {pos_pct_15m:.1f}% UP) is realistic")

def create_visualization(results, category_results):
    """Create plotly visualization of results"""
    
    # Prepare data for main plot
    horizons = list(results.keys())
    horizon_labels = []
    for h in horizons:
        if h >= 60:
            horizon_labels.append(f"{h//60}h")
        else:
            horizon_labels.append(f"{h}m")
    
    positive_counts = [results[h]['positive'] for h in horizons]
    negative_counts = [results[h]['negative'] for h in horizons]
    positive_pcts = [results[h]['positive'] / max(results[h]['total_valid'], 1) * 100 for h in horizons]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Positive vs Negative Returns by Time Horizon',
            'Percentage of Positive Returns Over Time',
            'Category Breakdown (15min horizon)',
            'Token Counts by Horizon'
        ),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'type': 'bar'}, {'secondary_y': False}]]
    )
    
    # Plot 1: Stacked bar chart
    fig.add_trace(
        go.Bar(name='Positive Returns', x=horizon_labels, y=positive_counts, 
               marker_color='green', text=positive_counts, textposition='inside'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Negative Returns', x=horizon_labels, y=negative_counts,
               marker_color='red', text=negative_counts, textposition='inside'),
        row=1, col=1
    )
    
    # Plot 2: Percentage line chart
    fig.add_trace(
        go.Scatter(x=horizon_labels, y=positive_pcts, mode='lines+markers',
                  name='% Positive', line=dict(color='blue', width=3),
                  text=[f'{p:.1f}%' for p in positive_pcts], textposition='top center'),
        row=1, col=2
    )
    
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="50% (Random)", row=1, col=2)
    
    # Plot 3: Category breakdown for 15min
    categories = list(category_results.keys())
    cat_positive = [category_results[cat][15]['positive'] for cat in categories]
    cat_negative = [category_results[cat][15]['negative'] for cat in categories]
    
    fig.add_trace(
        go.Bar(name='Positive (15m)', x=categories, y=cat_positive, marker_color='lightgreen'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Negative (15m)', x=categories, y=cat_negative, marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Plot 4: Total valid tokens by horizon
    total_counts = [results[h]['total_valid'] for h in horizons]
    fig.add_trace(
        go.Bar(x=horizon_labels, y=total_counts, name='Total Tokens',
               marker_color='lightblue', text=total_counts, textposition='outside'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Memecoin Returns Distribution Analysis",
        showlegend=True,
        barmode='stack'
    )
    
    # Update subplot titles and axes
    fig.update_xaxes(title_text="Time Horizon", row=1, col=1)
    fig.update_xaxes(title_text="Time Horizon", row=1, col=2)
    fig.update_xaxes(title_text="Category", row=2, col=1)
    fig.update_xaxes(title_text="Time Horizon", row=2, col=2)
    
    fig.update_yaxes(title_text="Number of Tokens", row=1, col=1)
    fig.update_yaxes(title_text="Percentage Positive", row=1, col=2)
    fig.update_yaxes(title_text="Number of Tokens", row=2, col=1)
    fig.update_yaxes(title_text="Total Valid Tokens", row=2, col=2)
    
    return fig

if __name__ == "__main__":
    analyze_returns_distribution() 