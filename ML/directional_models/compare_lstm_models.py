"""
Compare Basic LSTM vs Advanced Hybrid LSTM Models
This script loads both models and provides detailed comparison of their performance.
"""

import torch
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path
import json
from typing import Dict, List, Tuple
import pandas as pd


def load_model_results(model_dir: Path) -> Dict:
    """Load saved model metrics and configuration"""
    
    metrics_path = model_dir / 'metrics.json'
    model_path = model_dir / '*.pth'
    
    # Load metrics
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None
    
    # Check if model exists
    model_files = list(model_dir.glob('*.pth'))
    model_exists = len(model_files) > 0
    
    # Load training history if available
    if model_exists and model_files[0].exists():
        checkpoint = torch.load(model_files[0], map_location='cpu')
        training_history = checkpoint.get('training_history', None)
        config = checkpoint.get('config', None)
    else:
        training_history = None
        config = None
    
    return {
        'metrics': metrics,
        'training_history': training_history,
        'config': config,
        'model_exists': model_exists,
        'model_path': model_files[0] if model_exists else None
    }


def create_performance_comparison(basic_metrics: Dict, advanced_metrics: Dict) -> go.Figure:
    """Create detailed performance comparison between models"""
    
    # Extract horizons and metric names
    horizons = list(basic_metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Accuracy Comparison',
            'Precision Comparison',
            'Recall Comparison',
            'F1 Score Comparison',
            'ROC AUC Comparison',
            'Overall Performance Summary'
        ],
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Color scheme
    basic_color = '#3498db'  # Blue
    advanced_color = '#e74c3c'  # Red
    
    # Plot individual metrics
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    
    for idx, (metric, pos) in enumerate(zip(metric_names, positions)):
        row, col = pos
        
        # Basic model values
        basic_values = [basic_metrics.get(h, {}).get(metric, 0) for h in horizons]
        # Advanced model values
        advanced_values = [advanced_metrics.get(h, {}).get(metric, 0) for h in horizons]
        
        # Calculate improvements
        improvements = [(adv - basic) / basic * 100 if basic > 0 else 0 
                       for basic, adv in zip(basic_values, advanced_values)]
        
        # Add bars for basic model
        fig.add_trace(
            go.Bar(
                name='Basic LSTM' if idx == 0 else None,
                x=horizons,
                y=basic_values,
                text=[f'{v:.2%}' for v in basic_values],
                textposition='auto',
                marker_color=basic_color,
                showlegend=idx == 0,
                legendgroup='basic'
            ),
            row=row, col=col
        )
        
        # Add bars for advanced model
        fig.add_trace(
            go.Bar(
                name='Advanced Hybrid LSTM' if idx == 0 else None,
                x=horizons,
                y=advanced_values,
                text=[f'{v:.2%}' for v in advanced_values],
                textposition='auto',
                marker_color=advanced_color,
                showlegend=idx == 0,
                legendgroup='advanced'
            ),
            row=row, col=col
        )
        
        # Add improvement annotations
        for i, (h, imp) in enumerate(zip(horizons, improvements)):
            if abs(imp) > 1:  # Only show if improvement > 1%
                fig.add_annotation(
                    x=i,
                    y=max(basic_values[i], advanced_values[i]) + 0.05,
                    text=f'{imp:+.1f}%',
                    showarrow=False,
                    font=dict(size=10, color='green' if imp > 0 else 'red'),
                    xref=f'x{idx+1}' if idx > 0 else 'x',
                    yref=f'y{idx+1}' if idx > 0 else 'y',
                    row=row, col=col
                )
    
    # Summary plot - radar chart
    avg_basic = {metric: np.mean([basic_metrics.get(h, {}).get(metric, 0) for h in horizons]) 
                 for metric in metric_names}
    avg_advanced = {metric: np.mean([advanced_metrics.get(h, {}).get(metric, 0) for h in horizons]) 
                    for metric in metric_names}
    
    # Create radar chart data
    categories = [m.replace('_', ' ').title() for m in metric_names]
    
    fig.add_trace(
        go.Scatter(
            x=categories + [categories[0]],  # Close the polygon
            y=[avg_basic[m] for m in metric_names] + [avg_basic[metric_names[0]]],
            mode='lines+markers',
            name='Basic LSTM Avg',
            line=dict(color=basic_color, width=2),
            fill='toself',
            fillcolor=basic_color,
            opacity=0.6,
            showlegend=False
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=categories + [categories[0]],
            y=[avg_advanced[m] for m in metric_names] + [avg_advanced[metric_names[0]]],
            mode='lines+markers',
            name='Advanced LSTM Avg',
            line=dict(color=advanced_color, width=2),
            fill='toself',
            fillcolor=advanced_color,
            opacity=0.4,
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='LSTM Model Performance Comparison: Basic vs Advanced Hybrid',
            font=dict(size=20)
        ),
        barmode='group',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes to percentage scale
    for i in range(1, 6):
        fig.update_yaxes(range=[0, 1], tickformat='.0%', row=(i-1)//3 + 1, col=(i-1)%3 + 1)
    
    return fig


def create_training_comparison(basic_history: Dict, advanced_history: Dict) -> go.Figure:
    """Compare training histories of both models"""
    
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Loss Comparison', 'Validation Loss Comparison'],
        horizontal_spacing=0.15
    )
    
    # Training losses
    if basic_history and 'train_losses' in basic_history:
        epochs_basic = list(range(1, len(basic_history['train_losses']) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs_basic,
                y=basic_history['train_losses'],
                mode='lines',
                name='Basic LSTM',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )
    
    if advanced_history and 'train_losses' in advanced_history:
        epochs_advanced = list(range(1, len(advanced_history['train_losses']) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs_advanced,
                y=advanced_history['train_losses'],
                mode='lines',
                name='Advanced LSTM',
                line=dict(color='#e74c3c', width=2)
            ),
            row=1, col=1
        )
    
    # Validation losses
    if basic_history and 'val_losses' in basic_history:
        fig.add_trace(
            go.Scatter(
                x=epochs_basic,
                y=basic_history['val_losses'],
                mode='lines',
                name='Basic LSTM Val',
                line=dict(color='#3498db', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    if advanced_history and 'val_losses' in advanced_history:
        fig.add_trace(
            go.Scatter(
                x=epochs_advanced,
                y=advanced_history['val_losses'],
                mode='lines',
                name='Advanced LSTM Val',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Training History Comparison',
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    
    return fig


def create_summary_table(basic_results: Dict, advanced_results: Dict) -> pd.DataFrame:
    """Create a summary comparison table"""
    
    summary_data = []
    
    # Model complexity
    if basic_results['config']:
        basic_config = basic_results['config']
        summary_data.append({
            'Metric': 'Hidden Size',
            'Basic LSTM': basic_config.get('hidden_size', 'N/A'),
            'Advanced LSTM': advanced_results['config'].get('hidden_size', 'N/A') if advanced_results['config'] else 'N/A'
        })
        
        summary_data.append({
            'Metric': 'Model Type',
            'Basic LSTM': 'Fixed 60-min window',
            'Advanced LSTM': 'Multi-scale + Expanding + Attention'
        })
    
    # Average performance across all horizons
    if basic_results['metrics'] and advanced_results['metrics']:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            basic_avg = np.mean([
                basic_results['metrics'].get(h, {}).get(metric, 0) 
                for h in basic_results['metrics'].keys()
            ])
            advanced_avg = np.mean([
                advanced_results['metrics'].get(h, {}).get(metric, 0) 
                for h in advanced_results['metrics'].keys()
            ])
            
            improvement = ((advanced_avg - basic_avg) / basic_avg * 100) if basic_avg > 0 else 0
            
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Basic LSTM': f'{basic_avg:.2%}',
                'Advanced LSTM': f'{advanced_avg:.2%}',
                'Improvement': f'{improvement:+.1f}%'
            })
    
    # Training efficiency
    if basic_results['training_history'] and advanced_results['training_history']:
        summary_data.append({
            'Metric': 'Best Epoch',
            'Basic LSTM': basic_results['training_history'].get('best_epoch', 'N/A'),
            'Advanced LSTM': advanced_results['training_history'].get('best_epoch', 'N/A')
        })
        
        summary_data.append({
            'Metric': 'Best Val Loss',
            'Basic LSTM': f"{basic_results['training_history'].get('best_val_loss', 0):.4f}",
            'Advanced LSTM': f"{advanced_results['training_history'].get('best_val_loss', 0):.4f}"
        })
    
    return pd.DataFrame(summary_data)


def main():
    """Main comparison pipeline"""
    
    print("="*60)
    print("📊 LSTM Model Comparison: Basic vs Advanced Hybrid")
    print("="*60)
    
    # Define model directories
    basic_dir = Path("ML/results/unified_lstm")
    advanced_dir = Path("ML/results/advanced_hybrid_lstm")
    comparison_dir = Path("ML/results/model_comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\nLoading model results...")
    basic_results = load_model_results(basic_dir)
    advanced_results = load_model_results(advanced_dir)
    
    # Check if both models exist
    if not basic_results['model_exists']:
        print(f"❌ Basic LSTM model not found in {basic_dir}")
        print("Please train the basic model first: python ML/directional_models/train_unified_lstm_model.py")
        return
    
    if not advanced_results['model_exists']:
        print(f"❌ Advanced LSTM model not found in {advanced_dir}")
        print("Please train the advanced model first: python ML/directional_models/train_advanced_hybrid_lstm.py")
        return
    
    print("✅ Both models found!")
    
    # Create comparison visualizations
    if basic_results['metrics'] and advanced_results['metrics']:
        print("\nCreating performance comparison...")
        perf_fig = create_performance_comparison(
            basic_results['metrics'],
            advanced_results['metrics']
        )
        perf_fig.write_html(comparison_dir / 'performance_comparison.html')
        print(f"✅ Saved to: {comparison_dir / 'performance_comparison.html'}")
    
    # Create training comparison
    if basic_results['training_history'] and advanced_results['training_history']:
        print("\nCreating training history comparison...")
        train_fig = create_training_comparison(
            basic_results['training_history'],
            advanced_results['training_history']
        )
        train_fig.write_html(comparison_dir / 'training_comparison.html')
        print(f"✅ Saved to: {comparison_dir / 'training_comparison.html'}")
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(basic_results, advanced_results)
    
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(comparison_dir / 'model_comparison_summary.csv', index=False)
    
    # Create detailed analysis report
    with open(comparison_dir / 'comparison_report.txt', 'w') as f:
        f.write("LSTM MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. MODEL ARCHITECTURES\n")
        f.write("-"*30 + "\n")
        f.write("Basic LSTM:\n")
        f.write("  - Fixed 60-minute lookback window\n")
        f.write("  - Standard LSTM layers\n")
        f.write("  - Single time scale\n\n")
        
        f.write("Advanced Hybrid LSTM:\n")
        f.write("  - Multi-scale fixed windows (15m, 1h, 4h)\n")
        f.write("  - Expanding window (60m-12h)\n")
        f.write("  - Self-attention mechanism\n")
        f.write("  - Cross-attention between scales\n")
        f.write("  - Hierarchical feature fusion\n\n")
        
        f.write("2. PERFORMANCE SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("3. KEY INSIGHTS\n")
        f.write("-"*30 + "\n")
        
        # Calculate average improvements
        if basic_results['metrics'] and advanced_results['metrics']:
            improvements = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                basic_vals = [basic_results['metrics'].get(h, {}).get(metric, 0) 
                             for h in basic_results['metrics'].keys()]
                advanced_vals = [advanced_results['metrics'].get(h, {}).get(metric, 0) 
                                for h in advanced_results['metrics'].keys()]
                
                avg_basic = np.mean(basic_vals)
                avg_advanced = np.mean(advanced_vals)
                improvements[metric] = ((avg_advanced - avg_basic) / avg_basic * 100) if avg_basic > 0 else 0
            
            best_improvement = max(improvements.items(), key=lambda x: x[1])
            worst_improvement = min(improvements.items(), key=lambda x: x[1])
            
            f.write(f"- Best improvement: {best_improvement[0]} ({best_improvement[1]:+.1f}%)\n")
            f.write(f"- Worst improvement: {worst_improvement[0]} ({worst_improvement[1]:+.1f}%)\n")
            f.write(f"- Average improvement across all metrics: {np.mean(list(improvements.values())):+.1f}%\n\n")
        
        f.write("4. RECOMMENDATIONS\n")
        f.write("-"*30 + "\n")
        f.write("- Use Advanced LSTM for better accuracy but higher computational cost\n")
        f.write("- Basic LSTM suitable for real-time applications with resource constraints\n")
        f.write("- Consider ensemble of both models for best results\n")
    
    print(f"\n✅ Comparison report saved to: {comparison_dir / 'comparison_report.txt'}")
    print(f"\n🎉 Model comparison complete! Check {comparison_dir} for all results.")


if __name__ == "__main__":
    main() 