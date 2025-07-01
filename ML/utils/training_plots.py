"""
Training visualization utilities for ML models.

Functions for creating training curves, performance metrics plots,
and enhanced financial analysis visualizations.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        train_metrics: Optional[Dict[str, List[float]]] = None,
                        val_metrics: Optional[Dict[str, List[float]]] = None,
                        title: str = "Training Progress") -> go.Figure:
    """Create comprehensive training progress visualization"""
    
    # Determine number of subplots needed
    n_plots = 1
    if train_metrics and val_metrics:
        n_plots += len(train_metrics)
    
    # Create subplots
    if n_plots == 1:
        fig = go.Figure()
        
        # Plot losses
        fig.add_trace(go.Scatter(
            x=list(range(len(train_losses))),
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(val_losses))),
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified'
        )
    else:
        # Multiple subplots for losses and metrics
        subplot_titles = ["Loss"] + list(train_metrics.keys())
        fig = make_subplots(
            rows=len(subplot_titles), cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        # Add loss plot
        fig.add_trace(
            go.Scatter(x=list(range(len(train_losses))), y=train_losses,
                      mode='lines', name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(val_losses))), y=val_losses,
                      mode='lines', name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Add metric plots
        for i, (metric_name, train_values) in enumerate(train_metrics.items(), 2):
            val_values = val_metrics.get(metric_name, [])
            
            fig.add_trace(
                go.Scatter(x=list(range(len(train_values))), y=train_values,
                          mode='lines', name=f'Train {metric_name}', 
                          line=dict(color='blue', dash='dot')),
                row=i, col=1
            )
            
            if val_values:
                fig.add_trace(
                    go.Scatter(x=list(range(len(val_values))), y=val_values,
                              mode='lines', name=f'Val {metric_name}',
                              line=dict(color='red', dash='dot')),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * n_plots,
            hovermode='x unified',
            showlegend=True
        )
        
        # Update x-axis labels
        for i in range(1, n_plots + 1):
            fig.update_xaxes(title_text="Epoch", row=i, col=1)
    
    return fig


def plot_loss_comparison(model_losses: Dict[str, Dict[str, List[float]]], 
                        title: str = "Model Loss Comparison") -> go.Figure:
    """
    Compare losses across multiple models
    
    Args:
        model_losses: Dict of model_name -> {'train': [...], 'val': [...]}
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for idx, (model_name, losses) in enumerate(model_losses.items()):
        color = colors[idx % len(colors)]
        epochs = list(range(1, len(losses['train']) + 1))
        
        # Training loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses['train'],
            mode='lines',
            name=f'{model_name} (Train)',
            line=dict(color=color, width=2, dash='solid')
        ))
        
        # Validation loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses['val'],
            mode='lines',
            name=f'{model_name} (Val)',
            line=dict(color=color, width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_learning_summary(training_history: Dict) -> str:
    """
    Create a text summary of training results
    
    Args:
        training_history: Dictionary with training metrics and losses
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("="*50)
    summary.append("TRAINING SUMMARY")
    summary.append("="*50)
    
    # Training configuration
    if 'config' in training_history:
        config = training_history['config']
        summary.append(f"Epochs: {config.get('epochs', 'N/A')}")
        summary.append(f"Batch Size: {config.get('batch_size', 'N/A')}")
        summary.append(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
        summary.append("")
    
    # Best epoch info
    if 'val_losses' in training_history:
        val_losses = training_history['val_losses']
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        summary.append(f"Best Epoch: {best_epoch}")
        summary.append(f"Best Validation Loss: {best_loss:.4f}")
        summary.append("")
    
    # Final metrics
    if 'final_metrics' in training_history:
        summary.append("Final Test Metrics:")
        for horizon, metrics in training_history['final_metrics'].items():
            summary.append(f"\n{horizon}:")
            for metric, value in metrics.items():
                summary.append(f"  {metric}: {value:.4f}")
    
    return "\n".join(summary)


def plot_financial_metrics_enhanced(metrics: Dict, model_name: str = "Model") -> go.Figure:
    """Create enhanced financial metrics visualization with multiple perspectives"""
    
    # Extract horizon names and prepare data
    horizons = list(metrics.keys())
    
    # Prepare data for different visualizations
    accuracy_data = []
    precision_data = []
    recall_data = []
    f1_data = []
    auc_data = []
    
    # Financial metrics
    profit_data = []
    sharpe_data = []
    win_rate_data = []
    return_capture_data = []
    
    for horizon in horizons:
        m = metrics[horizon]
        accuracy_data.append(m.get('accuracy', 0))
        precision_data.append(m.get('precision', 0))
        recall_data.append(m.get('recall', 0))
        f1_data.append(m.get('f1_score', 0))
        auc_data.append(m.get('roc_auc', 0.5))
        
        # Financial metrics (with fallbacks)
        profit_data.append(m.get('strategy_return', m.get('profit_factor', 0)))
        sharpe_data.append(m.get('sharpe_ratio', m.get('strategy_sharpe', 0)))
        win_rate_data.append(m.get('win_rate', m.get('strategy_win_rate', 0)))
        return_capture_data.append(m.get('return_capture_ratio', m.get('return_correlation', 0)))
    
    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Classification Metrics",
            "Financial Performance", 
            "Risk-Adjusted Returns",
            "Trading Effectiveness"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Classification Metrics (Top Left)
    fig.add_trace(
        go.Scatter(x=horizons, y=accuracy_data, mode='lines+markers',
                  name='Accuracy', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=horizons, y=precision_data, mode='lines+markers',
                  name='Precision', line=dict(color='green', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=horizons, y=recall_data, mode='lines+markers',
                  name='Recall', line=dict(color='red', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=horizons, y=f1_data, mode='lines+markers',
                  name='F1 Score', line=dict(color='purple', width=2)),
        row=1, col=1
    )
    
    # 2. Financial Performance (Top Right) - Dual Y-axis
    fig.add_trace(
        go.Bar(x=horizons, y=profit_data, name='Profit/Return',
               marker_color='lightgreen', opacity=0.7),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=horizons, y=win_rate_data, mode='lines+markers',
                  name='Win Rate', line=dict(color='darkgreen', width=3),
                  yaxis='y2'),
        row=1, col=2, secondary_y=True
    )
    
    # 3. Risk-Adjusted Returns (Bottom Left)
    fig.add_trace(
        go.Scatter(x=horizons, y=sharpe_data, mode='lines+markers',
                  name='Sharpe Ratio', line=dict(color='orange', width=3),
                  fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=horizons, y=auc_data, mode='lines+markers',
                  name='ROC AUC', line=dict(color='navy', width=2)),
        row=2, col=1
    )
    
    # 4. Trading Effectiveness (Bottom Right)
    fig.add_trace(
        go.Bar(x=horizons, y=return_capture_data, name='Return Capture',
               marker_color='lightblue', opacity=0.7),
        row=2, col=2
    )
    
    # Add horizontal reference lines
    # Accuracy/Precision/Recall reference at 50%
    for col in [1]:
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="Random (50%)", row=1, col=col)
    
    # AUC reference at 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                 annotation_text="Random AUC", row=2, col=1)
    
    # Sharpe reference at 1.0 (good performance)
    fig.add_hline(y=1.0, line_dash="dash", line_color="green", opacity=0.5,
                 annotation_text="Good Sharpe", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{model_name} - Comprehensive Performance Analysis",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time Horizon", row=2, col=1)
    fig.update_xaxes(title_text="Time Horizon", row=2, col=2)
    
    fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Return/Profit", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2, secondary_y=True, range=[0, 1])
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Capture Ratio", row=2, col=2, range=[0, 1])
    
    return fig


def plot_risk_return_analysis(metrics: Dict, model_name: str = "Model") -> go.Figure:
    """Create risk-return scatter plot for different horizons"""
    
    horizons = list(metrics.keys())
    returns = []
    risks = []
    sharpe_ratios = []
    win_rates = []
    
    for horizon in horizons:
        m = metrics[horizon]
        returns.append(m.get('strategy_return', m.get('mean_return', 0)))
        risks.append(m.get('volatility', m.get('return_std', 0.1)))
        sharpe_ratios.append(m.get('sharpe_ratio', m.get('strategy_sharpe', 0)))
        win_rates.append(m.get('win_rate', m.get('strategy_win_rate', 0.5)))
    
    # Create bubble chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers+text',
        marker=dict(
            size=[w*50 for w in win_rates],  # Size based on win rate
            color=sharpe_ratios,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            sizemode='diameter',
            sizeref=0.1,
            line=dict(width=2, color='black')
        ),
        text=horizons,
        textposition='middle center',
        name='Horizons',
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Risk: %{x:.3f}<br>'
            'Return: %{y:.3f}<br>'
            'Win Rate: %{marker.size}<br>'
            'Sharpe: %{marker.color:.2f}'
            '<extra></extra>'
        )
    ))
    
    # Add efficient frontier reference
    efficient_x = np.linspace(min(risks), max(risks), 50)
    efficient_y = [max(0, x * 2) for x in efficient_x]  # Simple linear relationship
    
    fig.add_trace(go.Scatter(
        x=efficient_x,
        y=efficient_y,
        mode='lines',
        name='Reference Line (Risk=2×Return)',
        line=dict(dash='dash', color='gray', width=1),
        opacity=0.5
    ))
    
    fig.update_layout(
        title=f"{model_name} - Risk-Return Profile by Time Horizon",
        xaxis_title="Risk (Volatility)",
        yaxis_title="Expected Return",
        height=600,
        hovermode='closest'
    )
    
    return fig 