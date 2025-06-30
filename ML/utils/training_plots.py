"""
Utility functions for plotting training curves and metrics
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        train_metrics: Optional[Dict[str, List[float]]] = None,
                        val_metrics: Optional[Dict[str, List[float]]] = None,
                        title: str = "Training Progress") -> go.Figure:
    """
    Plot training and validation curves with losses and optional metrics
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Optional dict of metric_name -> list of values per epoch
        val_metrics: Optional dict of metric_name -> list of values per epoch
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    
    # Determine number of subplots needed
    n_plots = 1  # Always have loss plot
    if train_metrics:
        n_plots += len(train_metrics)
    
    # Calculate subplot layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create subplot titles
    subplot_titles = ['Loss']
    if train_metrics:
        subplot_titles.extend([k.replace('_', ' ').title() for k in train_metrics.keys()])
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Plot losses
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=train_losses,
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=val_losses,
            mode='lines+markers',
            name='Val Loss',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Find best epoch (minimum validation loss)
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    
    # Add marker for best epoch on loss plot
    fig.add_trace(
        go.Scatter(
            x=[best_epoch],
            y=[best_val_loss],
            mode='markers',
            name=f'Best Epoch ({best_epoch})',
            marker=dict(color='green', size=12, symbol='star')
        ),
        row=1, col=1
    )
    
    # Plot additional metrics if provided
    if train_metrics and val_metrics:
        plot_idx = 2
        for metric_name in train_metrics.keys():
            row = ((plot_idx - 1) // n_cols) + 1
            col = ((plot_idx - 1) % n_cols) + 1
            
            # Training metric
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_metrics[metric_name],
                    mode='lines+markers',
                    name=f'Train {metric_name}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    showlegend=plot_idx == 2  # Only show legend for first metric
                ),
                row=row, col=col
            )
            
            # Validation metric
            if metric_name in val_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=val_metrics[metric_name],
                        mode='lines+markers',
                        name=f'Val {metric_name}',
                        line=dict(color='red', width=2),
                        marker=dict(size=4),
                        showlegend=plot_idx == 2
                    ),
                    row=row, col=col
                )
            
            plot_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        height=300 * n_rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Epoch", gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    # Add annotation about training details
    fig.add_annotation(
        text=f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})",
        xref="paper", yref="paper",
        x=0.02, y=-0.05,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
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