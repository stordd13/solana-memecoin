# visualization_gradio.py
"""
Shared Gradio visualization module for interactive interfaces.
Used across all phase scripts with standardized components.
"""

import gradio as gr
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class GradioVisualizer:
    """
    Shared Gradio interface components for memecoin analysis visualization.
    """
    
    def __init__(self, title: str = "Memecoin Analysis"):
        self.title = title
        self.current_results = None
        
    def create_k_selection_plot(self, k_analysis: Dict) -> go.Figure:
        """Create interactive K-selection plot with elbow and silhouette."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Elbow curve
        fig.add_trace(
            go.Scatter(
                x=k_analysis['k_range'],
                y=k_analysis['inertias'],
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Mark optimal K from elbow
        optimal_k_elbow = k_analysis['optimal_k_elbow']
        if optimal_k_elbow in k_analysis['k_range']:
            idx = k_analysis['k_range'].index(optimal_k_elbow)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k_elbow],
                    y=[k_analysis['inertias'][idx]],
                    mode='markers',
                    name=f'Elbow K={optimal_k_elbow}',
                    marker=dict(color='red', size=15, symbol='star')
                ),
                row=1, col=1
            )
        
        # Silhouette scores
        fig.add_trace(
            go.Scatter(
                x=k_analysis['k_range'],
                y=k_analysis['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Mark optimal K from silhouette
        optimal_k_sil = k_analysis['optimal_k_silhouette']
        if optimal_k_sil in k_analysis['k_range']:
            idx = k_analysis['k_range'].index(optimal_k_sil)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k_sil],
                    y=[k_analysis['silhouette_scores'][idx]],
                    mode='markers',
                    name=f'Best Silhouette K={optimal_k_sil}',
                    marker=dict(color='orange', size=15, symbol='star')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Optimal Number of Clusters Analysis',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        return fig
    
    def create_stability_plot(self, stability: Dict) -> go.Figure:
        """Create stability analysis visualization."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ARI Stability', 'Silhouette Stability'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        runs = list(range(1, len(stability['ari_scores']) + 1))
        
        # ARI scores
        fig.add_trace(
            go.Scatter(
                x=runs,
                y=stability['ari_scores'],
                mode='lines+markers',
                name='ARI Scores',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # CEO threshold line for ARI
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                     annotation_text="CEO Threshold (0.75)", row=1, col=1)
        
        # Mean ARI line
        fig.add_hline(y=stability['mean_ari'], line_dash="solid", line_color="green",
                     annotation_text=f"Mean ARI ({stability['mean_ari']:.3f})", row=1, col=1)
        
        # Silhouette scores
        fig.add_trace(
            go.Scatter(
                x=runs,
                y=stability['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette Scores',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # CEO threshold line for Silhouette
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="CEO Threshold (0.5)", row=1, col=2)
        
        # Mean silhouette line
        fig.add_hline(y=stability['mean_silhouette'], line_dash="solid", line_color="blue",
                     annotation_text=f"Mean Silhouette ({stability['mean_silhouette']:.3f})", row=1, col=2)
        
        fig.update_layout(
            title='Clustering Stability Analysis',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Stability Run", row=1, col=1)
        fig.update_xaxes(title_text="Stability Run", row=1, col=2)
        fig.update_yaxes(title_text="ARI Score", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        return fig
    
    def create_tsne_plot(self, tsne_coords: np.ndarray, labels: np.ndarray, 
                        token_names: List[str]) -> go.Figure:
        """Create interactive t-SNE visualization."""
        fig = go.Figure()
        
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            cluster_tokens = [token_names[j] for j in np.where(mask)[0]]
            
            fig.add_trace(go.Scatter(
                x=tsne_coords[mask, 0],
                y=tsne_coords[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_tokens,
                marker=dict(size=8, color=colors[i % len(colors)]),
                hovertemplate='<b>%{text}</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="t-SNE Visualization of Token Clusters",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def create_tsne_plot_3d(self, tsne_coords_3d: np.ndarray, labels: np.ndarray, 
                           token_names: List[str]) -> go.Figure:
        """Create interactive 3D t-SNE visualization."""
        fig = go.Figure()
        
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            cluster_tokens = [token_names[j] for j in np.where(mask)[0]]
            
            fig.add_trace(go.Scatter3d(
                x=tsne_coords_3d[mask, 0],
                y=tsne_coords_3d[mask, 1],
                z=tsne_coords_3d[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_tokens,
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                hovertemplate='<b>%{text}</b><br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<br>t-SNE 3: %{z:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="3D t-SNE Visualization of Token Clusters",
            scene=dict(
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
                zaxis_title="t-SNE Component 3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            hovermode='closest'
        )
        
        return fig
    
    def create_feature_comparison_plot(self, raw_features: Dict, log_features: Dict) -> go.Figure:
        """Create comparison plot for A/B testing raw vs log returns."""
        from plotly.subplots import make_subplots
        
        # Calculate feature statistics for comparison
        raw_stats = self._calculate_feature_stats(raw_features)
        log_stats = self._calculate_feature_stats(log_features)
        
        feature_names = list(raw_stats.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Means', 'Feature Std Deviations', 
                          'Feature Ranges', 'Feature Correlations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Feature means comparison
        fig.add_trace(
            go.Bar(x=feature_names, y=[raw_stats[f]['mean'] for f in feature_names],
                  name='Raw Returns', marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=feature_names, y=[log_stats[f]['mean'] for f in feature_names],
                  name='Log Returns', marker_color='red'),
            row=1, col=2
        )
        
        # Feature standard deviations
        fig.add_trace(
            go.Bar(x=feature_names, y=[raw_stats[f]['std'] for f in feature_names],
                  name='Raw Returns Std', marker_color='lightblue', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=feature_names, y=[log_stats[f]['std'] for f in feature_names],
                  name='Log Returns Std', marker_color='lightcoral', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Raw vs Log Returns Feature Comparison',
            height=800,
            showlegend=True
        )
        
        # Rotate x-axis labels for readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _calculate_feature_stats(self, features_dict: Dict) -> Dict:
        """Calculate statistics for features dictionary."""
        if not features_dict:
            return {}
        
        # Convert to arrays for each feature
        feature_arrays = {}
        feature_names = list(next(iter(features_dict.values())).keys())
        
        for feature_name in feature_names:
            values = [features[feature_name] for features in features_dict.values() 
                     if np.isfinite(features[feature_name])]
            if values:
                feature_arrays[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
            else:
                feature_arrays[feature_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'range': 0.0
                }
        
        return feature_arrays
    
    def create_metrics_summary(self, results: Dict) -> str:
        """Create formatted metrics summary for display."""
        summary = "## Analysis Results Summary\n\n"
        
        if 'meets_ceo_requirements' in results:
            req = results['meets_ceo_requirements']
            summary += "### CEO Requirements Status:\n"
            summary += f"- **ARI Threshold (≥0.75)**: {'✅ PASS' if req['ari_threshold'] else '❌ FAIL'}\n"
            summary += f"- **Silhouette Threshold (≥0.5)**: {'✅ PASS' if req['silhouette_threshold'] else '❌ FAIL'}\n"
            summary += f"- **Overall Status**: {'🎉 SUCCESS' if req['stability_achieved'] else '⚠️ NEEDS IMPROVEMENT'}\n\n"
        
        if 'optimal_k' in results:
            summary += f"### Optimal K Selected: {results['optimal_k']}\n\n"
        
        if 'stability' in results:
            stab = results['stability']
            summary += "### Stability Metrics:\n"
            summary += f"- **Mean ARI**: {stab['mean_ari']:.4f} ± {stab['std_ari']:.4f}\n"
            summary += f"- **Mean Silhouette**: {stab['mean_silhouette']:.4f} ± {stab['std_silhouette']:.4f}\n"
            summary += f"- **Stability Runs**: {stab['stability_runs']}\n\n"
        
        if 'final_clustering' in results:
            cluster = results['final_clustering']
            summary += "### Final Clustering:\n"
            summary += f"- **Clusters**: {cluster['n_clusters']}\n"
            summary += f"- **Silhouette Score**: {cluster['silhouette_score']:.4f}\n"
            summary += f"- **Inertia**: {cluster['inertia']:.2f}\n\n"
        
        return summary
    
    def launch_interface(self, analysis_function, 
                        interface_components: List = None,
                        share: bool = False,
                        server_port: int = 7860) -> gr.Interface:
        """
        Launch Gradio interface with standardized layout.
        
        Args:
            analysis_function: Main analysis function to call
            interface_components: Custom interface components
            share: Whether to create public link
            server_port: Port for local server
            
        Returns:
            Gradio Interface object
        """
        def run_analysis(*args):
            """Wrapper for analysis function."""
            try:
                results = analysis_function(*args)
                self.current_results = results
                
                # Generate summary
                summary = self.create_metrics_summary(results)
                
                # Generate plots if available
                plots = []
                if 'k_analysis' in results:
                    plots.append(self.create_k_selection_plot(results['k_analysis']))
                
                if 'stability' in results:
                    plots.append(self.create_stability_plot(results['stability']))
                
                if 'tsne_2d' in results and 'final_clustering' in results:
                    plots.append(self.create_tsne_plot(
                        results['tsne_2d'], 
                        results['final_clustering']['labels'],
                        results['token_names']
                    ))
                
                return summary, *plots
                
            except Exception as e:
                error_msg = f"## Error During Analysis\n\n```\n{str(e)}\n```"
                return error_msg, None, None, None
        
        # Default interface if no custom components provided
        if interface_components is None:
            interface_components = [
                gr.Number(value=1000, label="Number of Tokens"),
                gr.Slider(minimum=3, maximum=15, step=1, value=5, label="Number of Clusters"),
                gr.Checkbox(value=False, label="Use Log Returns"),
                gr.Number(value=42, label="Random Seed")
            ]
        
        # Create interface
        interface = gr.Interface(
            fn=run_analysis,
            inputs=interface_components,
            outputs=[
                gr.Markdown(label="Results Summary"),
                gr.Plot(label="K-Selection Analysis"),
                gr.Plot(label="Stability Analysis"),
                gr.Plot(label="t-SNE Visualization")
            ],
            title=self.title,
            description="Interactive Memecoin Behavioral Archetype Analysis",
            allow_flagging="never"
        )
        
        # Launch interface
        interface.launch(share=share, server_port=server_port)
        
        return interface


def create_comparison_interface(raw_results: Dict, log_results: Dict) -> gr.Interface:
    """
    Create comparison interface for A/B testing raw vs log returns.
    
    Args:
        raw_results: Results from raw returns analysis
        log_results: Results from log returns analysis
        
    Returns:
        Gradio Interface for comparison
    """
    def display_comparison():
        comparison_text = "## Raw vs Log Returns Comparison\n\n"
        
        # Compare key metrics
        if 'stability' in raw_results and 'stability' in log_results:
            raw_ari = raw_results['stability']['mean_ari']
            log_ari = log_results['stability']['mean_ari']
            raw_sil = raw_results['final_clustering']['silhouette_score']
            log_sil = log_results['final_clustering']['silhouette_score']
            
            comparison_text += "### Key Metrics Comparison:\n"
            comparison_text += f"| Metric | Raw Returns | Log Returns | Winner |\n"
            comparison_text += f"|--------|-------------|-------------|--------|\n"
            comparison_text += f"| Mean ARI | {raw_ari:.4f} | {log_ari:.4f} | {'Raw' if raw_ari > log_ari else 'Log'} |\n"
            comparison_text += f"| Silhouette | {raw_sil:.4f} | {log_sil:.4f} | {'Raw' if raw_sil > log_sil else 'Log'} |\n\n"
            
            # Determine overall winner
            raw_score = (raw_ari >= 0.75) + (raw_sil >= 0.5) + (raw_ari > log_ari) + (raw_sil > log_sil)
            log_score = (log_ari >= 0.75) + (log_sil >= 0.5) + (log_ari > raw_ari) + (log_sil > raw_sil)
            
            if raw_score > log_score:
                comparison_text += "### 🏆 **WINNER: Raw Returns**\n"
            elif log_score > raw_score:
                comparison_text += "### 🏆 **WINNER: Log Returns**\n"
            else:
                comparison_text += "### 🤝 **TIE: Both methods perform similarly**\n"
        
        return comparison_text
    
    interface = gr.Interface(
        fn=display_comparison,
        inputs=[],
        outputs=gr.Markdown(),
        title="A/B Test Results: Raw vs Log Returns",
        description="Comparison of clustering performance between raw and log returns"
    )
    
    return interface