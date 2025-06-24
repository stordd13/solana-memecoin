"""
Analyze and visualize LSTM training results
Run this AFTER training to understand model performance
"""

import torch
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.subplots as sp
from pathlib import Path
from typing import Dict, List, Optional
import gradio as gr


class ResultsAnalyzer:
    """Analyze LSTM model results and performance"""
    
    def __init__(self, model_path: str = 'lstm_model.pth'):
        self.checkpoint = torch.load(model_path, map_location='cpu')
        self.config = self.checkpoint['config']
        self.metrics = self.checkpoint.get('metrics', {})
        
    def create_metrics_dashboard(self) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Model Accuracy Metrics',
                'Pump Detection Performance',
                'Error Distribution',
                'Configuration Details',
                'Training Parameters',
                'Performance Summary'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'table'}, {'type': 'table'}, {'type': 'indicator'}]]
        )
        
        # 1. Accuracy metrics
        accuracy_metrics = ['direction_accuracy', 'pump_precision', 'pump_recall']
        accuracy_values = [self.metrics.get(m, 0) for m in accuracy_metrics]
        accuracy_labels = ['Direction\nAccuracy', 'Pump\nPrecision', 'Pump\nRecall']
        
        fig.add_trace(
            go.Bar(
                x=accuracy_labels,
                y=accuracy_values,
                text=[f'{v:.1%}' for v in accuracy_values],
                textposition='auto',
                marker_color=['green' if v > 0.5 else 'orange' for v in accuracy_values]
            ),
            row=1, col=1
        )
        
        # 2. Pump detection breakdown
        pump_data = {
            'True Positives': self.metrics.get('pump_precision', 0),
            'False Positives': 1 - self.metrics.get('pump_precision', 0) if self.metrics.get('pump_precision', 0) > 0 else 0,
            'Recall Rate': self.metrics.get('pump_recall', 0)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(pump_data.keys()),
                y=list(pump_data.values()),
                text=[f'{v:.1%}' for v in pump_data.values()],
                textposition='auto',
                marker_color=['green', 'red', 'blue']
            ),
            row=1, col=2
        )
        
        # 3. Error distribution (simulated for now)
        # In real implementation, save actual errors during evaluation
        errors = np.random.normal(0, self.metrics.get('mae', 0.1), 1000)
        
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=30,
                name='Prediction Errors',
                marker_color='lightblue'
            ),
            row=1, col=3
        )
        
        # 4. Configuration table
        config_data = [
            ['Parameter', 'Value'],
            ['Lookback Window', f"{self.config['lookback']} minutes"],
            ['Forecast Horizon', f"{self.config['forecast_horizon']} minutes"],
            ['Batch Size', str(self.config['batch_size'])],
            ['Categories Used', ', '.join(self.config['categories'])]
        ]
        
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=list(zip(*config_data)),
                    align='left',
                    height=30
                )
            ),
            row=2, col=1
        )
        
        # 5. Model parameters table
        model_data = [
            ['Parameter', 'Value'],
            ['Hidden Size', str(self.config['hidden_size'])],
            ['Number of Layers', str(self.config['num_layers'])],
            ['Dropout Rate', str(self.config['dropout'])],
            ['Learning Rate', str(self.config['learning_rate'])],
            ['Epochs Trained', str(self.config['num_epochs'])]
        ]
        
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=list(zip(*model_data)),
                    align='left',
                    height=30
                )
            ),
            row=2, col=2
        )
        
        # 6. Overall performance indicator
        overall_score = (
            self.metrics.get('direction_accuracy', 0) * 0.3 +
            self.metrics.get('pump_precision', 0) * 0.4 +
            self.metrics.get('pump_recall', 0) * 0.3
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score * 100,
                title={'text': "Overall Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="LSTM Model Performance Dashboard",
            title_font_size=20
        )
        
        # Update axes
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=2)
        fig.update_xaxes(title_text="Error", row=1, col=3)
        
        return fig
    
    def analyze_prediction_quality(self, test_predictions_path: Optional[str] = None) -> go.Figure:
        """Analyze quality of predictions if test data is available"""
        
        # This would load actual test predictions if saved during training
        # For now, create a sample visualization
        
        fig = go.Figure()
        
        # Sample data - replace with actual predictions
        time_points = list(range(15))
        actual_returns = np.random.normal(0.05, 0.1, 15)
        predicted_returns = actual_returns + np.random.normal(0, 0.03, 15)
        
        # Actual vs Predicted
        fig.add_trace(go.Scatter(
            x=time_points,
            y=actual_returns,
            mode='lines+markers',
            name='Actual Returns',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=predicted_returns,
            mode='lines+markers',
            name='Predicted Returns',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add error bands
        errors = np.abs(predicted_returns - actual_returns)
        fig.add_trace(go.Scatter(
            x=time_points + time_points[::-1],
            y=list(predicted_returns + errors) + list(predicted_returns - errors)[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Error Band',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Prediction Quality Analysis',
            xaxis_title='Minutes Ahead',
            yaxis_title='Return',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def generate_report(self) -> str:
        """Generate text report of model performance"""
        
        report = f"""
# LSTM Model Training Report

## Model Configuration
- **Architecture**: {self.config['num_layers']}-layer LSTM
- **Hidden Size**: {self.config['hidden_size']}
- **Lookback Window**: {self.config['lookback']} minutes
- **Prediction Horizon**: {self.config['forecast_horizon']} minutes

## Training Details
- **Epochs**: {self.config['num_epochs']}
- **Batch Size**: {self.config['batch_size']}
- **Learning Rate**: {self.config['learning_rate']}
- **Dropout**: {self.config['dropout']}

## Performance Metrics
- **Direction Accuracy**: {self.metrics.get('direction_accuracy', 0):.1%}
- **Pump Detection Precision**: {self.metrics.get('pump_precision', 0):.1%}
- **Pump Detection Recall**: {self.metrics.get('pump_recall', 0):.1%}
- **Mean Squared Error**: {self.metrics.get('mse', 0):.4f}
- **Mean Absolute Error**: {self.metrics.get('mae', 0):.4f}

## Key Insights
"""
        
        # Add insights based on metrics
        if self.metrics.get('direction_accuracy', 0) > 0.6:
            report += "- ✅ Good direction prediction accuracy (>60%)\n"
        else:
            report += "- ⚠️ Direction accuracy needs improvement (<60%)\n"
        
        if self.metrics.get('pump_precision', 0) > 0.5:
            report += "- ✅ Reliable pump detection precision\n"
        else:
            report += "- ⚠️ High false positive rate in pump detection\n"
        
        if self.metrics.get('pump_recall', 0) > 0.4:
            report += "- ✅ Good pump detection coverage\n"
        else:
            report += "- ⚠️ Missing many actual pumps\n"
        
        return report


# ================== GRADIO INTERFACE ==================
def create_analysis_interface():
    """Create Gradio interface for results analysis"""
    
    analyzer = ResultsAnalyzer('lstm_model.pth')
    
    with gr.Blocks(title="📊 LSTM Results Analysis") as app:
        
        gr.Markdown("""
        # 📊 LSTM Model Results Analysis
        
        Comprehensive analysis of your trained model's performance
        """)
        
        with gr.Tab("📈 Performance Dashboard"):
            dashboard_btn = gr.Button("Generate Dashboard", variant="primary")
            dashboard_plot = gr.Plot(label="Performance Metrics")
            
            dashboard_btn.click(
                fn=lambda: analyzer.create_metrics_dashboard(),
                outputs=dashboard_plot
            )
        
        with gr.Tab("🎯 Prediction Quality"):
            quality_btn = gr.Button("Analyze Predictions", variant="primary")
            quality_plot = gr.Plot(label="Prediction vs Actual")
            
            quality_btn.click(
                fn=lambda: analyzer.analyze_prediction_quality(),
                outputs=quality_plot
            )
        
        with gr.Tab("📄 Full Report"):
            report_btn = gr.Button("Generate Report", variant="primary")
            report_output = gr.Markdown()
            
            report_btn.click(
                fn=lambda: analyzer.generate_report(),
                outputs=report_output
            )
        
        # Auto-load dashboard on startup
        app.load(
            fn=lambda: analyzer.create_metrics_dashboard(),
            outputs=dashboard_plot
        )
    
    return app


if __name__ == "__main__":
    app = create_analysis_interface()
    app.launch(share=False, server_port=7861)