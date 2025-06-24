"""
Gradio Interface for LSTM Predictions
Run this AFTER training to make predictions on new tokens
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import plotly.graph_objects as go
from typing import List, Tuple, Optional

# Import model architecture from training script
from train_lstm_model import LSTMPredictor


class TokenPredictor:
    """Handle predictions for individual tokens"""
    
    def __init__(self, model_path: str = 'lstm_model.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = None
        self.config = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model and configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = LSTMPredictor(
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=0,  # No dropout for inference
            forecast_horizon=self.config['forecast_horizon']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        self.scaler = checkpoint['scaler']
        
        print(f"Model loaded successfully!")
        print(f"Lookback: {self.config['lookback']} minutes")
        print(f"Forecast: {self.config['forecast_horizon']} minutes")
    
    def predict_token(self, file_path: str, show_last_n: int = 240) -> Tuple[go.Figure, str, str]:
        """Make prediction for a single token"""
        
        try:
            # Load price data
            df = pl.read_parquet(file_path)
            prices = df['price'].to_numpy()
            
            if len(prices) < self.config['lookback']:
                return None, "❌ Not enough data points", ""
            
            # Get last lookback window
            lookback_window = prices[-self.config['lookback']:]
            
            # Normalize
            lookback_norm = self.scaler.transform(lookback_window.reshape(-1, 1)).flatten()
            
            # Predict
            with torch.no_grad():
                x = torch.FloatTensor(lookback_norm).unsqueeze(0).to(self.device)
                pred_norm = self.model(x).cpu().numpy()[0]
            
            # Inverse transform prediction
            pred_prices = self.scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
            
            # Create plot
            fig = self.create_prediction_plot(prices, pred_prices, show_last_n)
            
            # Calculate metrics
            current_price = prices[-1]
            metrics = self.calculate_metrics(current_price, pred_prices)
            
            # Generate trading signal
            signal = self.generate_signal(metrics)
            
            return fig, metrics, signal
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}", ""
    
    def create_prediction_plot(self, prices: np.ndarray, predictions: np.ndarray, 
                              show_last_n: int) -> go.Figure:
        """Create interactive plot with predictions"""
        
        # Limit historical data shown
        if show_last_n > 0 and len(prices) > show_last_n:
            prices = prices[-show_last_n:]
            start_idx = len(prices)
        else:
            start_idx = len(prices)
        
        # Create time axis
        historical_time = list(range(len(prices)))
        future_time = list(range(len(prices), len(prices) + len(predictions)))
        
        # Create figure
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical_time,
            y=prices,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2),
            hovertemplate='Minute: %{x}<br>Price: $%{y:.6f}<extra></extra>'
        ))
        
        # Prediction
        fig.add_trace(go.Scatter(
            x=future_time,
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            hovertemplate='Minute: %{x}<br>Predicted: $%{y:.6f}<extra></extra>'
        ))
        
        # Connect last historical to first prediction
        fig.add_trace(go.Scatter(
            x=[historical_time[-1], future_time[0]],
            y=[prices[-1], predictions[0]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add shaded prediction region
        fig.add_vrect(
            x0=historical_time[-1], 
            x1=future_time[-1],
            fillcolor="red", 
            opacity=0.1,
            layer="below", 
            line_width=0,
        )
        
        # Mark entry point
        fig.add_trace(go.Scatter(
            x=[historical_time[-1]],
            y=[prices[-1]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='star'),
            name='Entry Point',
            hovertemplate='Entry: $%{y:.6f}<extra></extra>'
        ))
        
        # Mark predicted peak
        peak_idx = np.argmax(predictions)
        fig.add_trace(go.Scatter(
            x=[future_time[peak_idx]],
            y=[predictions[peak_idx]],
            mode='markers',
            marker=dict(size=12, color='gold', symbol='star'),
            name='Predicted Peak',
            hovertemplate='Peak at minute %{x}<br>Price: $%{y:.6f}<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title=f'Price Prediction - Next {len(predictions)} Minutes',
            xaxis_title='Time (minutes)',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def calculate_metrics(self, current_price: float, predictions: np.ndarray) -> str:
        """Calculate trading metrics"""
        
        max_price = predictions.max()
        min_price = predictions.min()
        final_price = predictions[-1]
        peak_time = np.argmax(predictions) + 1
        
        max_return = (max_price - current_price) / current_price * 100
        final_return = (final_price - current_price) / current_price * 100
        
        metrics = f"""
### 📊 Prediction Analysis

**Current Status:**
- 💰 Current Price: ${current_price:.6f}
- ⏱️ Prediction Window: {len(predictions)} minutes

**Price Predictions:**
- 📈 Max Price: ${max_price:.6f} ({max_return:+.1f}%)
- 📉 Min Price: ${min_price:.6f}
- 🎯 Final Price: ${final_price:.6f} ({final_return:+.1f}%)

**Optimal Strategy:**
- ⭐ Best Exit Time: Minute {peak_time}
- 💎 Maximum Return: {max_return:.1f}%
"""
        return metrics
    
    def generate_signal(self, metrics: str) -> str:
        """Generate trading signal based on metrics"""
        
        # Extract max return from metrics
        import re
        match = re.search(r'Maximum Return: ([\d.]+)%', metrics)
        if match:
            max_return = float(match.group(1))
        else:
            max_return = 0
        
        if max_return > 100:
            signal = "### 🚀 **STRONG BUY** - Major pump potential! (>100% gain)"
        elif max_return > 50:
            signal = "### 🟢 **BUY** - Significant opportunity (>50% gain)"
        elif max_return > 20:
            signal = "### 🟡 **MODERATE BUY** - Decent gains expected (>20%)"
        elif max_return > 10:
            signal = "### ⚪ **WEAK BUY** - Small gains possible (>10%)"
        else:
            signal = "### 🔴 **AVOID** - Limited upside (<10%)"
        
        return signal


class BatchAnalyzer:
    """Analyze multiple tokens at once"""
    
    def __init__(self, predictor: TokenPredictor):
        self.predictor = predictor
    
    def analyze_batch(self, files: List, top_n: int = 10) -> pd.DataFrame:
        """Analyze multiple tokens and rank by potential"""
        
        results = []
        
        for file in files:
            try:
                # Load data
                df = pl.read_parquet(file.name)
                prices = df['price'].to_numpy()
                
                if len(prices) < self.predictor.config['lookback']:
                    continue
                
                # Get prediction
                lookback_window = prices[-self.predictor.config['lookback']:]
                lookback_norm = self.predictor.scaler.transform(lookback_window.reshape(-1, 1)).flatten()
                
                with torch.no_grad():
                    x = torch.FloatTensor(lookback_norm).unsqueeze(0).to(self.predictor.device)
                    pred_norm = self.predictor.model(x).cpu().numpy()[0]
                
                pred_prices = self.predictor.scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                current_price = prices[-1]
                max_return = (pred_prices.max() - current_price) / current_price * 100
                peak_time = np.argmax(pred_prices) + 1
                
                results.append({
                    'Token': Path(file.name).stem,
                    'Current Price': f'${current_price:.6f}',
                    'Max Return %': f'{max_return:.1f}%',
                    'Peak Time (min)': peak_time,
                    'Signal': self._get_signal_emoji(max_return)
                })
                
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        # Create DataFrame and sort by return
        if results:
            df = pd.DataFrame(results)
            df['_sort_value'] = df['Max Return %'].str.rstrip('%').astype(float)
            df = df.sort_values('_sort_value', ascending=False).drop('_sort_value', axis=1)
            return df.head(top_n)
        else:
            return pd.DataFrame(columns=['Token', 'Current Price', 'Max Return %', 'Peak Time (min)', 'Signal'])
    
    def _get_signal_emoji(self, max_return: float) -> str:
        """Get emoji signal based on return"""
        if max_return > 100:
            return "🚀🚀🚀"
        elif max_return > 50:
            return "🚀🚀"
        elif max_return > 20:
            return "🚀"
        elif max_return > 10:
            return "📈"
        else:
            return "➡️"


# ================== GRADIO INTERFACE ==================
def create_interface():
    """Create Gradio interface"""
    
    # Initialize predictor
    predictor = TokenPredictor('lstm_model.pth')
    batch_analyzer = BatchAnalyzer(predictor)
    
    # CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .markdown-text {
        font-size: 14px;
    }
    """
    
    with gr.Blocks(title="🚀 Memecoin LSTM Predictor", css=css) as app:
        
        gr.Markdown("""
        # 🚀 Memecoin Price Prediction System
        
        Powered by LSTM neural network trained on 10,000+ memecoins
        """)
        
        with gr.Tab("📈 Single Token Prediction"):
            gr.Markdown("Upload a token's price data to get predictions for the next 15 minutes")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Token Data (.parquet)",
                        file_types=[".parquet"]
                    )
                    
                    show_last = gr.Slider(
                        minimum=60,
                        maximum=1440,
                        value=240,
                        step=60,
                        label="Show Last N Minutes"
                    )
                    
                    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="Price Prediction Chart")
            
            with gr.Row():
                with gr.Column():
                    metrics_output = gr.Markdown(label="Metrics")
                
                with gr.Column():
                    signal_output = gr.Markdown(label="Trading Signal")
            
            # Single prediction handler
            predict_btn.click(
                fn=lambda f, n: predictor.predict_token(f.name, n) if f else (None, "Please upload a file", ""),
                inputs=[file_input, show_last],
                outputs=[plot_output, metrics_output, signal_output]
            )
        
        with gr.Tab("📊 Batch Analysis"):
            gr.Markdown("Upload multiple tokens to find the best trading opportunities")
            
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        label="Upload Multiple Tokens",
                        file_count="multiple",
                        file_types=[".parquet"]
                    )
                    
                    top_n = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Show Top N Tokens"
                    )
                    
                    analyze_btn = gr.Button("🔍 Analyze All", variant="primary", size="lg")
                
                with gr.Column():
                    batch_output = gr.Dataframe(
                        label="Top Trading Opportunities",
                        headers=["Token", "Current Price", "Max Return %", "Peak Time (min)", "Signal"],
                        datatype=["str", "str", "str", "number", "str"]
                    )
            
            # Batch analysis handler
            analyze_btn.click(
                fn=lambda files, n: batch_analyzer.analyze_batch(files, n) if files else pd.DataFrame(),
                inputs=[batch_files, top_n],
                outputs=batch_output
            )
        
        with gr.Tab("ℹ️ Model Info"):
            gr.Markdown(f"""
            ### Model Configuration
            - **Architecture**: LSTM with {predictor.config['num_layers']} layers
            - **Hidden Size**: {predictor.config['hidden_size']}
            - **Lookback Window**: {predictor.config['lookback']} minutes
            - **Prediction Horizon**: {predictor.config['forecast_horizon']} minutes
            - **Training Data**: 10,000+ memecoins
            
            ### How to Use
            1. **Single Prediction**: Upload one token to see detailed prediction
            2. **Batch Analysis**: Upload multiple tokens to find best opportunities
            3. **Trading Signals**:
               - 🚀🚀🚀 = >100% potential gain
               - 🚀🚀 = >50% potential gain
               - 🚀 = >20% potential gain
               - 📈 = >10% potential gain
               - ➡️ = <10% potential gain
            
            ### Important Notes
            - Predictions assume you can trade at current price (no slippage)
            - Real trading involves spreads and liquidity constraints
            - Use predictions as one signal among many
            """)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False, server_port=7860)