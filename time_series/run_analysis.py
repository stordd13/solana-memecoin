"""
Run time series analysis and generate trading signals
"""

import pandas as pd
import numpy as np
from data_loader import DataLoader
from time_series_models import TimeSeriesModeler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def main():
    # Initialize data loader and modeler
    data_loader = DataLoader()
    modeler = TimeSeriesModeler()
    
    # Load data
    print("Loading data...")
    df = data_loader.load_data()
    
    # Get unique tokens
    tokens = df['token'].unique()
    
    # Store results
    results = {}
    
    # Analyze each token
    for token in tokens:
        print(f"\nAnalyzing {token}...")
        
        # Generate trading signals using both models
        lstm_signals = modeler.generate_trading_signals(df, token, model_type='lstm')
        sarima_signals = modeler.generate_trading_signals(df, token, model_type='sarima')
        
        # Evaluate strategies
        lstm_performance = modeler.evaluate_strategy(lstm_signals)
        sarima_performance = modeler.evaluate_strategy(sarima_signals)
        
        # Store results
        results[token] = {
            'lstm': lstm_performance,
            'sarima': sarima_performance
        }
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot price and predictions
        plt.subplot(2, 1, 1)
        plt.plot(lstm_signals['datetime'], lstm_signals['actual_price'], label='Actual Price')
        plt.plot(lstm_signals['datetime'], lstm_signals['predicted_price'], label='LSTM Prediction')
        plt.plot(sarima_signals['datetime'], sarima_signals['predicted_price'], label='SARIMA Prediction')
        plt.title(f'{token} Price and Predictions')
        plt.legend()
        
        # Plot trading signals
        plt.subplot(2, 1, 2)
        plt.plot(lstm_signals['datetime'], lstm_signals['signal'], label='LSTM Signals')
        plt.plot(sarima_signals['datetime'], sarima_signals['signal'], label='SARIMA Signals')
        plt.title(f'{token} Trading Signals')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'analysis_results/{token}_analysis.png')
        plt.close()
    
    # Save results
    with open('analysis_results/trading_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("----------------")
    for token, perf in results.items():
        print(f"\n{token}:")
        print("LSTM Strategy:")
        print(f"  Total Return: {perf['lstm']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {perf['lstm']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {perf['lstm']['win_rate']:.2%}")
        print("SARIMA Strategy:")
        print(f"  Total Return: {perf['sarima']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {perf['sarima']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {perf['sarima']['win_rate']:.2%}")

if __name__ == "__main__":
    main() 