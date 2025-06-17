"""
Time series modeling for trading strategy development
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModeler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, df: pd.DataFrame, token: str, 
                    sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        # Filter for specific token and first 24 hours
        token_data = df[df['token'] == token].copy()
        token_data = token_data.sort_values('datetime')
        
        # Calculate returns
        token_data['returns'] = token_data['price'].pct_change()
        
        # Drop NaN values
        token_data = token_data.dropna()
        
        if len(token_data) < sequence_length + 1:
            raise ValueError(f"Not enough data points for token {token}. Need at least {sequence_length + 1} points.")
        
        # Create features
        features = ['price', 'returns']
        X = token_data[features].values
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:(i + sequence_length)])
            y_sequences.append(X_scaled[i + sequence_length, 0])  # Predict next price
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        series = series.dropna()
        if len(series) < 2:
            return {
                'adf_statistic': None,
                'p_value': None,
                'is_stationary': False
            }
        result = adfuller(series)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def fit_sarima(self, df: pd.DataFrame, token: str, 
                  order: Tuple[int, int, int] = (1, 1, 1),
                  seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)) -> Dict:
        """Fit SARIMA model to price data"""
        token_data = df[df['token'] == token].copy()
        token_data = token_data.sort_values('datetime')
        
        if len(token_data) < 50:  # Minimum required for SARIMA
            raise ValueError(f"Not enough data points for token {token}. Need at least 50 points.")
        
        # Check stationarity
        stationarity = self.check_stationarity(token_data['price'])
        
        try:
            # Fit SARIMA model
            model = SARIMAX(token_data['price'],
                          order=order,
                          seasonal_order=seasonal_order)
            
            results = model.fit(disp=False)
            
            # Make predictions
            forecast = results.forecast(steps=24)  # Predict next 24 periods
            
            return {
                'model': results,
                'forecast': forecast,
                'stationarity': stationarity,
                'aic': results.aic,
                'bic': results.bic
            }
        except Exception as e:
            print(f"Error fitting SARIMA for {token}: {str(e)}")
            return None
    
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model for price prediction (DISABLED)"""
        # Disabled due to TensorFlow issues
        # raise NotImplementedError("LSTM modeling is currently disabled. Please install TensorFlow and uncomment the code.")
        return None
    
    def fit_lstm(self, df: pd.DataFrame, token: str, 
                sequence_length: int = 60,
                epochs: int = 50,
                batch_size: int = 32) -> Dict:
        """Fit LSTM model to price data (DISABLED: TensorFlow not available)"""
        # Disabled due to TensorFlow issues
        # raise NotImplementedError("LSTM modeling is currently disabled. Please install TensorFlow and uncomment the code.")
        return None
    
    def generate_trading_signals(self, df: pd.DataFrame, token: str,
                               model_type: str = 'lstm',
                               threshold: float = 0.02) -> pd.DataFrame:
        """Generate trading signals based on model predictions"""
        try:
            if model_type == 'lstm':
                model_results = self.fit_lstm(df, token)
                if model_results is None:
                    raise ValueError(f"Failed to fit LSTM model for {token}")
                predictions = np.concatenate([
                    model_results['train_predictions'],
                    model_results['test_predictions']
                ])
            else:
                model_results = self.fit_sarima(df, token)
                if model_results is None:
                    raise ValueError(f"Failed to fit SARIMA model for {token}")
                predictions = model_results['forecast']
            
            # Get actual data
            token_data = df[df['token'] == token].copy()
            token_data = token_data.sort_values('datetime')
            
            # Ensure predictions and actual data have same length
            min_len = min(len(predictions), len(token_data))
            predictions = predictions[:min_len]
            token_data = token_data.iloc[:min_len]
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'datetime': token_data['datetime'].values,
                'actual_price': token_data['price'].values,
                'predicted_price': predictions,
            })
            
            # Calculate predicted returns
            signals['predicted_return'] = signals['predicted_price'].pct_change()
            signals['predicted_return'] = signals['predicted_return'].fillna(0)
            
            # Generate trading signals
            signals['signal'] = 0
            signals.loc[signals['predicted_return'] > threshold, 'signal'] = 1  # Buy
            signals.loc[signals['predicted_return'] < -threshold, 'signal'] = -1  # Sell
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals for {token}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def evaluate_strategy(self, signals: pd.DataFrame) -> Dict:
        """Evaluate trading strategy performance"""
        if signals.empty:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }
        
        try:
            # Calculate returns
            signals['strategy_return'] = signals['signal'].shift(1) * signals['actual_price'].pct_change()
            signals['strategy_return'] = signals['strategy_return'].fillna(0)
            
            # Calculate metrics
            total_return = (1 + signals['strategy_return']).prod() - 1
            returns_std = signals['strategy_return'].std()
            sharpe_ratio = (signals['strategy_return'].mean() / returns_std * np.sqrt(252)) if returns_std > 0 else 0
            max_drawdown = (signals['strategy_return'].cumsum() - 
                          signals['strategy_return'].cumsum().cummax()).min()
            
            return {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float((signals['strategy_return'] > 0).mean()),
                'num_trades': int((signals['signal'] != 0).sum())
            }
        except Exception as e:
            print(f"Error evaluating strategy: {str(e)}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            } 