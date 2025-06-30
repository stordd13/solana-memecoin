"""
Compare scaling methods for cryptocurrency data
Shows why Winsorization is better than RobustScaler for extreme outliers
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from winsorizer import Winsorizer


def generate_crypto_like_data(n_samples=1000):
    """Generate data that mimics cryptocurrency price movements"""
    # Base returns with some volatility
    base_returns = np.random.normal(0, 0.02, n_samples)
    
    # Add extreme events (pumps and dumps)
    pump_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    dump_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    
    base_returns[pump_indices] = np.random.uniform(2, 10, len(pump_indices))  # 200-1000% pumps
    base_returns[dump_indices] = np.random.uniform(-0.9, -0.5, len(dump_indices))  # 50-90% dumps
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(base_returns))
    
    # Add some features
    features = np.column_stack([
        prices,
        np.roll(prices, 1),  # Lag 1
        np.roll(prices, 5),  # Lag 5
        np.diff(np.concatenate([[prices[0]], prices])),  # Price changes
        base_returns * 100  # Returns in percentage
    ])
    
    return features[5:], base_returns[5:]  # Remove NaN rows


def compare_scalers():
    """Compare different scaling methods on crypto-like data"""
    # Generate data
    X, returns = generate_crypto_like_data()
    
    # Initialize scalers
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'Winsorizer (0.5%, 99.5%)': Winsorizer(0.005, 0.995),
        'Winsorizer (1%, 99%)': Winsorizer(0.01, 0.99)
    }
    
    # Transform data
    transformed_data = {}
    for name, scaler in scalers.items():
        if scaler is None:
            transformed_data[name] = X
        else:
            transformed_data[name] = scaler.fit_transform(X.copy())
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(transformed_data.items()):
        ax = axes[idx]
        
        # Plot histogram of first feature (prices)
        ax.hist(data[:, 0], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}')
        ax.set_xlabel('Scaled Values')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        stats_text = f'Mean: {np.mean(data[:, 0]):.3f}\n'
        stats_text += f'Std: {np.std(data[:, 0]):.3f}\n'
        stats_text += f'Min: {np.min(data[:, 0]):.3f}\n'
        stats_text += f'Max: {np.max(data[:, 0]):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('ML/results/scaling_comparison.png', dpi=150)
    print("✅ Scaling comparison saved to ML/results/scaling_comparison.png")
    
    # Print numerical comparison
    print("\n" + "="*60)
    print("NUMERICAL COMPARISON OF SCALING METHODS")
    print("="*60)
    
    # Check for issues with RobustScaler
    robust_scaler = RobustScaler()
    robust_scaler.fit(X)
    
    print("\nRobustScaler IQR values (scale_):")
    for i, iqr in enumerate(robust_scaler.scale_[:5]):
        print(f"  Feature {i}: {iqr:.6f} {'⚠️ ZERO IQR!' if iqr < 1e-6 else ''}")
    
    # Compare outlier handling
    extreme_sample = np.array([[10000, 9000, 8000, 1000, 500]])  # Extreme pump
    
    print("\nHandling extreme outlier (10x pump):")
    for name, scaler in scalers.items():
        if scaler is not None:
            transformed = scaler.transform(extreme_sample)
            print(f"  {name}: {transformed[0, 0]:.3f}")


if __name__ == "__main__":
    compare_scalers() 