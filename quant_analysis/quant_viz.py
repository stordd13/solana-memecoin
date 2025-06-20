"""
Quantitative visualizations for memecoin analysis
Professional financial market analysis plots using only price and timestamp
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import warnings
from collections import defaultdict
from scipy import stats
import numpy as np
warnings.filterwarnings('ignore')

class QuantVisualizations:
    """Create professional quant analysis plots for memecoin data"""
    
    def __init__(self):
        self.style_config = {
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        }
        plt.rcParams.update(self.style_config)
    
    def plot_entry_exit_matrix(self, df: pl.DataFrame, 
                              entry_windows: List[int] = [5, 10, 15, 30, 60],
                              exit_windows: List[int] = [5, 10, 15, 30, 60]) -> go.Figure:
        """
        Plot optimal entry/exit timing matrix
        Shows average returns for different entry/exit window combinations
        """
        matrix = pl.DataFrame(index=entry_windows, columns=exit_windows)
        
        for entry_window in entry_windows:
            for exit_window in exit_windows:
                returns = []
                
                # Calculate returns for each combination
                for i in range(entry_window, len(df) - exit_window, entry_window):
                    # Entry signal: positive momentum
                    entry_momentum = (df['price'].to_numpy()[i] / df['price'].to_numpy()[i-entry_window] - 1)
                    
                    if entry_momentum > 0:  # Enter on positive momentum
                        entry_price = df['price'].to_numpy()[i]
                        exit_price = df['price'].to_numpy()[i + exit_window]
                        trade_return = (exit_price / entry_price - 1) * 100
                        returns.append(trade_return)
                
                matrix.loc[entry_window, exit_window] = pl.Series(returns).mean() if returns else 0
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix.to_numpy().astype(float),
            x=[f"{w}min" for w in exit_windows],
            y=[f"{w}min" for w in entry_windows],
            colorscale='RdBu',
            zmid=0,
            text=np.round(matrix.to_numpy().astype(float), 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Avg Return %")
        ))
        
        fig.update_layout(
            title="Entry/Exit Timing Matrix - Average Returns (%)",
            xaxis_title="Exit Window",
            yaxis_title="Entry Window",
            height=600,
            width=800
        )
        
        return fig
    
    def plot_price_momentum_heatmap(self, df: pl.DataFrame,
                                   periods: List[int] = [5, 10, 15, 30, 60, 120, 240]) -> go.Figure:
        """
        Create heatmap of momentum across different lookback periods over time
        """
        # Calculate momentum for each period
        momentum_data = pl.DataFrame()
        
        for period in periods:
            momentum_data[f'{period}min'] = df['price'].pct_change(period) * 100
        
        # Resample to reduce data points for heatmap
        resample_freq = max(1, len(df) // 100)  # Max 100 time points
        momentum_resampled = momentum_data.iloc[::resample_freq].T
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=momentum_resampled.to_numpy(),
            x=df['datetime'].to_numpy()[::resample_freq],
            y=[f'{p}min' for p in periods],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Return %"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Momentum Heatmap - Returns Across Different Timeframes",
            xaxis_title="Time",
            yaxis_title="Lookback Period",
            height=600
        )
        
        return fig
    
    def plot_risk_adjusted_performance(self, dfs: List[pl.DataFrame], 
                                     names: List[str]) -> go.Figure:
        """
        Compare risk-adjusted performance metrics across multiple tokens
        """
        metrics_data = []
        
        for df, name in zip(dfs, names):
            # Use log returns for all risk metrics
            returns = np.log(df['price'] / df['price'].shift(1)).drop_nulls().to_numpy()
            
            # Calculate metrics
            total_return = (df['price'].to_numpy()[-1] / df['price'].to_numpy()[0] - 1) * 100
            volatility = returns.std() * np.sqrt(525600) * 100  # Annualized
            
            # Sharpe ratio
            sharpe = returns.mean() / returns.std() * np.sqrt(525600) if returns.std() > 0 else 0
            
            # Max drawdown (on cumulative log returns)
            cumulative = pl.Series(returns.cumsum().apply(np.exp)).to_numpy()
            running_max = pl.Series(cumulative.expanding().max()).to_numpy()
            drawdown = ((cumulative - running_max) / running_max).min() * 100
            
            # Calmar ratio (return / max drawdown)
            calmar = total_return / abs(drawdown) if drawdown != 0 else 0
            
            metrics_data.append({
                'Token': name,
                'Total Return %': total_return,
                'Volatility %': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown %': drawdown,
                'Calmar Ratio': calmar
            })
        
        metrics_df = pl.DataFrame(metrics_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return vs Volatility', 'Sharpe Ratios',
                          'Max Drawdown', 'Calmar Ratios'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Scatter plot: Return vs Volatility
        fig.add_trace(
            go.Scatter(
                x=metrics_df['Volatility %'],
                y=metrics_df['Total Return %'],
                mode='markers+text',
                text=metrics_df['Token'],
                textposition="top center",
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Bar chart: Sharpe Ratios
        fig.add_trace(
            go.Bar(x=metrics_df['Token'], y=metrics_df['Sharpe Ratio']),
            row=1, col=2
        )
        
        # Bar chart: Max Drawdown
        fig.add_trace(
            go.Bar(x=metrics_df['Token'], y=metrics_df['Max Drawdown %']),
            row=2, col=1
        )
        
        # Bar chart: Calmar Ratios
        fig.add_trace(
            go.Bar(x=metrics_df['Token'], y=metrics_df['Calmar Ratio']),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Volatility %", row=1, col=1)
        fig.update_yaxes(title_text="Total Return %", row=1, col=1)
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Risk-Adjusted Performance Comparison")
        
        return fig
    
    def plot_temporal_risk_reward(self, df: pl.DataFrame,
                                 time_horizons: List[int] = [5, 15, 30, 60, 120, 240]) -> go.Figure:
        """
        Plot risk/reward ratios across different time horizons
        """
        results = []
        
        for horizon in time_horizons:
            # Calculate rolling returns for this horizon
            rolling_returns = df['price'].pct_change(horizon)
            
            # Separate positive and negative returns
            positive_returns = rolling_returns[rolling_returns > 0]
            negative_returns = rolling_returns[rolling_returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_gain = positive_returns.mean() * 100
                avg_loss = abs(negative_returns.mean()) * 100
                win_rate = len(positive_returns) / len(rolling_returns.drop_nulls()) * 100
                
                # Risk/Reward ratio
                risk_reward = avg_gain / avg_loss if avg_loss > 0 else 0
                
                # Sharpe ratio approximation
                sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(525600 / horizon)
                
                results.append({
                    'horizon': horizon,
                    'win_rate': win_rate,
                    'avg_gain': avg_gain,
                    'avg_loss': avg_loss,
                    'risk_reward': risk_reward,
                    'sharpe': sharpe
                })
        
        results_df = pl.DataFrame(results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate by Time Horizon', 'Risk/Reward Ratio',
                          'Average Gain vs Loss', 'Sharpe Ratio'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Win Rate
        fig.add_trace(
            go.Scatter(x=results_df['horizon'], y=results_df['win_rate'],
                      mode='lines+markers', name='Win Rate %',
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        # Risk/Reward
        fig.add_trace(
            go.Scatter(x=results_df['horizon'], y=results_df['risk_reward'],
                      mode='lines+markers', name='Risk/Reward',
                      line=dict(color='blue', width=3)),
            row=1, col=2
        )
        
        # Gain vs Loss bars
        fig.add_trace(
            go.Bar(x=results_df['horizon'], y=results_df['avg_gain'],
                   name='Avg Gain %', marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=results_df['horizon'], y=-results_df['avg_loss'],
                   name='Avg Loss %', marker_color='red'),
            row=2, col=1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(x=results_df['horizon'], y=results_df['sharpe'],
                      mode='lines+markers', name='Sharpe Ratio',
                      line=dict(color='purple', width=3)),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Time Horizon (minutes)")
        fig.update_layout(height=800, showlegend=True,
                         title_text="Temporal Risk/Reward Analysis")
        
        return fig
    
    def plot_volatility_surface(self, df: pl.DataFrame,
                               windows: List[int] = [5, 10, 15, 30, 60, 120],
                               percentiles: List[int] = [10, 25, 50, 75, 90]) -> go.Figure:
        """
        CORRECTED: Plot volatility surface across different time windows and percentiles
        Fixed logic: proper rolling volatility calculation with correct annualization
        """
        if df.is_empty() or len(df) < max(windows):
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for volatility surface", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate returns first (this is the correct approach)
        df_with_returns = df.with_columns([
            df['price'].pct_change().alias('returns')
        ])
        
        # Calculate rolling volatility for different windows using proper Polars operations
        volatility_results = []
        
        for window in windows:
            if len(df_with_returns) >= window * 2:  # Need enough data
                # Calculate rolling standard deviation of returns (correct approach)
                rolling_vol = df_with_returns.with_columns([
                    df_with_returns['returns'].rolling_std(window_size=window, min_periods=max(1, window//2)).alias('rolling_vol')
                ])['rolling_vol'].drop_nulls()
                
                # Annualize volatility properly for minute data
                # sqrt(525600) for minute-to-annual conversion
                annualized_vol = rolling_vol * np.sqrt(525600 / window)
                
                # Calculate percentiles for this window
                for percentile in percentiles:
                    vol_value = annualized_vol.quantile(percentile / 100)
                    if vol_value is not None and not np.isnan(vol_value):
                        volatility_results.append({
                            'window': window,
                            'percentile': percentile,
                            'volatility': vol_value * 100  # Convert to percentage
                        })
        
        if not volatility_results:
            fig = go.Figure()
            fig.add_annotation(text="No valid volatility data calculated", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        surface_df = pl.DataFrame(volatility_results)
        
        # Pivot for surface plot using Polars
        pivot_df = surface_df.pivot(index='percentile', columns='window', values='volatility')
        
        # Convert to numpy for plotly
        percentiles_array = sorted(surface_df['percentile'].unique().to_list())
        windows_array = sorted(surface_df['window'].unique().to_list())
        
        # Create Z matrix for surface plot
        z_matrix = []
        for percentile in percentiles_array:
            row = []
            for window in windows_array:
                # Find the volatility value for this percentile-window combination
                vol_val = surface_df.filter(
                    (pl.col('percentile') == percentile) & (pl.col('window') == window)
                )['volatility']
                if len(vol_val) > 0:
                    row.append(vol_val[0])
                else:
                    row.append(np.nan)
            z_matrix.append(row)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=windows_array,
            y=percentiles_array,
            z=z_matrix,
            colorscale='Viridis',
            colorbar=dict(title="Annualized Volatility %"),
            hovertemplate='Window: %{x} min<br>Percentile: %{y}<br>Volatility: %{z:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title="Volatility Surface - Time Window vs Percentile<br>(Corrected: Proper Rolling Volatility Calculation)",
            scene=dict(
                xaxis_title="Time Window (minutes)",
                yaxis_title="Percentile",
                zaxis_title="Annualized Volatility (%)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            width=1000
        )
        
        return fig
    
    def plot_volatility_analysis_dashboard(self, df: pl.DataFrame, windows: List[int] = [5, 10, 30, 60, 240]) -> go.Figure:
        """
        Create comprehensive volatility analysis dashboard with multiple visualizations
        """
        if df.is_empty() or len(df) < max(windows):
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for volatility analysis", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate returns and various volatility metrics using Polars
        df_analysis = df.with_columns([
            df['price'].pct_change().alias('returns')
        ])
        
        # Calculate rolling volatilities for different windows
        volatility_data = []
        for window in windows:
            if len(df_analysis) >= window * 2:
                rolling_vol = df_analysis.with_columns([
                    df_analysis['returns'].rolling_std(window_size=window, min_periods=max(1, window//2)).alias('rolling_vol')
                ])['rolling_vol'].drop_nulls() * np.sqrt(525600 / window) * 100  # Annualize and convert to %
                
                if len(rolling_vol) > 0:
                    volatility_data.append({
                        'window': window,
                        'mean_vol': rolling_vol.mean(),
                        'median_vol': rolling_vol.median(),
                        'std_vol': rolling_vol.std(),
                        'min_vol': rolling_vol.min(),
                        'max_vol': rolling_vol.max(),
                        'q25_vol': rolling_vol.quantile(0.25),
                        'q75_vol': rolling_vol.quantile(0.75)
                    })
        
        if not volatility_data:
            fig = go.Figure()
            fig.add_annotation(text="No valid volatility data calculated", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        vol_df = pl.DataFrame(volatility_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Volatility by Time Window',
                'Volatility Range (Min-Max)',
                'Volatility Distribution (Box Plot Style)',
                'Volatility Stability (Std Dev)'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        windows_list = vol_df['window'].to_list()
        
        # Plot 1: Mean volatility with error bars
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['mean_vol'].to_list(),
                mode='lines+markers',
                name='Mean Volatility',
                line=dict(color='blue', width=3),
                marker=dict(size=10),
                error_y=dict(type='data', array=vol_df['std_vol'].to_list(), visible=True),
                text=[f"{v:.1f}%" for v in vol_df['mean_vol'].to_list()],
                textposition='top center'
            ), row=1, col=1
        )
        
        # Plot 2: Min-Max range
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['max_vol'].to_list(),
                mode='lines+markers',
                name='Max Volatility',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['min_vol'].to_list(),
                mode='lines+markers',
                name='Min Volatility',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)'
            ), row=1, col=2
        )
        
        # Plot 3: Quartiles (box plot style)
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['q75_vol'].to_list(),
                mode='lines+markers',
                name='75th Percentile',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6)
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['median_vol'].to_list(),
                mode='lines+markers',
                name='Median',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=windows_list,
                y=vol_df['q25_vol'].to_list(),
                mode='lines+markers',
                name='25th Percentile',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.2)'
            ), row=2, col=1
        )
        
        # Plot 4: Volatility stability (standard deviation)
        fig.add_trace(
            go.Bar(
                x=[f"{w}min" for w in windows_list],
                y=vol_df['std_vol'].to_list(),
                name='Volatility Std Dev',
                marker_color='darkblue',
                text=[f"{v:.1f}%" for v in vol_df['std_vol'].to_list()],
                textposition='outside'
            ), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Comprehensive Volatility Analysis Dashboard",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Window (minutes)", row=1, col=1)
        fig.update_xaxes(title_text="Time Window (minutes)", row=1, col=2)
        fig.update_xaxes(title_text="Time Window (minutes)", row=2, col=1)
        fig.update_xaxes(title_text="Time Window", row=2, col=2)
        
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Std Dev (%)", row=2, col=2)
        
        return fig
    
    def plot_multi_token_volatility_comparison(self, token_data: List[tuple], windows: List[int] = [5, 30, 60, 240]) -> go.Figure:
        """
        Compare volatility patterns across multiple tokens
        token_data: List of (token_name, dataframe) tuples
        """
        if not token_data:
            fig = go.Figure()
            fig.add_annotation(text="No token data provided", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate volatility for each token
        all_volatility_data = []
        
        for token_name, df in token_data:
            if df.is_empty() or len(df) < max(windows):
                continue
                
            df_analysis = df.with_columns([
                df['price'].pct_change().alias('returns')
            ])
            
            for window in windows:
                if len(df_analysis) >= window * 2:
                    rolling_vol = df_analysis.with_columns([
                        df_analysis['returns'].rolling_std(window_size=window, min_periods=max(1, window//2)).alias('rolling_vol')
                    ])['rolling_vol'].drop_nulls() * np.sqrt(525600 / window) * 100
                    
                    if len(rolling_vol) > 0:
                        all_volatility_data.append({
                            'token': token_name,
                            'window': window,
                            'mean_vol': rolling_vol.mean(),
                            'median_vol': rolling_vol.median(),
                            'max_vol': rolling_vol.max(),
                            'std_vol': rolling_vol.std()
                        })
        
        if not all_volatility_data:
            fig = go.Figure()
            fig.add_annotation(text="No valid volatility data calculated", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        vol_comparison_df = pl.DataFrame(all_volatility_data)
        
        # Create subplots for comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Volatility by Window (Top 10 Tokens)',
                'Volatility Distribution by Window',
                'Most vs Least Volatile Tokens',
                'Volatility Stability Ranking'
            ),
            specs=[[{"type": "scatter"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Top 10 tokens by average volatility
        avg_vol_by_token = vol_comparison_df.group_by('token').agg([
            pl.col('mean_vol').mean().alias('overall_avg_vol')
        ]).sort('overall_avg_vol', descending=True).head(10)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, token in enumerate(avg_vol_by_token['token'].to_list()):
            token_data = vol_comparison_df.filter(pl.col('token') == token).sort('window')
            fig.add_trace(
                go.Scatter(
                    x=token_data['window'].to_list(),
                    y=token_data['mean_vol'].to_list(),
                    mode='lines+markers',
                    name=token[:10] + '...' if len(token) > 10 else token,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ), row=1, col=1
            )
        
        # Plot 2: Box plots by window
        for window in windows:
            window_data = vol_comparison_df.filter(pl.col('window') == window)
            fig.add_trace(
                go.Box(
                    y=window_data['mean_vol'].to_list(),
                    name=f"{window}min",
                    boxpoints='outliers'
                ), row=1, col=2
            )
        
        # Plot 3: Most vs Least volatile
        most_volatile = avg_vol_by_token.head(5)
        least_volatile = avg_vol_by_token.tail(5)
        
        fig.add_trace(
            go.Bar(
                x=most_volatile['token'].to_list(),
                y=most_volatile['overall_avg_vol'].to_list(),
                name='Most Volatile',
                marker_color='red',
                text=[f"{v:.1f}%" for v in most_volatile['overall_avg_vol'].to_list()],
                textposition='outside'
            ), row=2, col=1
        )
        
        # Plot 4: Volatility stability (lowest std dev = most stable)
        stability_ranking = vol_comparison_df.group_by('token').agg([
            pl.col('std_vol').mean().alias('avg_std_vol')
        ]).sort('avg_std_vol').head(10)
        
        fig.add_trace(
            go.Bar(
                x=stability_ranking['token'].to_list(),
                y=stability_ranking['avg_std_vol'].to_list(),
                name='Volatility Stability',
                marker_color='green',
                text=[f"{v:.1f}%" for v in stability_ranking['avg_std_vol'].to_list()],
                textposition='outside'
            ), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1400,
            title_text=f"Multi-Token Volatility Comparison - {len(token_data)} Tokens",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Window (minutes)", row=1, col=1)
        fig.update_xaxes(title_text="Time Window", row=1, col=2)
        fig.update_xaxes(title_text="Token", tickangle=45, row=2, col=1)
        fig.update_xaxes(title_text="Token (Most Stable)", tickangle=45, row=2, col=2)
        
        fig.update_yaxes(title_text="Mean Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Avg Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volatility Std Dev (%)", row=2, col=2)
        
        return fig
    
    def plot_microstructure_analysis(self, df: pl.DataFrame, results: Dict, window: int = 60) -> go.Figure:
        """
        Professional market microstructure analysis visualization
        
        Creates a comprehensive 6-panel dashboard showing:
        1. Price with volatility regime coloring
        2. Realized volatility (1h and 4h)
        3. Market efficiency indicators
        4. Liquidity proxies (Amihud measures)
        5. Price impact and velocity
        6. Market quality metrics (autocorrelation, clustering)
        """
        # Get time series data from results
        ts_data = results['time_series_data']
        
        # Create 6-panel subplot
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price & Volatility Regimes', 
                'Volatility Comparison',
                'Efficiency & Impact',
                'Liquidity Measures',
                'Price Dynamics',
                'Quality Indicators'
            ),
            shared_xaxes=True,
            vertical_spacing=0.10,
            horizontal_spacing=0.15,
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Extract time series for plotting
        datetime_vals = ts_data['datetime'].to_numpy()
        price_vals = ts_data['price'].to_numpy()
        returns_vals = ts_data['returns'].to_numpy()
        realized_vol_1h = ts_data['realized_vol_1h'].to_numpy()
        realized_vol_4h = ts_data['realized_vol_4h'].to_numpy()
        price_efficiency = ts_data['price_efficiency_1h'].to_numpy()
        return_autocorr = ts_data['return_autocorr'].to_numpy()
        vol_clustering = ts_data['vol_clustering'].to_numpy()
        amihud_vals = ts_data['amihud_illiquidity'].to_numpy()
        price_velocity = ts_data['price_velocity'].to_numpy()
        
        # Panel 1: Price with volatility regime coloring
        # Color code based on realized volatility percentiles
        vol_75th = np.nanpercentile(realized_vol_1h, 75)
        vol_25th = np.nanpercentile(realized_vol_1h, 25)
        
        # Create color array based on volatility regime
        colors = np.where(realized_vol_1h > vol_75th, 'red',
                 np.where(realized_vol_1h < vol_25th, 'green', 'blue'))
        
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=price_vals,
                mode='lines', name='Price',
                line=dict(color='darkblue', width=1.5),
                hovertemplate='<b>Price:</b> %{y:.6f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add volatility as secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=realized_vol_1h * 100,
                mode='lines', name='Vol 1h',
                line=dict(color='red', width=1, dash='dot'),
                yaxis='y2',
                hovertemplate='<b>Volatility:</b> %{y:.1f}%<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Panel 2: Realized Volatility Comparison
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=realized_vol_1h * 100,
                mode='lines', name='Vol 1h',
                line=dict(color='red', width=2),
                hovertemplate='<b>1h Vol:</b> %{y:.1f}%<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=realized_vol_4h * 100,
                mode='lines', name='Vol 4h',
                line=dict(color='darkred', width=2, dash='dash'),
                hovertemplate='<b>4h Vol:</b> %{y:.1f}%<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Add volatility ratio on secondary axis
        vol_ratio = np.where(realized_vol_4h != 0, realized_vol_1h / realized_vol_4h, 1)
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=vol_ratio,
                mode='lines', name='Ratio',
                line=dict(color='purple', width=1),
                yaxis='y4',
                hovertemplate='<b>Vol Ratio:</b> %{y:.2f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=1, col=2, secondary_y=True
        )
        
        # Panel 3: Market Efficiency & Price Impact
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=price_efficiency,
                mode='lines', name='Efficiency',
                line=dict(color='green', width=2),
                hovertemplate='<b>Efficiency:</b> %{y:.3f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add Kyle's lambda as reference line if available
        if not np.isnan(results['kyle_lambda']):
            fig.add_hline(
                y=results['kyle_lambda'], 
                line_dash="dash", line_color="orange",
                annotation_text=f"Kyle's λ: {results['kyle_lambda']:.2e}",
                row=2, col=1
            )
        
        # Add price impact proxy on secondary axis
        price_impact_proxy = np.abs(returns_vals) * 1000  # Scale for visibility
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=price_impact_proxy,
                mode='lines', name='Impact',
                line=dict(color='orange', width=1, dash='dot'),
                yaxis='y6',
                hovertemplate='<b>Impact Proxy:</b> %{y:.3f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=2, col=1, secondary_y=True
        )
        
        # Panel 4: Liquidity Indicators (Amihud)
        # Remove extreme outliers for better visualization
        amihud_clean = np.where(np.abs(amihud_vals) > np.nanpercentile(np.abs(amihud_vals), 95), 
                               np.nan, amihud_vals)
        
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=amihud_clean,
                mode='lines', name='Amihud',
                line=dict(color='brown', width=2),
                hovertemplate='<b>Amihud:</b> %{y:.3f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=2, col=2
        )
        
        # Add average as reference
        avg_amihud = np.nanmean(amihud_clean)
        if not np.isnan(avg_amihud):
            fig.add_hline(
                y=avg_amihud, 
                line_dash="dash", line_color="brown",
                annotation_text=f"Avg: {avg_amihud:.3f}",
                row=2, col=2
            )
        
        # Panel 5: Price Velocity & Market Impact
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=price_velocity,
                mode='lines', name='Velocity',
                line=dict(color='navy', width=2),
                hovertemplate='<b>Velocity:</b> %{y:.6f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=3, col=1
        )
        
        # Add rolling average of price velocity
        velocity_ma = ts_data.with_columns([
            ts_data['price_velocity'].rolling_mean(30, min_periods=15).alias('velocity_ma')
        ])['velocity_ma'].to_numpy()
        
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=velocity_ma,
                mode='lines', name='Vel MA',
                line=dict(color='lightblue', width=1, dash='dash'),
                yaxis='y10',
                hovertemplate='<b>Velocity MA:</b> %{y:.6f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=3, col=1, secondary_y=True
        )
        
        # Panel 6: Market Quality Metrics
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=return_autocorr,
                mode='lines', name='Autocorr',
                line=dict(color='purple', width=2),
                hovertemplate='<b>Autocorr:</b> %{y:.3f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=datetime_vals, y=vol_clustering,
                mode='lines', name='Clustering',
                line=dict(color='magenta', width=2, dash='dash'),
                yaxis='y12',
                hovertemplate='<b>Vol Clustering:</b> %{y:.3f}<br><b>Time:</b> %{x}<extra></extra>',
                showlegend=True
            ),
            row=3, col=2, secondary_y=True
        )
        
        # Add zero reference lines where appropriate
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5, row=3, col=2)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", line_width=0.5, row=1, col=2, secondary_y=True)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🔬 Market Microstructure Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=2)
        
        # Update y-axes with proper titles
        fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text="Volatility %", row=1, col=1, secondary_y=True, title_font=dict(size=12))
        fig.update_yaxes(title_text="Volatility %", row=1, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text="Vol Ratio", row=1, col=2, secondary_y=True, title_font=dict(size=12))
        fig.update_yaxes(title_text="Efficiency", row=2, col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text="Impact Proxy", row=2, col=1, secondary_y=True, title_font=dict(size=12))
        fig.update_yaxes(title_text="Amihud", row=2, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text="Price Velocity", row=3, col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text="Velocity MA", row=3, col=1, secondary_y=True, title_font=dict(size=12))
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text="Vol Clustering", row=3, col=2, secondary_y=True, title_font=dict(size=12))
        
        # Update x-axes with time labels and formatting for all subplots
        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_xaxes(
                    title_text="Time", 
                    title_font=dict(size=12),
                    showgrid=False,
                    showticklabels=True,
                    tickfont=dict(size=10),
                    row=row, col=col
                )
        
        # Update y-axes formatting for all subplots
        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=True,
                    tickfont=dict(size=10),
                    row=row, col=col
                )
                # Also update secondary y-axes if they exist
                fig.update_yaxes(
                    showgrid=False,
                    showticklabels=True,
                    tickfont=dict(size=10),
                    row=row, col=col, secondary_y=True
                )
        
        return fig
    
    def plot_microstructure_summary_dashboard(self, results: Dict) -> go.Figure:
        """
        Create a summary dashboard showing key microstructure metrics as gauges and indicators
        """
        # Extract key metrics with shorter names
        metrics = {
            'Vol 1h (%)': results.get('avg_realized_volatility_1h', 0) * 100,
            'Vol 4h (%)': results.get('avg_realized_volatility_4h', 0) * 100,
            'Spread (bps)': results.get('bid_ask_spread_estimate', 0) * 10000,  # in basis points
            'Kyle λ (×1000)': abs(results.get('kyle_lambda', 0)) * 1000 if not np.isnan(results.get('kyle_lambda', np.nan)) else 0,
            'Efficiency': results.get('avg_price_efficiency', 0),
            'Clustering': results.get('volatility_clustering', 0)
        }
        
        # Create gauge chart with no subplot titles (we'll use gauge titles instead)
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Define gauge configurations
        gauge_configs = [
            {'max': 200, 'color': 'red', 'suffix': ''},       # Vol 1h (already includes %)
            {'max': 150, 'color': 'darkred', 'suffix': ''},   # Vol 4h (already includes %)  
            {'max': 100, 'color': 'orange', 'suffix': ''},    # Spread (already includes bps)
            {'max': 10, 'color': 'purple', 'suffix': ''},     # Kyle Lambda (already scaled)
            {'max': 2, 'color': 'green', 'suffix': ''},       # Price Efficiency
            {'max': 1, 'color': 'magenta', 'suffix': ''}      # Vol Clustering
        ]
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            config = gauge_configs[i]
            row, col = positions[i]
            
            fig.add_trace(
                go.Indicator(
                    mode = "gauge+number",
                    value = value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': metric_name, 'font': {'size': 14}},
                    gauge = {
                        'axis': {'range': [None, config['max']], 'tickfont': {'size': 10}},
                        'bar': {'color': config['color']},
                        'steps': [
                            {'range': [0, config['max']*0.3], 'color': "lightgray"},
                            {'range': [config['max']*0.3, config['max']*0.7], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': config['max']*0.9
                        }
                    },
                    number = {'suffix': config['suffix'], 'font': {'size': 16}}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=650,
            title_text="🎯 Microstructure Metrics Summary",
            title_x=0.5,
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        return fig
    
    def plot_price_distribution_evolution(self, df: pl.DataFrame, 
                                        n_periods: int = 6) -> go.Figure:
        """
        Plot comprehensive price distribution evolution analysis
        
        Features:
        - Pure Polars implementation for performance
        - Statistical distribution testing (normality, skewness, kurtosis)
        - Multiple distribution overlays (normal, t-distribution, skewed normal)
        - Evolution metrics tracking
        - Q-Q plots for distribution assessment
        - Distribution comparison statistics
        """
        if df.height < n_periods * 10:  # Need minimum data per period
            raise ValueError(f"Insufficient data: need at least {n_periods * 10} points")
        
        # Calculate period boundaries
        period_size = df.height // n_periods
        
        # Calculate all period statistics efficiently with Polars
        period_stats = []
        period_returns_data = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else df.height
            
            # Extract period data
            period_df = df.slice(start_idx, end_idx - start_idx)
            
            # Calculate returns for this period using Polars
            period_returns_df = period_df.select([
                pl.col('price').pct_change().drop_nulls().alias('returns')
            ])
            period_returns = period_returns_df['returns'] * 100
            
            if period_returns.len() > 5:  # Need minimum returns for stats
                # Calculate comprehensive statistics using Polars
                stats = period_returns_df.select([
                    pl.mean('returns').alias('mean'),
                    pl.std('returns').alias('std'),
                    pl.median('returns').alias('median'),
                    pl.min('returns').alias('min'),
                    pl.max('returns').alias('max'),
                    pl.quantile('returns', 0.25).alias('q25'),
                    pl.quantile('returns', 0.75).alias('q75'),
                    pl.quantile('returns', 0.05).alias('q05'),
                    pl.quantile('returns', 0.95).alias('q95'),
                    pl.len().alias('count')
                ]).row(0, named=True)
                
                # Apply the percentage conversion to the stats
                for key in ['mean', 'std', 'median', 'min', 'max', 'q25', 'q75', 'q05', 'q95']:
                    if key in stats:
                        stats[key] = stats[key] * 100
                
                # Calculate skewness and kurtosis manually (Polars doesn't have these)
                returns_array = period_returns.to_numpy()
                mean_val = stats['mean']
                std_val = stats['std']
                
                if std_val > 1e-10:  # Avoid division by zero
                    # Skewness calculation
                    skewness = np.mean(((returns_array - mean_val) / std_val) ** 3)
                    # Excess kurtosis calculation  
                    kurtosis = np.mean(((returns_array - mean_val) / std_val) ** 4) - 3
                    
                    # Jarque-Bera test statistic for normality
                    n = len(returns_array)
                    jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)
                    
                    # Shapiro-Wilk test p-value (approximate)
                    from scipy import stats as scipy_stats
                    if n >= 3 and n <= 5000:
                        _, shapiro_p = scipy_stats.shapiro(returns_array)
                    else:
                        shapiro_p = 0.0
                else:
                    skewness = kurtosis = jb_stat = shapiro_p = 0
                
                period_stats.append({
                    'period': i + 1,
                    'start_time': period_df['datetime'].first(),
                    'end_time': period_df['datetime'].last(),
                    **stats,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'jb_statistic': jb_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05 if shapiro_p > 0 else False
                })
                
                period_returns_data.append((i + 1, returns_array))
        
        if not period_stats:
            raise ValueError("No valid periods found with sufficient data")
        
        # Create comprehensive visualization with better spacing
        n_cols = min(3, n_periods) if n_periods <= 6 else 3  # Max 3 columns for readability
        n_rows = 3 if n_periods <= 6 else 4  # Add extra row if needed
        
        # Create shorter subplot titles to avoid overlap
        subplot_titles = [f"P{i+1}" for i in range(n_periods)]
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.12,  # Increased spacing
            horizontal_spacing=0.10,  # Increased spacing
            row_heights=[0.35, 0.25, 0.25, 0.15] if n_rows == 4 else [0.4, 0.3, 0.3]
        )
        
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink', 'gray']
        
        for idx, (period_num, returns_array) in enumerate(period_returns_data):
            if idx >= n_periods:
                break
                
            # Calculate subplot position with better layout
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            color = colors[idx % len(colors)]
            
            # All plots go in their respective positions (no row-specific logic needed)
            plot_type = ((idx // n_cols) % 3)  # 0=histogram, 1=qq, 2=box
            
            if plot_type == 0:  # Distribution histogram with overlays
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=returns_array, 
                        nbinsx=25, 
                        name=f"P{period_num} Returns",
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Normal distribution overlay
                stats = period_stats[idx]
                x_range = np.linspace(returns_array.min(), returns_array.max(), 100)
                normal_dist = scipy_stats.norm.pdf(x_range, stats['mean'], stats['std'])
                normal_dist = normal_dist * len(returns_array) * (returns_array.max() - returns_array.min()) / 25
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range, 
                        y=normal_dist, 
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f"P{period_num} Normal",
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add distribution fit (t-distribution if heavy tails)
                if abs(stats['kurtosis']) > 1:  # Heavy tails detected
                    try:
                        # Fit t-distribution
                        t_params = scipy_stats.t.fit(returns_array)
                        t_dist = scipy_stats.t.pdf(x_range, *t_params)
                        t_dist = t_dist * len(returns_array) * (returns_array.max() - returns_array.min()) / 25
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range, 
                                y=t_dist, 
                                mode='lines',
                                line=dict(color='green', width=2, dash='dot'),
                                name=f"P{period_num} t-dist",
                                showlegend=False
                            ),
                            row=row, col=col
                        )
                    except:
                        pass  # Skip if fitting fails
            
            elif plot_type == 1:  # Q-Q plot against normal distribution
                # Generate Q-Q plot data
                sorted_data = np.sort(returns_array)
                n = len(sorted_data)
                theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_data,
                        mode='markers',
                        marker=dict(color=color, size=4),
                        name=f"P{period_num} Q-Q",
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add perfect normal line
                line_range = [min(theoretical_quantiles.min(), sorted_data.min()),
                             max(theoretical_quantiles.max(), sorted_data.max())]
                fig.add_trace(
                    go.Scatter(
                        x=line_range,
                        y=line_range,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f"P{period_num} Perfect Normal",
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            elif plot_type == 2:  # Box plot with outlier detection
                fig.add_trace(
                    go.Box(
                        y=returns_array,
                        name=f"Period {period_num}",
                        marker_color=color,
                        showlegend=False,
                        boxpoints='outliers'
                    ),
                    row=row, col=col
                )
        
        # Update layout with better spacing and formatting
        fig.update_layout(
            height=1000,  # Increased height for better readability
            title_text="Price Distribution Evolution Analysis",
            title_font_size=16,
            showlegend=False,
            margin=dict(t=80, b=60, l=60, r=60)  # Better margins
        )
        
        # Update all axes with consistent, smaller fonts to avoid overlap
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                # Determine plot type based on position
                plot_idx = (row - 1) * n_cols + (col - 1)
                if plot_idx < len(period_returns_data):
                    plot_type = ((plot_idx // n_cols) % 3)
                    
                    if plot_type == 0:  # Histogram
                        fig.update_xaxes(title_text="Returns (%)", row=row, col=col, 
                                       title_font=dict(size=9), tickfont=dict(size=8))
                        fig.update_yaxes(title_text="Freq", row=row, col=col, 
                                       title_font=dict(size=9), tickfont=dict(size=8))
                    elif plot_type == 1:  # Q-Q plot
                        fig.update_xaxes(title_text="Theoretical Q", row=row, col=col, 
                                       title_font=dict(size=9), tickfont=dict(size=8))
                        fig.update_yaxes(title_text="Sample Q", row=row, col=col, 
                                       title_font=dict(size=9), tickfont=dict(size=8))
                    elif plot_type == 2:  # Box plot
                        fig.update_xaxes(title_text="", row=row, col=col, 
                                       tickfont=dict(size=8))
                        fig.update_yaxes(title_text="Returns (%)", row=row, col=col, 
                                       title_font=dict(size=9), tickfont=dict(size=8))
        
        return fig, period_stats
    
    def plot_distribution_evolution_summary(self, period_stats: List[Dict]) -> go.Figure:
        """
        Create a summary dashboard showing how distribution characteristics evolve over time
        """
        if not period_stats:
            raise ValueError("No period statistics provided")
        
        # Convert to DataFrame for easier manipulation
        stats_df = pl.DataFrame(period_stats)
        
        # Create 2x3 dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Mean & Volatility Evolution',
                'Distribution Shape (Skewness & Kurtosis)', 
                'Normality Test Results',
                'Return Range Evolution',
                'Statistical Significance',
                'Distribution Quality Score'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        periods = stats_df['period'].to_list()
        
        # Row 1, Col 1: Mean & Volatility Evolution
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['mean'].to_list(),
                mode='lines+markers',
                name='Mean Return',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['std'].to_list(),
                mode='lines+markers',
                name='Volatility',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Row 1, Col 2: Distribution Shape
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['skewness'].to_list(),
                mode='lines+markers',
                name='Skewness',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['kurtosis'].to_list(),
                mode='lines+markers',
                name='Excess Kurtosis',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=1, col=2, secondary_y=True
        )
        
        # Add reference lines for normal distribution
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2, secondary_y=False)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2, secondary_y=True)
        
        # Row 1, Col 3: Normality Test Results
        normality_scores = [1 if stat['is_normal'] else 0 for stat in period_stats]
        shapiro_p_values = stats_df['shapiro_p_value'].to_list()
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=normality_scores,
                name='Normal Distribution',
                marker_color=['green' if x == 1 else 'red' for x in normality_scores],
                opacity=0.7
            ),
            row=1, col=3
        )
        
        # Row 2, Col 1: Return Range Evolution
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['max'].to_list(),
                mode='lines+markers',
                name='Max Return',
                line=dict(color='lightgreen', width=2),
                fill=None
            ),
            row=2, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=stats_df['min'].to_list(),
                mode='lines+markers',
                name='Min Return',
                line=dict(color='lightcoral', width=2),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)'
            ),
            row=2, col=1, secondary_y=False
        )
        
        # Add IQR on secondary axis
        iqr = [q75 - q25 for q75, q25 in zip(stats_df['q75'].to_list(), stats_df['q25'].to_list())]
        fig.add_trace(
            go.Scatter(
                x=periods, 
                y=iqr,
                mode='lines+markers',
                name='IQR',
                line=dict(color='purple', width=3, dash='dot'),
                marker=dict(size=6)
            ),
            row=2, col=1, secondary_y=True
        )
        
        # Row 2, Col 2: Statistical Significance (Jarque-Bera)
        jb_stats = stats_df['jb_statistic'].to_list()
        # JB critical value at 5% significance level with 2 degrees of freedom
        jb_critical = 5.99
        
        colors = ['green' if jb < jb_critical else 'red' for jb in jb_stats]
        fig.add_trace(
            go.Bar(
                x=periods,
                y=jb_stats,
                name='Jarque-Bera Statistic',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Add critical value line
        fig.add_hline(y=jb_critical, line_dash="dash", line_color="red", 
                     annotation_text="Critical Value (5%)", row=2, col=2)
        
        # Row 2, Col 3: Distribution Quality Score
        # Calculate composite quality score
        quality_scores = []
        for stat in period_stats:
            score = 0
            # Normality bonus
            if stat['is_normal']:
                score += 30
            # Low skewness bonus (closer to 0 is better)
            score += max(0, 20 - abs(stat['skewness']) * 10)
            # Reasonable kurtosis bonus (not too heavy tails)
            score += max(0, 20 - abs(stat['kurtosis']) * 5)
            # Sample size bonus
            score += min(30, stat['count'] / 10)
            
            quality_scores.append(min(100, score))
        
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=quality_scores,
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='purple', width=4),
                marker=dict(size=10, color=quality_scores, 
                          colorscale='RdYlGn', showscale=False),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ),
            row=2, col=3
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Period", row=1, col=1)
        fig.update_yaxes(title_text="Mean Return (%)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Period", row=1, col=2)
        fig.update_yaxes(title_text="Skewness", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Excess Kurtosis", row=1, col=2, secondary_y=True)
        
        fig.update_xaxes(title_text="Period", row=1, col=3)
        fig.update_yaxes(title_text="Is Normal", row=1, col=3)
        
        fig.update_xaxes(title_text="Period", row=2, col=1)
        fig.update_yaxes(title_text="Return Range (%)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="IQR (%)", row=2, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Period", row=2, col=2)
        fig.update_yaxes(title_text="JB Statistic", row=2, col=2)
        
        fig.update_xaxes(title_text="Period", row=2, col=3)
        fig.update_yaxes(title_text="Quality Score (0-100)", row=2, col=3)
        
        fig.update_layout(
            height=850,
            title_text="Distribution Evolution Summary Dashboard",
            title_font_size=16,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.15,  # Moved legend to bottom to avoid title overlap
                xanchor="center", 
                x=0.5,
                font=dict(size=10)  # Smaller legend font
            ),
            margin=dict(t=60, b=120, l=60, r=60)  # More bottom margin for legend
        )
        
        return fig
    
    def plot_optimal_holding_period(self, df: pl.DataFrame,
                                   max_period: int = 240,
                                   step: int = 5) -> go.Figure:
        """
        Analyze optimal holding periods based on return distributions
        """
        holding_periods = range(step, max_period + 1, step)
        results = []
        
        for period in holding_periods:
            # Calculate returns for this holding period
            period_returns = []
            
            for i in range(0, len(df) - period, period // 2):  # Overlap by 50%
                ret = (df['price'].to_numpy()[i + period] / df['price'].to_numpy()[i] - 1) * 100
                period_returns.append(ret)
            
            if period_returns:
                results.append({
                    'period': period,
                    'mean_return': np.mean(period_returns),
                    'median_return': np.median(period_returns),
                    'std_return': np.std(period_returns),
                    'sharpe': np.mean(period_returns) / (np.std(period_returns) + 1e-10),
                    'win_rate': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100,
                    'max_return': max(period_returns),
                    'min_return': min(period_returns)
                })
        
        results_df = pl.DataFrame(results)
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Return vs Risk', 'Win Rate by Holding Period',
                          'Risk-Adjusted Returns', 'Return Range'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Mean return and std
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['mean_return'],
                      mode='lines+markers', name='Mean Return %',
                      line=dict(color='green', width=2)),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['std_return'],
                      mode='lines+markers', name='Std Dev %',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1, secondary_y=True
        )
        
        # Win rate
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['win_rate'],
                      mode='lines+markers', name='Win Rate %',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # Sharpe ratio
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['sharpe'],
                      mode='lines+markers', name='Sharpe Ratio',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Return range
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['max_return'],
                      mode='lines', name='Max Return',
                      line=dict(color='lightgreen', width=1)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=results_df['period'], y=results_df['min_return'],
                      mode='lines', name='Min Return',
                      line=dict(color='lightcoral', width=1),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.2)'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Holding Period (minutes)")
        fig.update_layout(height=800, title_text="Optimal Holding Period Analysis")
        
        return fig
    
    def plot_regime_analysis(self, df: pl.DataFrame, lookback: int = 60) -> go.Figure:
        """
        Plot market regime analysis (trending, ranging, volatile)
        """
        df_regime = df.copy()
        
        # Calculate indicators
        df_regime['returns'] = df_regime['price'].pct_change()
        df_regime['sma_short'] = df_regime['price'].rolling(lookback // 4).mean()
        df_regime['sma_long'] = df_regime['price'].rolling(lookback).mean()
        df_regime['volatility'] = df_regime['returns'].rolling(lookback // 2).std()
        
        # Define regimes
        vol_threshold = df_regime['volatility'].rolling(lookback * 2).mean()
        
        conditions = [
            (df_regime['sma_short'] > df_regime['sma_long']) & (df_regime['volatility'] < vol_threshold),
            (df_regime['sma_short'] < df_regime['sma_long']) & (df_regime['volatility'] < vol_threshold),
            (df_regime['volatility'] > vol_threshold)
        ]
        
        choices = [1, -1, 0]  # 1=uptrend, -1=downtrend, 0=high volatility
        df_regime['regime'] = pl.Series(np.select(conditions, choices, default=0.5)).to_numpy()  # 0.5=ranging
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price with Regime Coloring', 'Volatility', 'Regime Indicator'),
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price with regime coloring
        colors = {1: 'green', -1: 'red', 0: 'orange', 0.5: 'gray'}
        
        for regime, color in colors.items():
            mask = df_regime['regime'] == regime
            if mask.any():
                fig.add_trace(
                    go.Scatter(x=df_regime.loc[mask, 'datetime'].to_numpy(),
                             y=df_regime.loc[mask, 'price'].to_numpy(),
                             mode='markers', marker=dict(color=color, size=3),
                             name={1: 'Uptrend', -1: 'Downtrend', 
                                  0: 'High Vol', 0.5: 'Ranging'}[regime]),
                    row=1, col=1
                )
        
        # Volatility
        fig.add_trace(
            go.Scatter(x=df_regime['datetime'].to_numpy(), y=df_regime['volatility'] * 100,
                      mode='lines', name='Volatility %',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Regime indicator
        fig.add_trace(
            go.Scatter(x=df_regime['datetime'].to_numpy(), y=df_regime['regime'],
                      mode='lines', name='Regime',
                      line=dict(color='black', width=1)),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text="Market Regime Analysis")
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig
    
    def plot_correlation_dynamics(self, dfs: List[pl.DataFrame], 
                                 names: List[str],
                                 window: int = 60) -> go.Figure:
        """
        Plot dynamic correlations between multiple tokens
        """
        if len(dfs) < 2:
            raise ValueError("Need at least 2 tokens for correlation analysis")
        
        # Calculate returns for all tokens
        returns_dict = {}
        for i, (df, name) in enumerate(zip(dfs, names)):
            returns_dict[name] = df['price'].pct_change()
        
        returns_df = pl.DataFrame(returns_dict)
        
        # Calculate rolling correlations
        corr_data = []
        timestamps = dfs[0]['datetime'].to_numpy()
        
        for i in range(window, len(returns_df)):
            window_data = returns_df.iloc[i-window:i]
            corr_matrix = window_data.corr()
            
            # Extract unique correlations
            for j in range(len(names)):
                for k in range(j+1, len(names)):
                    corr_data.append({
                        'datetime': timestamps[i],
                        'pair': f"{names[j]}-{names[k]}",
                        'correlation': corr_matrix.iloc[j, k]
                    })
        
        corr_df = pl.DataFrame(corr_data)
        
        # Create plot
        fig = go.Figure()
        
        for pair in corr_df['pair'].unique():
            pair_data = corr_df[corr_df['pair'] == pair]
            fig.add_trace(
                go.Scatter(x=pair_data['datetime'], y=pair_data['correlation'],
                          mode='lines', name=pair, line=dict(width=2))
            )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", 
                      annotation_text="Strong Positive")
        fig.add_hline(y=-0.7, line_dash="dot", line_color="red",
                      annotation_text="Strong Negative")
        
        fig.update_layout(
            title=f"Dynamic Correlations ({window}-minute rolling window)",
            xaxis_title="Time",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1, 1]),
            height=600
        )
        
        return fig

    def aggregate_entry_exit_matrices(self, token_data_list, entry_windows, exit_windows):
        """
        Calculate aggregated entry/exit matrix across multiple tokens
        """
        all_returns = defaultdict(list)
        
        for token_name, df in token_data_list:
            for entry_window in entry_windows:
                for exit_window in exit_windows:
                    # Calculate returns for this combination
                    returns = self.calculate_window_returns(df, entry_window, exit_window)
                    all_returns[(entry_window, exit_window)].extend(returns)
        
        # Build the matrix as a list of lists
        matrix_data = []
        confidence_data = []
        for entry in entry_windows:
            row = []
            conf_row = []
            for exit in exit_windows:
                returns = all_returns[(entry, exit)]
                row.append(np.mean(returns) if returns else 0.0)
                conf_row.append(1.96 * (np.std(returns) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0)
            matrix_data.append(row)
            confidence_data.append(conf_row)
        # Create polars DataFrames by transposing the data
        matrix = pl.DataFrame({str(col): [row[i] for row in matrix_data] for i, col in enumerate(exit_windows)})
        confidence_matrix = pl.DataFrame({str(col): [row[i] for row in confidence_data] for i, col in enumerate(exit_windows)})
        return matrix, confidence_matrix

    def plot_multi_token_entry_exit_matrix(self, aggregated_matrix, confidence_matrix, n_tokens, entry_windows):
        """
        Plot aggregated entry/exit matrix with confidence intervals
        """
        # Ensure numeric values for heatmap and text
        z_matrix = aggregated_matrix.to_numpy().astype(float)
        z_conf = confidence_matrix.to_numpy().astype(float)
        text_matrix = np.round(z_matrix, 2)
        text_conf = np.round(z_conf, 2)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Returns (%)', 'Statistical Confidence (95% CI)'),
            horizontal_spacing=0.15,  # Increase spacing between subplots
            column_widths=[0.45, 0.45]  # Make subplots slightly smaller to accommodate colorbars
        )
        
        # Main heatmap
        fig.add_trace(
            go.Heatmap(
                z=z_matrix,
                x=[f"{w}min" for w in list(aggregated_matrix.columns)],
                y=[f"{w}min" for w in entry_windows],
                colorscale='RdBu',
                zmid=0,
                text=text_matrix,
                texttemplate='%{text}%',
                textfont={"size": 10},
                colorbar=dict(
                    title="Avg Return %", 
                    x=0.42,  # Move colorbar closer to first plot
                    len=0.8,  # Make colorbar shorter
                    thickness=15
                ),
                hovertemplate='Entry: %{y}<br>Exit: %{x}<br>Avg Return: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Confidence interval heatmap  
        fig.add_trace(
            go.Heatmap(
                z=z_conf,
                x=[f"{w}min" for w in list(confidence_matrix.columns)],
                y=[f"{w}min" for w in entry_windows],
                colorscale='Viridis',
                text=text_conf,
                texttemplate='±%{text}%',
                textfont={"size": 10},
                colorbar=dict(
                    title="95% CI", 
                    x=1.05,  # Move colorbar further right
                    len=0.8,  # Make colorbar shorter
                    thickness=15
                ),
                hovertemplate='Entry: %{y}<br>Exit: %{x}<br>95% CI: ±%{z:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Exit Window", row=1, col=1)
        fig.update_xaxes(title_text="Exit Window", row=1, col=2)
        fig.update_yaxes(title_text="Entry Window", row=1, col=1)
        fig.update_yaxes(title_text="Entry Window", row=1, col=2)
        
        fig.update_layout(
            title=f"Multi-Token Entry/Exit Analysis (n={n_tokens} tokens)",
            height=600,
            width=1200,  # Increase total width to accommodate better spacing
            showlegend=False
        )
        return fig

    def plot_trade_timing_heatmap(self, dfs: List[pl.DataFrame], max_entry_minute: int = 240, max_exit_lag: int = 60) -> go.Figure:
        """
        Plot a heatmap showing the average return for each (entry minute, exit lag) pair across all tokens.
        Args:
            dfs: List of pandas DataFrames, one per token, each with 'price' and 'datetime'.
            max_entry_minute: Only consider entry times up to this many minutes after launch (for clarity/performance).
            max_exit_lag: Maximum exit lag (in minutes) to consider.
        Returns:
            Plotly Figure (heatmap)
        """
        entry_minutes = np.arange(0, max_entry_minute)
        exit_lags = np.arange(1, max_exit_lag + 1)
        heatmap = np.full((len(exit_lags), len(entry_minutes)), np.nan)
        counts = np.zeros_like(heatmap)

        # For each token, accumulate returns for each (entry, lag)
        for df in dfs:
            prices = df['price'].to_numpy()
            n = min(len(prices), max_entry_minute + max_exit_lag)
            for entry_idx, entry_minute in enumerate(entry_minutes):
                if entry_minute >= n - max_exit_lag:
                    continue
                entry_price = prices[entry_minute]
                if np.isnan(entry_price) or entry_price == 0:
                    continue
                for lag_idx, lag in enumerate(exit_lags):
                    exit_idx = entry_minute + lag
                    if exit_idx >= n:
                        continue
                    exit_price = prices[exit_idx]
                    if np.isnan(exit_price):
                        continue
                    trade_return = (exit_price / entry_price - 1) * 100
                    if np.isfinite(trade_return):
                        if np.isnan(heatmap[lag_idx, entry_idx]):
                            heatmap[lag_idx, entry_idx] = 0
                        heatmap[lag_idx, entry_idx] += trade_return
                        counts[lag_idx, entry_idx] += 1

        # Average
        with np.errstate(invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)

        # Plot
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            x=entry_minutes,
            y=exit_lags,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Avg Return %"),
            hoverongaps=False
        ))
        fig.update_layout(
            title="Trade Timing Heatmap: Avg Return by Entry Minute and Exit Lag",
            xaxis_title="Entry Minute (since launch)",
            yaxis_title="Exit Lag (minutes after entry)",
            height=600,
            width=1000
        )
        return fig

    def calculate_window_returns(self, df: pl.DataFrame, entry_window: int, exit_window: int, momentum_threshold: float = 0.0) -> list:
        """
        Calculate non-overlapping returns for a given entry/exit window using momentum logic (like single-token view).
        Only enter if momentum over entry_window is above threshold. Skip ahead by exit_window after a trade.
        """
        returns = []
        i = entry_window
        while i < len(df) - exit_window:
            entry_momentum = (df['price'].to_numpy()[i] / df['price'].to_numpy()[i-entry_window] - 1)
            if entry_momentum > momentum_threshold:
                entry_price = df['price'].to_numpy()[i]
                exit_price = df['price'].to_numpy()[i + exit_window]
                trade_return = (exit_price / entry_price - 1) * 100
                returns.append(trade_return)
                i += exit_window  # Skip ahead to avoid overlapping
            else:
                i += 1
        return returns

    def plot_entry_exit_moment_matrix_optimized(self, dfs: list, max_entry_minute: int = 240, max_exit_minute: int = 240) -> go.Figure:
        """
        FULLY OPTIMIZED: Plot entry/exit moment matrix using pure Polars cross joins
        Extremely fast vectorized operations - no nested loops at all
        """
        all_results = []
        
        for df_idx, df in enumerate(dfs):
            if not isinstance(df, pl.DataFrame):
                continue
                
            # Ensure we have enough data and limit to max_exit_minute
            n_rows = min(len(df), max_exit_minute)
            if n_rows < 2:
                continue
                
            # Create indexed price data
            price_data = df.head(n_rows).with_row_index('minute').select(['minute', 'price'])
            
            # Create entry and exit minute ranges
            entry_minutes = pl.DataFrame({'entry_minute': range(min(max_entry_minute, n_rows - 1))})
            exit_minutes = pl.DataFrame({'exit_minute': range(1, n_rows)})  # Start from 1 to ensure exit > entry
            
            # Cross join to get all combinations
            combinations = entry_minutes.join(exit_minutes, how='cross')
            
            # Filter to only valid combinations (exit > entry)
            valid_combinations = combinations.filter(pl.col('exit_minute') > pl.col('entry_minute'))
            
            if valid_combinations.is_empty():
                continue
            
            # Join with price data to get entry prices
            with_entry_prices = valid_combinations.join(
                price_data.rename({'minute': 'entry_minute', 'price': 'entry_price'}),
                on='entry_minute',
                how='left'
            )
            
            # Join with price data to get exit prices
            with_both_prices = with_entry_prices.join(
                price_data.rename({'minute': 'exit_minute', 'price': 'exit_price'}),
                on='exit_minute',
                how='left'
            )
            
            # Calculate returns using pure Polars vectorized operations
            returns_df = with_both_prices.filter(
                (pl.col('entry_price').is_not_null()) & 
                (pl.col('exit_price').is_not_null()) &
                (pl.col('entry_price') > 0)
            ).with_columns([
                ((pl.col('exit_price') / pl.col('entry_price') - 1) * 100).alias('return_pct'),
                pl.lit(df_idx).alias('token_id')
            ])
            
            all_results.append(returns_df)
        
        if not all_results:
            fig = go.Figure()
            fig.add_annotation(text="No valid data for moment matrix", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Combine all results using Polars concat
        combined_results = pl.concat(all_results)
        
        # Aggregate by entry/exit minute pairs using Polars group_by
        aggregated = combined_results.group_by(['entry_minute', 'exit_minute']).agg([
            pl.col('return_pct').mean().alias('avg_return'),
            pl.col('return_pct').count().alias('trade_count'),
            pl.col('return_pct').std().alias('return_std')
        ]).sort(['entry_minute', 'exit_minute'])
        
        # Create matrix for heatmap
        entry_range = range(max_entry_minute)
        exit_range = range(max_exit_minute)
        heatmap = np.full((max_entry_minute, max_exit_minute), np.nan)
        
        # Fill matrix with aggregated data - this is the only remaining loop
        for row in aggregated.iter_rows():
            entry_idx, exit_idx, avg_return, trade_count, return_std = row
            if 0 <= entry_idx < max_entry_minute and 0 <= exit_idx < max_exit_minute:
                heatmap[entry_idx, exit_idx] = avg_return
        
        # Mask lower triangle (entry >= exit) - this ensures we only show valid trades
        mask = np.tri(max_entry_minute, max_exit_minute, k=0, dtype=bool)
        heatmap = np.where(mask, np.nan, heatmap)
        
        # Create enhanced heatmap with better formatting
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            x=list(exit_range),
            y=list(entry_range),
            colorscale='RdBu',
            zmid=0,
                         colorbar=dict(title="Avg Return %"),
            hoverongaps=False,
            hovertemplate='Entry: %{y} min<br>Exit: %{x} min<br>Avg Return: %{z:.2f}%<extra></extra>',
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Entry/Exit Moment Matrix (Polars Optimized)<br>Avg Return % by Entry/Exit Minute - {len(dfs)} Tokens",
            xaxis_title="Exit Minute",
            yaxis_title="Entry Minute",
            height=700,
            width=1000,
            xaxis=dict(tickmode='linear', dtick=20),
            yaxis=dict(tickmode='linear', dtick=20)
        )
        
        return fig

    def plot_entry_exit_moment_matrix(self, dfs: list, max_entry_minute: int = 240, max_exit_minute: int = 240) -> go.Figure:
        """
        Plot a heatmap showing the average return for each (entry minute, exit minute) pair across all tokens.
        Args:
            dfs: List of pandas DataFrames, one per token, each with 'price' and 'datetime'.
            max_entry_minute: Only consider entry times up to this many minutes after launch.
            max_exit_minute: Only consider exit times up to this many minutes after launch.
        Returns:
            Plotly Figure (heatmap)
        """
        entry_minutes = np.arange(0, max_entry_minute)
        exit_minutes = np.arange(0, max_exit_minute)
        heatmap = np.full((len(entry_minutes), len(exit_minutes)), np.nan)
        counts = np.zeros_like(heatmap)

        for df in dfs:
            prices = df['price'].to_numpy()
            n = min(len(prices), max_exit_minute)
            for entry_idx, entry_minute in enumerate(entry_minutes):
                if entry_minute >= n - 1:
                    continue
                entry_price = prices[entry_minute]
                if np.isnan(entry_price) or entry_price == 0:
                    continue
                for exit_idx, exit_minute in enumerate(exit_minutes):
                    if exit_minute <= entry_minute or exit_minute >= n:
                        continue
                    exit_price = prices[exit_minute]
                    if np.isnan(exit_price):
                        continue
                    trade_return = (exit_price / entry_price - 1) * 100
                    if np.isfinite(trade_return):
                        if np.isnan(heatmap[entry_idx, exit_idx]):
                            heatmap[entry_idx, exit_idx] = 0
                        heatmap[entry_idx, exit_idx] += trade_return
                        counts[entry_idx, exit_idx] += 1

        # Average
        with np.errstate(invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, np.nan)

        # Mask lower triangle (entry >= exit)
        mask = np.tri(len(entry_minutes), len(exit_minutes), k=0, dtype=bool)
        heatmap = np.where(mask, np.nan, heatmap)

        # Plot
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            x=exit_minutes,
            y=entry_minutes,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Avg Return %"),
            hoverongaps=False,
            text=np.round(heatmap, 2),
            texttemplate='%{text}%',
            textfont={"size": 8}
        ))
        fig.update_layout(
            title="Entry/Exit Moment Matrix: Avg Return by Entry and Exit Minute",
            xaxis_title="Exit Minute",
            yaxis_title="Entry Minute",
            height=700,
            width=1000
        )
        return fig

    def plot_lifecycle_summary_charts(self, lifecycle_df: pl.DataFrame, analysis_metrics: List[str]) -> go.Figure:
        """
        Create comprehensive summary charts for 24-hour lifecycle analysis
        Shows multiple metrics across lifecycle segments
        """
        # Aggregate data by segment
        numeric_cols = [col for col in lifecycle_df.columns if col.endswith('_Pct') or col in ['Trend_Strength', 'Volume_Proxy']]
        
        if not numeric_cols:
            # Create empty figure if no numeric data
            fig = go.Figure()
            fig.add_annotation(text="No numeric data available for visualization", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        summary_stats = lifecycle_df.group_by('Lifecycle_Segment').agg([
            pl.col(col).mean().alias(f'Avg_{col}') for col in numeric_cols if col in lifecycle_df.columns
        ] + [
            pl.col('Token').count().alias('Token_Count'),
            pl.col('Hours_Into_Lifecycle').first().alias('Hours_Range')
        ]).sort('Lifecycle_Segment')
        
        # Count actual available metrics in the data
        available_metrics = []
        if "Returns" in analysis_metrics and 'Avg_Cumulative_Return_Pct' in summary_stats.columns:
            available_metrics.append("Returns")
        if "Volatility" in analysis_metrics and 'Avg_Volatility_Pct' in summary_stats.columns:
            available_metrics.append("Volatility")
        if "Price Momentum" in analysis_metrics and 'Avg_Win_Rate_Pct' in summary_stats.columns:
            available_metrics.append("Price Momentum")
        if "Volume Proxy" in analysis_metrics and 'Avg_Volume_Proxy' in summary_stats.columns:
            available_metrics.append("Volume Proxy")
        if "Trend Strength" in analysis_metrics and 'Avg_Trend_Strength' in summary_stats.columns:
            available_metrics.append("Trend Strength")
        
        n_metrics = len(available_metrics)
        if n_metrics == 0:
            fig = go.Figure()
            fig.add_annotation(text="No metrics available for visualization", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate grid dimensions
        n_cols = 2 if n_metrics > 1 else 1
        n_rows = (n_metrics + 1) // 2 if n_metrics > 1 else 1
        
        subplot_titles = []
        for metric in available_metrics:
            if metric == "Returns":
                subplot_titles.append("Average Returns by Lifecycle Segment")
            elif metric == "Volatility":
                subplot_titles.append("Average Volatility by Lifecycle Segment")
            elif metric == "Price Momentum":
                subplot_titles.append("Win Rate by Lifecycle Segment")
            elif metric == "Volume Proxy":
                subplot_titles.append("Volume Proxy by Lifecycle Segment")
            elif metric == "Trend Strength":
                subplot_titles.append("Trend Strength by Lifecycle Segment")
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        segments = summary_stats['Lifecycle_Segment'].to_list()
        hours_ranges = summary_stats['Hours_Range'].to_list()
        
        # Plot each available metric
        for plot_idx, metric in enumerate(available_metrics, 1):
            row, col = ((plot_idx - 1) // n_cols) + 1, ((plot_idx - 1) % n_cols) + 1
            
            if metric == "Returns":
                returns = summary_stats['Avg_Cumulative_Return_Pct'].to_list()
                fig.add_trace(
                    go.Bar(
                        x=hours_ranges,
                        y=returns,
                        name="Avg Returns",
                        marker_color=['green' if r > 0 else 'red' for r in returns],
                        text=[f"{r:.2f}%" for r in returns],
                        textposition='outside'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(title_text="Return %", row=row, col=col)
                fig.update_xaxes(title_text="Hours into Lifecycle", row=row, col=col)
                
            elif metric == "Volatility":
                volatility = summary_stats['Avg_Volatility_Pct'].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=hours_ranges,
                        y=volatility,
                        mode='lines+markers',
                        name="Avg Volatility",
                        line=dict(color='orange', width=3),
                        marker=dict(size=8),
                        text=[f"{v:.2f}%" for v in volatility],
                        textposition='top center'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(title_text="Volatility %", row=row, col=col)
                fig.update_xaxes(title_text="Hours into Lifecycle", row=row, col=col)
                
            elif metric == "Price Momentum":
                win_rates = summary_stats['Avg_Win_Rate_Pct'].to_list()
                fig.add_trace(
                    go.Bar(
                        x=hours_ranges,
                        y=win_rates,
                        name="Win Rate",
                        marker_color='blue',
                        text=[f"{w:.1f}%" for w in win_rates],
                        textposition='outside'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(title_text="Win Rate %", row=row, col=col)
                fig.update_xaxes(title_text="Hours into Lifecycle", row=row, col=col)
                
            elif metric == "Volume Proxy":
                volume_proxy = summary_stats['Avg_Volume_Proxy'].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=hours_ranges,
                        y=volume_proxy,
                        mode='lines+markers',
                        name="Volume Proxy",
                        line=dict(color='purple', width=3),
                        marker=dict(size=8),
                        text=[f"{v:.2f}" for v in volume_proxy],
                        textposition='top center'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(title_text="Volume Proxy", row=row, col=col)
                fig.update_xaxes(title_text="Hours into Lifecycle", row=row, col=col)
                
            elif metric == "Trend Strength":
                trend_strength = summary_stats['Avg_Trend_Strength'].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=hours_ranges,
                        y=trend_strength,
                        mode='lines+markers',
                        name="Trend Strength",
                        line=dict(color='darkgreen', width=3),
                        marker=dict(size=8),
                        text=[f"{t:.3f}" for t in trend_strength],
                        textposition='top center'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(title_text="Trend Strength", row=row, col=col)
                fig.update_xaxes(title_text="Hours into Lifecycle", row=row, col=col)
        
        fig.update_layout(
            height=400 * n_rows,
            width=1200,
            title_text="24-Hour Lifecycle Analysis - Summary Charts",
            showlegend=False
        )
        
        return fig
    
    def plot_lifecycle_aggregated_analysis(self, lifecycle_df: pl.DataFrame, metric_col: str, segment_col: str = 'Lifecycle_Segment') -> go.Figure:
        """
        Create aggregated analysis plots instead of unreadable heatmaps for large token counts
        Shows distribution statistics, percentiles, and patterns across lifecycle segments
        """
        if metric_col not in lifecycle_df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Metric '{metric_col}' not found in data", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate statistics for each segment
        segment_stats = lifecycle_df.group_by(segment_col).agg([
            pl.col(metric_col).mean().alias('Mean'),
            pl.col(metric_col).median().alias('Median'),
            pl.col(metric_col).std().alias('Std_Dev'),
            pl.col(metric_col).quantile(0.25).alias('Q25'),
            pl.col(metric_col).quantile(0.75).alias('Q75'),
            pl.col(metric_col).quantile(0.1).alias('P10'),
            pl.col(metric_col).quantile(0.9).alias('P90'),
            pl.col(metric_col).min().alias('Min'),
            pl.col(metric_col).max().alias('Max'),
            pl.col(metric_col).count().alias('Count')
        ]).sort(segment_col)
        
        segments = segment_stats[segment_col].to_list()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{metric_col} - Central Tendency',
                f'{metric_col} - Distribution Spread', 
                f'{metric_col} - Percentile Bands',
                f'{metric_col} - Extreme Values'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Mean and Median
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Mean'].to_list(),
                mode='lines+markers', name='Mean',
                line=dict(color='blue', width=3), marker=dict(size=8)
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Median'].to_list(),
                mode='lines+markers', name='Median',
                line=dict(color='red', width=2, dash='dash'), marker=dict(size=6)
            ), row=1, col=1
        )
        
        # Plot 2: Standard deviation and IQR
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Std_Dev'].to_list(),
                mode='lines+markers', name='Std Dev',
                line=dict(color='orange', width=3), marker=dict(size=8)
            ), row=1, col=2
        )
        iqr = (segment_stats['Q75'] - segment_stats['Q25']).to_list()
        fig.add_trace(
            go.Scatter(
                x=segments, y=iqr,
                mode='lines+markers', name='IQR (Q75-Q25)',
                line=dict(color='purple', width=2, dash='dot'), marker=dict(size=6)
            ), row=1, col=2
        )
        
        # Plot 3: Percentile bands (10th-90th percentile)
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['P90'].to_list(),
                mode='lines', name='90th Percentile',
                line=dict(color='lightgreen', width=1),
                showlegend=False
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['P10'].to_list(),
                mode='lines', name='10th Percentile',
                line=dict(color='lightgreen', width=1),
                fill='tonexty', fillcolor='rgba(144,238,144,0.3)',
                showlegend=False
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Median'].to_list(),
                mode='lines+markers', name='Median',
                line=dict(color='green', width=2), marker=dict(size=6)
            ), row=2, col=1
        )
        
        # Plot 4: Min/Max range
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Max'].to_list(),
                mode='markers', name='Max',
                marker=dict(color='red', size=10, symbol='triangle-up')
            ), row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Min'].to_list(),
                mode='markers', name='Min',
                marker=dict(color='blue', size=10, symbol='triangle-down')
            ), row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=segments, y=segment_stats['Mean'].to_list(),
                mode='lines', name='Mean',
                line=dict(color='black', width=2, dash='dash')
            ), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Lifecycle Analysis: {metric_col} Distribution Statistics",
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Lifecycle Segment")
        fig.update_yaxes(title_text=metric_col)
        
        return fig
    
    def plot_lifecycle_token_ranking(self, lifecycle_df: pl.DataFrame, metric_col: str, segment_col: str = 'Lifecycle_Segment', top_n: int = 20) -> go.Figure:
        """
        Show top/bottom performing tokens for a specific metric across lifecycle segments
        Much more useful than an unreadable 1000-token heatmap
        """
        if metric_col not in lifecycle_df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Metric '{metric_col}' not found in data", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Special handling for 24h metrics (only available in last segment)
        if metric_col in ['Total_24h_Return_Pct', 'Peak_24h_Return_Pct']:
            # For 24h metrics, we only use the last segment values (not average across segments)
            token_performance = lifecycle_df.filter(pl.col(metric_col).is_not_null()).group_by('Token').agg([
                pl.col(metric_col).first().alias('Avg_Performance'),  # Use first (only) non-null value
                pl.lit(0.0).alias('Volatility'),  # No volatility for single-point metrics
                pl.col(metric_col).count().alias('Segments_Count')
            ]).sort('Avg_Performance', descending=True)
        else:
            # For regular metrics, calculate average performance per token across all segments
            token_performance = lifecycle_df.group_by('Token').agg([
                pl.col(metric_col).mean().alias('Avg_Performance'),
                pl.col(metric_col).std().alias('Volatility'),
                pl.col(metric_col).count().alias('Segments_Count')
            ]).sort('Avg_Performance', descending=True)
        
        # Get top and bottom performers
        top_performers = token_performance.head(top_n)
        bottom_performers = token_performance.tail(top_n)
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Top {top_n} Performers',
                f'Bottom {top_n} Performers', 
                f'Performance vs Risk',
                f'Distribution (All Tokens)'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]],
            vertical_spacing=0.25,  # Increased from 0.12
            horizontal_spacing=0.15  # Increased from 0.1
        )
        
        # Helper function to truncate long token names
        def truncate_token_name(name, max_length=12):
            return name[:max_length] + "..." if len(name) > max_length else name
        
        # Top performers bar chart
        top_token_names = [truncate_token_name(name) for name in top_performers['Token'].to_list()]
        fig.add_trace(
            go.Bar(
                x=top_token_names,
                y=top_performers['Avg_Performance'].to_list(),
                name='Top Performers',
                marker_color='green',
                text=[f"{v:.2f}" for v in top_performers['Avg_Performance'].to_list()],
                textposition='outside',
                hovertext=top_performers['Token'].to_list(),  # Full names on hover
                hovertemplate='<b>%{hovertext}</b><br>Performance: %{y:.2f}<extra></extra>'
            ), row=1, col=1
        )
        
        # Bottom performers bar chart
        bottom_token_names = [truncate_token_name(name) for name in bottom_performers['Token'].to_list()]
        fig.add_trace(
            go.Bar(
                x=bottom_token_names,
                y=bottom_performers['Avg_Performance'].to_list(),
                name='Bottom Performers',
                marker_color='red',
                text=[f"{v:.2f}" for v in bottom_performers['Avg_Performance'].to_list()],
                textposition='outside',
                hovertext=bottom_performers['Token'].to_list(),  # Full names on hover
                hovertemplate='<b>%{hovertext}</b><br>Performance: %{y:.2f}<extra></extra>'
            ), row=1, col=2
        )
        
        # Performance vs Volatility scatter (remove overlapping text, use hover instead)
        fig.add_trace(
            go.Scatter(
                x=top_performers['Volatility'].to_list(),
                y=top_performers['Avg_Performance'].to_list(),
                mode='markers',  # Remove text mode to avoid overlap
                name='Top Performers',
                marker=dict(size=10, color='green', opacity=0.7),
                text=top_performers['Token'].to_list(),  # Keep for hover
                hovertemplate='<b>%{text}</b><br>Performance: %{y:.2f}<br>Volatility: %{x:.2f}<extra></extra>'
            ), row=2, col=1
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=token_performance['Avg_Performance'].to_list(),
                nbinsx=30,
                name='Distribution',
                marker_color='blue',
                opacity=0.7
            ), row=2, col=2
        )
        
        # Update layout with better spacing
        fig.update_layout(
            height=1000,  # Increased height for better spacing
            width=1400,
            title_text=f"Token Performance Ranking: {metric_col}",
            showlegend=False,
            margin=dict(t=100, b=150, l=80, r=80)  # Increased margins
        )
        
        # Better x-axis formatting for token names
        fig.update_xaxes(
            tickangle=90,  # Vertical labels instead of 45°
            tickfont=dict(size=10),  # Smaller font
            row=1, col=1
        )
        fig.update_xaxes(
            tickangle=90,  # Vertical labels instead of 45°
            tickfont=dict(size=10),  # Smaller font
            row=1, col=2
        )
        fig.update_xaxes(title_text="Risk (Volatility)", row=2, col=1)
        fig.update_xaxes(title_text="Average Performance", row=2, col=2)
        fig.update_yaxes(title_text=metric_col, row=1, col=1)
        fig.update_yaxes(title_text=metric_col, row=1, col=2)
        fig.update_yaxes(title_text="Performance", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig

    def plot_lifecycle_heatmap(self, lifecycle_df: pl.DataFrame, metric_col: str, segment_col: str = 'Lifecycle_Segment') -> go.Figure:
        """
        DEPRECATED: This function creates unreadable heatmaps with many tokens.
        Use plot_lifecycle_aggregated_analysis() or plot_lifecycle_token_ranking() instead.
        
        Kept for backward compatibility but shows a warning message.
        """
        n_tokens = lifecycle_df['Token'].n_unique()
        
        if n_tokens > 50:
            fig = go.Figure()
            fig.add_annotation(
                text=f"⚠️ Heatmap with {n_tokens} tokens is not readable!<br><br>"
                     f"Use 'Aggregated Analysis' or 'Token Ranking' visualizations instead.<br>"
                     f"These provide much better insights for large datasets.",
                xref="paper", yref="paper", 
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16, color="red"),
                bgcolor="lightyellow",
                bordercolor="orange",
                borderwidth=2
            )
            fig.update_layout(
                title=f"Heatmap Not Suitable for {n_tokens} Tokens",
                height=400,
                width=800
            )
            return fig
        
        # Original heatmap logic for small datasets (≤50 tokens)
        if metric_col not in lifecycle_df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Metric '{metric_col}' not found in data", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Pivot data for heatmap
        pivot_data = lifecycle_df.pivot(
            index='Token', 
            columns=segment_col, 
            values=metric_col
        ).fill_null(0)
        
        # Get data for heatmap
        tokens = pivot_data.get_column('Token').to_list() if 'Token' in pivot_data.columns else pivot_data.select(pl.col('*').exclude(segment_col)).columns
        segments = [col for col in pivot_data.columns if col != 'Token']
        
        # Create matrix
        z_data = []
        for token in tokens:
            if token in pivot_data.get_column('Token').to_list():
                row_data = pivot_data.filter(pl.col('Token') == token).select(segments).to_numpy()[0]
            else:
                row_data = [0] * len(segments)
            z_data.append(row_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=segments,
            y=tokens,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title=metric_col),
            hoverongaps=False,
            text=np.round(np.array(z_data), 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title=f"Lifecycle Heatmap: {metric_col} (Small Dataset)",
            xaxis_title="Lifecycle Segment",
            yaxis_title="Token",
            height=max(400, len(tokens) * 20),
            width=800
        )
        
        return fig

    def plot_lifecycle_comparison(self, lifecycle_df: pl.DataFrame, comparison_type: str = "early_vs_late") -> go.Figure:
        """
        Create comparison plots for different lifecycle phases
        """
        if comparison_type == "early_vs_late":
            # Compare first 25% vs last 25% of segments
            segments = sorted(lifecycle_df['Lifecycle_Segment'].unique().to_list())
            n_segments = len(segments)
            
            early_segments = segments[:max(1, n_segments // 4)]
            late_segments = segments[-max(1, n_segments // 4):]
            
            early_data = lifecycle_df.filter(pl.col('Lifecycle_Segment').is_in(early_segments))
            late_data = lifecycle_df.filter(pl.col('Lifecycle_Segment').is_in(late_segments))
            
            # Calculate averages for comparison with null handling
            # Filter out 24h metrics that only exist in last segment and wouldn't be meaningful for early/late comparison
            metrics_to_compare = [col for col in lifecycle_df.columns 
                                if col.endswith('_Pct') and col not in ['Total_24h_Return_Pct', 'Peak_24h_Return_Pct']]
            
            comparison_data = []
            for metric in metrics_to_compare:
                if metric in early_data.columns and metric in late_data.columns:
                    # Handle null values properly
                    early_values = early_data[metric].drop_nulls()
                    late_values = late_data[metric].drop_nulls()
                    
                    # Only proceed if we have data for both phases
                    if len(early_values) > 0 and len(late_values) > 0:
                        early_avg = early_values.mean()
                        late_avg = late_values.mean()
                        
                        # Additional null check after mean calculation
                        if early_avg is not None and late_avg is not None:
                            comparison_data.append({
                                'Metric': metric.replace('_Pct', '').replace('_', ' ').title(),
                                'Early Phase': early_avg,
                                'Late Phase': late_avg,
                                'Difference': late_avg - early_avg
                            })
            
            if not comparison_data:
                fig = go.Figure()
                fig.add_annotation(
                    text="No comparison data available<br><br>This might happen if:<br>• Metrics contain only null values<br>• Early/late phases have no valid data<br>• Selected metrics are not present in lifecycle data", 
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14),
                    bgcolor="lightyellow",
                    bordercolor="orange",
                    borderwidth=1
                )
                fig.update_layout(
                    title="Early vs Late Comparison - No Data Available",
                    height=400,
                    width=800
                )
                return fig
            
            comparison_df = pl.DataFrame(comparison_data)
            
            fig = go.Figure()
            
            # Add bars for early and late phases
            fig.add_trace(go.Bar(
                name='Early Phase',
                x=comparison_df['Metric'].to_list(),
                y=comparison_df['Early Phase'].to_list(),
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Late Phase',
                x=comparison_df['Metric'].to_list(),
                y=comparison_df['Late Phase'].to_list(),
                marker_color='darkblue'
            ))
            
            # Add summary statistics to the plot
            early_segments_str = ", ".join(early_segments)
            late_segments_str = ", ".join(late_segments)
            
            fig.update_layout(
                title=f'Early vs Late Lifecycle Phase Comparison<br><sub>Early: {early_segments_str} | Late: {late_segments_str}</sub>',
                xaxis_title='Metrics',
                yaxis_title='Average Value (%)',
                barmode='group',
                height=600,
                width=1200,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=100)
            )
            
            # Add value annotations on bars
            for i, (early_val, late_val) in enumerate(zip(comparison_df['Early Phase'].to_list(), comparison_df['Late Phase'].to_list())):
                # Add text annotations for better readability
                fig.add_annotation(
                    x=i,
                    y=max(early_val, late_val) + abs(max(early_val, late_val)) * 0.05,
                    text=f"Δ: {late_val - early_val:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color="red" if late_val < early_val else "green")
                )
            
            return fig
        
        else:
            # Default empty figure for other comparison types
            fig = go.Figure()
            fig.add_annotation(text=f"Comparison type '{comparison_type}' not implemented", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def plot_multi_token_risk_metrics(self, combined_results: pl.DataFrame) -> go.Figure:
        """
        Create comprehensive visualizations for Multi-Token Risk Metrics Analysis
        Shows average win rate, Sharpe ratio, and risk/reward ratio per time horizon
        """
        if combined_results.is_empty():
            fig = go.Figure()
            fig.add_annotation(text="No data available for visualization", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate summary statistics by horizon
        summary_stats = combined_results.group_by('horizon_minutes').agg([
            pl.col('sharpe_ratio').mean().alias('Avg_Sharpe'),
            pl.col('win_rate').mean().alias('Avg_Win_Rate'),
            pl.col('risk_reward_ratio').mean().alias('Avg_Risk_Reward'),
            pl.col('sharpe_ratio').std().alias('Sharpe_Std'),
            pl.col('win_rate').std().alias('Win_Rate_Std'),
            pl.col('risk_reward_ratio').std().alias('Risk_Reward_Std'),
            pl.col('Token').count().alias('Token_Count')
        ]).sort('horizon_minutes')
        
        horizons = summary_stats['horizon_minutes'].to_list()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Win Rate by Time Horizon',
                'Average Sharpe Ratio by Time Horizon',
                'Average Risk/Reward Ratio by Time Horizon',
                'Metrics Distribution Comparison'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Win Rate with error bars
        win_rates = summary_stats['Avg_Win_Rate'].to_list()
        win_rate_stds = summary_stats['Win_Rate_Std'].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=horizons, y=win_rates,
                mode='lines+markers',
                name='Avg Win Rate',
                line=dict(color='green', width=3),
                marker=dict(size=10),
                error_y=dict(type='data', array=win_rate_stds, visible=True),
                text=[f"{wr:.1f}%" for wr in win_rates],
                textposition='top center',
                hovertemplate='Horizon: %{x} min<br>Win Rate: %{y:.1f}%<br>Std: %{error_y.array:.1f}%<extra></extra>'
            ), row=1, col=1
        )
        
        # Plot 2: Sharpe Ratio with error bars
        sharpe_ratios = summary_stats['Avg_Sharpe'].to_list()
        sharpe_stds = summary_stats['Sharpe_Std'].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=horizons, y=sharpe_ratios,
                mode='lines+markers',
                name='Avg Sharpe Ratio',
                line=dict(color='blue', width=3),
                marker=dict(size=10),
                error_y=dict(type='data', array=sharpe_stds, visible=True),
                text=[f"{sr:.2f}" for sr in sharpe_ratios],
                textposition='top center',
                hovertemplate='Horizon: %{x} min<br>Sharpe: %{y:.2f}<br>Std: %{error_y.array:.2f}<extra></extra>'
            ), row=1, col=2
        )
        
        # Plot 3: Risk/Reward Ratio with error bars
        risk_rewards = summary_stats['Avg_Risk_Reward'].to_list()
        risk_reward_stds = summary_stats['Risk_Reward_Std'].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=horizons, y=risk_rewards,
                mode='lines+markers',
                name='Avg Risk/Reward',
                line=dict(color='purple', width=3),
                marker=dict(size=10),
                error_y=dict(type='data', array=risk_reward_stds, visible=True),
                text=[f"{rr:.2f}" for rr in risk_rewards],
                textposition='top center',
                hovertemplate='Horizon: %{x} min<br>Risk/Reward: %{y:.2f}<br>Std: %{error_y.array:.2f}<extra></extra>'
            ), row=2, col=1
        )
        
        # Plot 4: Comparative bar chart showing all metrics normalized
        # Normalize metrics to 0-1 scale for comparison
        max_win_rate = max(win_rates) if win_rates else 1
        max_sharpe = max([abs(s) for s in sharpe_ratios]) if sharpe_ratios else 1
        max_risk_reward = max(risk_rewards) if risk_rewards else 1
        
        normalized_win_rates = [wr / max_win_rate for wr in win_rates]
        normalized_sharpe = [sr / max_sharpe for sr in sharpe_ratios]
        normalized_risk_reward = [rr / max_risk_reward for rr in risk_rewards]
        
        x_pos = np.arange(len(horizons))
        width = 0.25
        
        fig.add_trace(
            go.Bar(
                x=[f"{h}min" for h in horizons],
                y=normalized_win_rates,
                name='Win Rate (norm)',
                marker_color='green',
                opacity=0.7,
                text=[f"{wr:.1f}%" for wr in win_rates],
                textposition='outside'
            ), row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=[f"{h}min" for h in horizons],
                y=normalized_sharpe,
                name='Sharpe (norm)',
                marker_color='blue',
                opacity=0.7,
                text=[f"{sr:.2f}" for sr in sharpe_ratios],
                textposition='outside'
            ), row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=[f"{h}min" for h in horizons],
                y=normalized_risk_reward,
                name='Risk/Reward (norm)',
                marker_color='purple',
                opacity=0.7,
                text=[f"{rr:.2f}" for rr in risk_rewards],
                textposition='outside'
            ), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Multi-Token Risk Metrics Analysis - {summary_stats['Token_Count'].sum()} Total Observations",
            showlegend=True,
            barmode='group'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Horizon (minutes)", row=1, col=1)
        fig.update_xaxes(title_text="Time Horizon (minutes)", row=1, col=2)
        fig.update_xaxes(title_text="Time Horizon (minutes)", row=2, col=1)
        fig.update_xaxes(title_text="Time Horizon", row=2, col=2)
        
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Risk/Reward Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Value", row=2, col=2)
        
        return fig