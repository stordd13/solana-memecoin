"""
Enhanced Token Correlation Analysis for Memecoin Analysis
Implements multi-method correlation analysis with PCA redundancy detection - ROADMAP SECTION 3

Key Features:
- Multi-method correlations (Pearson, Spearman, Kendall) - ROADMAP REQUIREMENT
- PCA explained variance ratio analysis - ROADMAP REQUIREMENT
- Log-returns correlation analysis - ROADMAP REQUIREMENT
- Interactive correlation heatmaps
- Significant correlation pair detection
"""

import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import pandas as pd
import networkx as nx
warnings.filterwarnings('ignore')


class TokenCorrelationAnalyzer:
    """Analyze correlations and relationships between memecoin tokens"""
    
    def __init__(self):
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.time_windows = [60, 240, 720, 1440]  # 1h, 4h, 12h, 24h
    
    def analyze_token_correlations(self, 
                                 token_data: Dict[str, pl.DataFrame],
                                 method: str = 'pearson',
                                 min_overlap: int = 100,
                                 use_log_returns: bool = True,
                                 use_robust_scaling: bool = False,
                                 use_rolling: bool = False,
                                 rolling_window: Optional[int] = None,
                                 use_lifecycle_sync: bool = False,
                                 lifecycle_minutes: int = 240,
                                 n_components: Optional[int] = None) -> Dict:
        """
        Enhanced correlation analysis - ROADMAP REQUIREMENT
        
        Args:
            use_lifecycle_sync: If True, sync tokens by lifecycle position rather than absolute time
            lifecycle_minutes: Number of minutes from launch to compare (when using lifecycle sync)
        """
        
        if len(token_data) < 2:
            return {'error': 'Need at least 2 tokens for correlation analysis'}
        
        try:
            # 1. Prepare synchronized data with new lifecycle option
            if use_lifecycle_sync:
                sync_data = self._lifecycle_synchronize_tokens(
                    token_data, 
                    lifecycle_minutes,
                    use_log_returns, 
                    use_robust_scaling
                )
            else:
                sync_data = self._synchronize_token_data(
                    token_data, 
                    use_log_returns, 
                    use_robust_scaling
                )
            
            if sync_data is None or sync_data.shape[1] < 2:
                return {'error': 'Insufficient synchronized data'}
            
            # 2. Check minimum overlap requirement
            if sync_data.height < min_overlap:
                return {
                    'error': f'Insufficient data overlap: {sync_data.height} points < {min_overlap} required',
                    'data_points_found': sync_data.height,
                    'min_overlap_required': min_overlap,
                    'suggestion': 'Try lifecycle synchronization mode or reduce minimum overlap'
                }
            
            # 3. Calculate correlation matrices - ROADMAP REQUIREMENT
            correlation_matrices = self._calculate_correlation_matrices(
                sync_data, 
                method,
                use_rolling,
                rolling_window
            )
            
            # 4. Find highly correlated pairs
            corr_pairs = self._find_significant_correlations(correlation_matrices['main'], 
                                                           threshold=0.5)
            
            # 5. Enhanced PCA analysis for redundancy detection - ROADMAP REQUIREMENT
            pca_analysis = self._enhanced_pca_analysis(sync_data, n_components=n_components)
            
            return {
                'status': 'success',
                'method': method,
                'tokens_analyzed': list(token_data.keys()),
                'data_points': sync_data.shape[0],
                'min_overlap_met': True,
                'correlation_matrices': correlation_matrices,
                'significant_pairs': corr_pairs,
                'pca_analysis': pca_analysis,
                'summary_stats': self._generate_correlation_summary(correlation_matrices['main']),
                'rolling_analysis': use_rolling,
                'rolling_window': rolling_window,
                'scaling_method': 'robust_scaler' if use_robust_scaling else 'simple_normalization' if not use_log_returns else 'log_returns',
                'sync_method': 'lifecycle' if use_lifecycle_sync else 'temporal',
                'lifecycle_minutes': lifecycle_minutes if use_lifecycle_sync else None
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    def _synchronize_token_data(self, 
                               token_data: Dict[str, pl.DataFrame],
                               use_log_returns: bool = True,
                               use_robust_scaling: bool = False) -> Optional[pl.DataFrame]:
        """Synchronize token data to common timeframe using Polars - IMPROVED VERSION"""
        
        if len(token_data) < 2:
            return None
        
        prepared_dfs = []
        
        # If using robust scaling, fit scaler on all price data first
        if use_robust_scaling and not use_log_returns:
            # Collect all prices for fitting the scaler
            all_prices = []
            for df in token_data.values():
                prices = df['price'].to_numpy()
                all_prices.extend(prices[~np.isnan(prices)])
            
            if len(all_prices) == 0:
                return None
                
            # Fit RobustScaler on all price data
            scaler = RobustScaler()
            scaler.fit(np.array(all_prices).reshape(-1, 1))
        else:
            scaler = None
        
        # Find the common time range across all tokens for better synchronization
        all_start_times = []
        all_end_times = []
        
        for token_name, df in token_data.items():
            if 'datetime' in df.columns and len(df) > 0:
                df_sorted = df.sort('datetime')
                all_start_times.append(df_sorted['datetime'].min())
                all_end_times.append(df_sorted['datetime'].max())
        
        if not all_start_times:
            return None
        
        # Use overlapping time range
        common_start = max(all_start_times)
        common_end = min(all_end_times)
        
        # Check if there's any meaningful overlap
        if common_start >= common_end:
            # No overlap - try a different approach with resampling
            return self._fallback_synchronization(token_data, use_log_returns, use_robust_scaling, scaler)
        
        for token_name, df in token_data.items():
            try:
                # Ensure proper datetime sorting with Polars
                df = df.sort('datetime')
                
                # Filter to common time range
                df_filtered = df.filter(
                    (pl.col('datetime') >= common_start) & 
                    (pl.col('datetime') <= common_end)
                )
                
                if len(df_filtered) < 10:  # Need at least 10 data points
                    continue
                
                # Calculate the metric to correlate using Polars
                if use_log_returns:
                    # Calculate log returns - ROADMAP REQUIREMENT
                    df_with_returns = df_filtered.with_columns([
                        (pl.col('price').log() - pl.col('price').shift(1).log()).alias('returns')
                    ]).drop_nulls('returns')
                    
                    # Rename the column to include token name
                    df_sync = df_with_returns.select([
                        pl.col('datetime'),
                        pl.col('returns').alias(token_name)
                    ])
                
                elif use_robust_scaling:
                    # Use RobustScaler for price normalization
                    prices = df_filtered['price'].to_numpy().reshape(-1, 1)
                    scaled_prices = scaler.transform(prices).flatten()
                    
                    # Create DataFrame with scaled prices
                    df_with_scaled = df_filtered.with_columns([
                        pl.Series('scaled_price', scaled_prices)
                    ])
                    
                    df_sync = df_with_scaled.select([
                        pl.col('datetime'),
                        pl.col('scaled_price').alias(token_name)
                    ])
                
                else:
                    # Use simple normalization (divide by first price)
                    first_price = df_filtered['price'][0]
                    if first_price <= 0:
                        continue
                        
                    df_with_norm = df_filtered.with_columns([
                        (pl.col('price') / first_price).alias('normalized_price')
                    ])
                    
                    df_sync = df_with_norm.select([
                        pl.col('datetime'),
                        pl.col('normalized_price').alias(token_name)
                    ])
                
                prepared_dfs.append(df_sync)
                
            except Exception as e:
                print(f"Error processing {token_name}: {e}")
                continue
        
        if len(prepared_dfs) < 2:
            # Try fallback approach
            return self._fallback_synchronization(token_data, use_log_returns, use_robust_scaling, scaler)
        
        # Try inner join first, then fall back to outer join with interpolation
        try:
            # Merge all dataframes using Polars join operations
            merged_df = prepared_dfs[0]
            for df_sync in prepared_dfs[1:]:
                merged_df = merged_df.join(df_sync, on='datetime', how='inner')
            
            # Drop rows with any null values
            merged_df = merged_df.drop_nulls()
            
            # If we have enough data, return it
            if merged_df.height >= 50:
                return merged_df
            
            # Otherwise try outer join with interpolation
            merged_df_outer = prepared_dfs[0]
            for df_sync in prepared_dfs[1:]:
                merged_df_outer = merged_df_outer.join(df_sync, on='datetime', how='outer')
            
            # Sort by datetime and forward fill missing values
            merged_df_outer = merged_df_outer.sort('datetime')
            
            # Forward fill and then backward fill to handle NaN values
            numeric_cols = [col for col in merged_df_outer.columns if col != 'datetime']
            for col in numeric_cols:
                merged_df_outer = merged_df_outer.with_columns([
                    pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward").alias(col)
                ])
            
            # Drop any remaining nulls
            merged_df_outer = merged_df_outer.drop_nulls()
            
            if merged_df_outer.height >= 30:  # Lower threshold for outer join
                return merged_df_outer
                
        except Exception as e:
            print(f"Error in join operations: {e}")
        
        # Last resort: use fallback method
        return self._fallback_synchronization(token_data, use_log_returns, use_robust_scaling, scaler)
    
    def _fallback_synchronization(self, 
                                 token_data: Dict[str, pl.DataFrame],
                                 use_log_returns: bool = True,
                                 use_robust_scaling: bool = False,
                                 scaler = None) -> Optional[pl.DataFrame]:
        """Fallback synchronization method using minute-level resampling"""
        
        try:
            synchronized_data = {}
            
            for token_name, df in token_data.items():
                df = df.sort('datetime')
                
                if len(df) < 20:  # Need minimum data
                    continue
                
                # Create minute-level index
                start_time = df['datetime'].min()
                end_time = df['datetime'].max()
                
                # Create minute range
                minute_range = pl.date_range(
                    start=start_time.replace(second=0, microsecond=0),
                    end=end_time.replace(second=0, microsecond=0),
                    interval="1m"
                )
                
                # Round datetime to minutes for better matching
                df_rounded = df.with_columns([
                    pl.col('datetime').dt.truncate("1m").alias('datetime_minute')
                ])
                
                # Group by minute and take first price (could also use mean)
                df_resampled = df_rounded.group_by('datetime_minute').agg([
                    pl.col('price').first().alias('price')
                ]).sort('datetime_minute').rename({'datetime_minute': 'datetime'})
                
                # Calculate the analysis metric
                if use_log_returns and len(df_resampled) > 1:
                    df_with_metric = df_resampled.with_columns([
                        (pl.col('price').log() - pl.col('price').shift(1).log()).alias('metric')
                    ]).drop_nulls('metric')
                elif use_robust_scaling and scaler is not None:
                    prices = df_resampled['price'].to_numpy().reshape(-1, 1)
                    scaled_prices = scaler.transform(prices).flatten()
                    df_with_metric = df_resampled.with_columns([
                        pl.Series('metric', scaled_prices)
                    ])
                else:
                    first_price = df_resampled['price'][0]
                    if first_price <= 0:
                        continue
                    df_with_metric = df_resampled.with_columns([
                        (pl.col('price') / first_price).alias('metric')
                    ])
                
                if len(df_with_metric) >= 10:
                    synchronized_data[token_name] = df_with_metric.select(['datetime', 'metric'])
            
            if len(synchronized_data) < 2:
                return None
            
            # Find common time range
            all_times = set()
            for df in synchronized_data.values():
                all_times.update(df['datetime'].to_list())
            
            common_times = sorted(all_times)
            
            # Create synchronized DataFrame
            result_data = {'datetime': common_times}
            
            for token_name, df in synchronized_data.items():
                # Create mapping from datetime to metric
                time_to_metric = dict(zip(df['datetime'].to_list(), df['metric'].to_list()))
                
                # Fill in metrics for all common times
                metrics = []
                for dt in common_times:
                    if dt in time_to_metric:
                        metrics.append(time_to_metric[dt])
                    else:
                        metrics.append(None)
                
                result_data[token_name] = metrics
            
            # Convert to Polars DataFrame
            result_df = pl.DataFrame(result_data)
            
            # Forward fill and backward fill
            numeric_cols = [col for col in result_df.columns if col != 'datetime']
            for col in numeric_cols:
                result_df = result_df.with_columns([
                    pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward").alias(col)
                ])
            
            # Drop any remaining nulls
            result_df = result_df.drop_nulls()
            
            return result_df if result_df.height >= 20 else None
            
        except Exception as e:
            print(f"Fallback synchronization failed: {e}")
            return None
    
    def _calculate_correlation_matrices(self, 
                                      sync_data: pl.DataFrame,
                                      method: str,
                                      use_rolling: bool = False,
                                      rolling_window: Optional[int] = None) -> Dict:
        """Calculate correlation matrices - ROADMAP REQUIREMENT"""
        
        matrices = {}
        
        # Get numeric columns (exclude datetime)
        numeric_cols = [col for col in sync_data.columns if col != 'datetime']
        
        # Convert to numpy for correlation calculation
        data_matrix = sync_data.select(numeric_cols).to_numpy()
        
        # Calculate main correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(data_matrix.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data_matrix, axis=0)
        else:  # kendall
            from scipy.stats import kendalltau
            n_cols = len(numeric_cols)
            corr_matrix = np.zeros((n_cols, n_cols))
            for i in range(n_cols):
                for j in range(n_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        tau, _ = kendalltau(data_matrix[:, i], data_matrix[:, j])
                        corr_matrix[i, j] = tau
        
        # Convert to Polars DataFrame with token names as both index and columns
        matrices['main'] = pl.DataFrame(
            corr_matrix,
            schema={col: pl.Float64 for col in numeric_cols}
        ).with_columns(pl.Series("token", numeric_cols))
        
        # Enhanced rolling correlation analysis
        if use_rolling and rolling_window and len(sync_data) > rolling_window:
            matrices['rolling_correlations'] = self._calculate_rolling_correlations(
                sync_data, numeric_cols, rolling_window, method
            )
        
        return matrices
    
    def _calculate_rolling_correlations(self, 
                                      sync_data: pl.DataFrame,
                                      numeric_cols: List[str],
                                      window: int,
                                      method: str) -> pl.DataFrame:
        """Calculate rolling correlations between all token pairs"""
        
        rolling_results = []
        
        # Calculate rolling correlations for all unique pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                token1, token2 = numeric_cols[i], numeric_cols[j]
                
                # Calculate rolling correlation using Polars
                rolling_corr = sync_data.select([
                    pl.col('datetime'),
                    pl.corr(
                        pl.col(token1).rolling_mean(window),
                        pl.col(token2).rolling_mean(window)
                    ).alias(f'{token1}_{token2}_correlation')
                ]).drop_nulls()
                
                rolling_results.append({
                    'token_pair': f"{token1}_{token2}",
                    'rolling_data': rolling_corr
                })
        
        # Combine into single result
        if rolling_results:
            return rolling_results[0]['rolling_data']  # Return first pair for now
        
        return pl.DataFrame()  # Empty DataFrame if no results
    
    def _find_significant_correlations(self, 
                                     corr_matrix: pl.DataFrame,
                                     threshold: float = 0.5) -> List[Dict]:
        """Find significantly correlated token pairs"""
        
        significant_pairs = []
        
        # Get numeric columns and token names
        numeric_cols = [col for col in corr_matrix.columns if col != 'token']
        token_names = corr_matrix['token'].to_list()
        
        # Convert to numpy for analysis
        corr_values = corr_matrix.select(numeric_cols).to_numpy()
        
        # Get upper triangle to avoid duplicates
        mask = np.triu(np.ones_like(corr_values), k=1).astype(bool)
        
        for i in range(len(token_names)):
            for j in range(len(token_names)):
                if mask[i, j]:
                    corr_val = corr_values[i, j]
                    if abs(corr_val) >= threshold:
                        significant_pairs.append({
                            'token1': token_names[i],
                            'token2': token_names[j],
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val)),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        })
        
        # Sort by absolute correlation value
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_pairs
    
    def _lifecycle_synchronize_tokens(self, 
                                    token_data: Dict[str, pl.DataFrame],
                                    lifecycle_minutes: int = 240,
                                    use_log_returns: bool = True,
                                    use_robust_scaling: bool = False) -> Optional[pl.DataFrame]:
        """
        NEW: Synchronize tokens by lifecycle position rather than absolute time
        This allows correlation analysis between tokens launched at different dates
        """
        
        if len(token_data) < 2:
            return None
        
        prepared_dfs = []
        
        # If using robust scaling, fit scaler on all price data first
        if use_robust_scaling and not use_log_returns:
            all_prices = []
            for df in token_data.values():
                prices = df['price'].to_numpy()
                all_prices.extend(prices[~np.isnan(prices)])
            
            if len(all_prices) == 0:
                return None
                
            scaler = RobustScaler()
            scaler.fit(np.array(all_prices).reshape(-1, 1))
        else:
            scaler = None
        
        for token_name, df in token_data.items():
            try:
                # Sort by datetime and take first N minutes from launch
                df_sorted = df.sort('datetime')
                
                # Take first lifecycle_minutes of data
                df_lifecycle = df_sorted.head(min(lifecycle_minutes, len(df_sorted)))
                
                if len(df_lifecycle) < 30:  # Need minimum data
                    continue
                
                # Create minute-index from launch (0, 1, 2, 3, ...)
                df_with_index = df_lifecycle.with_columns([
                    pl.arange(0, len(df_lifecycle)).alias('lifecycle_minute')
                ])
                
                # Calculate the metric to correlate
                if use_log_returns:
                    df_with_returns = df_with_index.with_columns([
                        (pl.col('price').log() - pl.col('price').shift(1).log()).alias('returns')
                    ]).drop_nulls('returns')
                    
                    df_sync = df_with_returns.select([
                        pl.col('lifecycle_minute'),
                        pl.col('returns').alias(token_name)
                    ])
                
                elif use_robust_scaling:
                    prices = df_lifecycle['price'].to_numpy().reshape(-1, 1)
                    scaled_prices = scaler.transform(prices).flatten()
                    
                    df_with_scaled = df_with_index.with_columns([
                        pl.Series('scaled_price', scaled_prices)
                    ])
                    
                    df_sync = df_with_scaled.select([
                        pl.col('lifecycle_minute'),
                        pl.col('scaled_price').alias(token_name)
                    ])
                
                else:
                    # Use simple normalization (divide by first price)
                    first_price = df_lifecycle['price'][0]
                    if first_price <= 0:
                        continue
                        
                    df_with_norm = df_with_index.with_columns([
                        (pl.col('price') / first_price).alias('normalized_price')
                    ])
                    
                    df_sync = df_with_norm.select([
                        pl.col('lifecycle_minute'),
                        pl.col('normalized_price').alias(token_name)
                    ])
                
                prepared_dfs.append(df_sync)
                
            except Exception as e:
                print(f"Error processing {token_name} in lifecycle sync: {e}")
                continue
        
        if len(prepared_dfs) < 2:
            return None
        
        # Find common lifecycle range
        min_length = min(len(df) for df in prepared_dfs)
        common_minutes = list(range(min_length))
        
        # Create synchronized DataFrame using lifecycle minutes
        result_data = {'lifecycle_minute': common_minutes}
        
        for df_sync in prepared_dfs:
            token_name = [col for col in df_sync.columns if col != 'lifecycle_minute'][0]
            
            # Truncate to common length
            df_truncated = df_sync.head(min_length)
            
            # Map lifecycle minute to value
            minute_to_value = dict(zip(
                df_truncated['lifecycle_minute'].to_list(),
                df_truncated[token_name].to_list()
            ))
            
            # Fill in values for all common minutes
            values = []
            for minute in common_minutes:
                if minute in minute_to_value:
                    values.append(minute_to_value[minute])
                else:
                    values.append(None)
            
            result_data[token_name] = values
        
        # Convert to Polars DataFrame
        result_df = pl.DataFrame(result_data)
        
        # Handle any remaining nulls with interpolation
        numeric_cols = [col for col in result_df.columns if col != 'lifecycle_minute']
        for col in numeric_cols:
            result_df = result_df.with_columns([
                pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward").alias(col)
            ])
        
        # Drop any remaining nulls
        result_df = result_df.drop_nulls()
        
        return result_df if result_df.height >= 20 else None

    def _enhanced_pca_analysis(self, sync_data: pl.DataFrame, n_components: int = None) -> Dict:
        """Enhanced PCA analysis with configurable components - ROADMAP REQUIREMENT"""
        
        try:
            # Get numeric columns (exclude datetime/lifecycle_minute)
            numeric_cols = [col for col in sync_data.columns if col not in ['datetime', 'lifecycle_minute']]
            
            if len(numeric_cols) < 2:
                return {'pca_available': False, 'reason': 'insufficient_tokens'}
            
            # Convert to numpy for PCA
            data_matrix = sync_data.select(numeric_cols).to_numpy()
            
            # Handle NaN values
            if np.isnan(data_matrix).any():
                data_matrix = np.nan_to_num(data_matrix, nan=0.0)
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_matrix)
            
            # Determine number of components
            max_components = min(len(numeric_cols), data_scaled.shape[0] - 1)
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(data_scaled)
            
            # Calculate additional metrics
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            return {
                'pca_available': True,
                'n_components': n_components,
                'n_tokens': len(numeric_cols),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'explained_variance': pca.explained_variance_.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'principal_components': principal_components.tolist(),
                'feature_names': numeric_cols,
                'loadings': pca.components_.tolist(),  # How much each original feature contributes to each PC
                'total_variance_captured': float(cumulative_variance[-1]),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }
            
        except Exception as e:
            return {'pca_available': False, 'error': str(e)}
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength"""
        if abs_corr >= 0.8:
            return 'very_strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_correlation_summary(self, corr_matrix: pl.DataFrame) -> Dict:
        """Generate summary statistics for correlation matrix"""
        
        # Get numeric columns (exclude token)
        numeric_cols = [col for col in corr_matrix.columns if col != 'token']
        corr_values = corr_matrix.select(numeric_cols).to_numpy()
        
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(corr_values), k=1).astype(bool)
        upper_triangle_values = corr_values[mask]
        
        return {
            'avg_correlation': float(np.nanmean(upper_triangle_values)),
            'max_correlation': float(np.nanmax(upper_triangle_values)),
            'min_correlation': float(np.nanmin(upper_triangle_values)),
            'std_correlation': float(np.nanstd(upper_triangle_values)),
            'high_correlations_count': int(np.sum(np.abs(upper_triangle_values) >= 0.7))
        }
    
    def create_correlation_heatmap(self, 
                                 correlation_matrix: pl.DataFrame,
                                 title: str = "Token Correlation Heatmap") -> go.Figure:
        """Create interactive correlation heatmap - ROADMAP REQUIREMENT"""
        
        # Get numeric columns and token names
        numeric_cols = [col for col in correlation_matrix.columns if col != 'token']
        token_names = correlation_matrix['token'].to_list()
        corr_values = correlation_matrix.select(numeric_cols).to_numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_values,
            x=token_names,
            y=token_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tokens",
            yaxis_title="Tokens",
            width=800,
            height=600
        )
        
        return fig

    def create_pca_visualization(self, pca_results: Dict, max_components: int = 6) -> go.Figure:
        """Create comprehensive PCA visualization"""
        
        if not pca_results.get('pca_available', False):
            return None
        
        n_components = min(pca_results['n_components'], max_components)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Explained Variance by Component',
                'Cumulative Explained Variance',
                'Principal Components Loadings',
                'Biplot (PC1 vs PC2)'
            ),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # 1. Explained variance bar chart
        components = [f'PC{i+1}' for i in range(n_components)]
        explained_var = pca_results['explained_variance_ratio'][:n_components]
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=explained_var,
                text=[f'{v:.1%}' for v in explained_var],
                textposition='auto',
                marker_color='lightblue',
                name='Explained Variance'
            ),
            row=1, col=1
        )
        
        # 2. Cumulative variance line
        cumulative_var = pca_results['cumulative_variance'][:n_components]
        
        fig.add_trace(
            go.Scatter(
                x=components,
                y=cumulative_var,
                mode='lines+markers',
                marker=dict(size=8, color='red'),
                line=dict(width=3, color='red'),
                name='Cumulative Variance'
            ),
            row=1, col=2
        )
        
        # Add 80% and 95% reference lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="gray", 
                     annotation_text="80%", row=1, col=2)
        fig.add_hline(y=0.95, line_dash="dash", line_color="orange", 
                     annotation_text="95%", row=1, col=2)
        
        # 3. Loadings heatmap
        loadings = np.array(pca_results['loadings'])[:n_components, :]
        feature_names = pca_results['feature_names']
        
        fig.add_trace(
            go.Heatmap(
                z=loadings,
                x=feature_names,
                y=components,
                colorscale='RdBu',
                zmid=0,
                text=np.round(loadings, 2),
                texttemplate='%{text}',
                textfont=dict(size=10),
                showscale=True,
                name='Loadings'
            ),
            row=2, col=1
        )
        
        # 4. Biplot (if we have at least 2 components)
        if n_components >= 2:
            pc_data = np.array(pca_results['principal_components'])
            
            # Scatter plot of PC1 vs PC2
            fig.add_trace(
                go.Scatter(
                    x=pc_data[:, 0],
                    y=pc_data[:, 1],
                    mode='markers+text',
                    text=feature_names,  # Show all names on hover
                    hoverinfo='text',
                    marker=dict(size=8, color='purple', opacity=0.7),
                    textposition='top center',
                    textfont=dict(size=9),
                    # Only show text for a few outlier points to avoid clutter
                    selectedpoints=self._get_outlier_indices(pc_data[:, 0], pc_data[:, 1])
                ),
                row=2, col=2
            )
            
            # Add arrows for loadings
            for i, feature in enumerate(feature_names):
                fig.add_annotation(
                    ax=0, ay=0,
                    x=loadings[0, i] * np.max(pc_data[:, 0]), 
                    y=loadings[1, i] * np.max(pc_data[:, 1]),
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#636EFA",
                    axref="x5", ayref="y5",
                    xref="x5", yref="y5"
                )
        
        # Update layout for biplot
        fig.update_xaxes(title_text="PC1", row=2, col=2)
        fig.update_yaxes(title_text="PC2", row=2, col=2)
        
        fig.update_layout(
            height=1000,
            title_text=f"PCA Analysis - {n_components} Components",
            showlegend=False
        )
        
        return fig

    def _get_outlier_indices(self, x, y, n=10):
        """Identify indices of points furthest from the origin"""
        distances = np.sqrt(x**2 + y**2)
        if len(distances) <= n:
            return list(range(len(distances)))
        
        # Return indices of the n points with the largest distances
        return np.argsort(distances)[-n:]

    def perform_fft_analysis(self, 
                             df: pl.DataFrame, 
                             analysis_column: str,
                             window_length: int,
                             overlap_pct: float) -> Dict:
        """Perform FFT analysis on a given column of a DataFrame."""
        try:
            # Prepare data
            if analysis_column == 'Log Returns':
                series = np.log(df['price']).diff().drop_nans()
            elif analysis_column == 'Detrended Prices':
                from scipy.signal import detrend
                series = pl.Series(detrend(df['price'].to_numpy()))
            elif analysis_column == 'Volume':
                if 'volume' not in df.columns:
                    return {'status': 'error', 'reason': 'Volume column not available'}
                series = df['volume']
            else: # Prices
                series = df['price']

            series = series.fill_nan(0).to_numpy()
            
            if len(series) < window_length:
                return {'status': 'error', 'reason': 'Not enough data for the specified window length'}

            # Perform FFT on sliding windows
            step = int(window_length * (1 - overlap_pct / 100))
            all_freqs = []
            
            for i in range(0, len(series) - window_length + 1, step):
                window = series[i:i+window_length]
                
                # Apply Hanning window to reduce spectral leakage
                window = window * np.hanning(len(window))
                
                fft_result = np.fft.fft(window)
                fft_freq = np.fft.fftfreq(len(window))
                
                # Get positive frequencies
                positive_mask = fft_freq > 0
                all_freqs.append(pd.DataFrame({
                    'freq': fft_freq[positive_mask],
                    'amplitude': np.abs(fft_result[positive_mask])
                }))
            
            if not all_freqs:
                return {'status': 'error', 'reason': 'FFT analysis did not produce results'}

            # Average the frequencies across all windows
            avg_freqs = pd.concat(all_freqs).groupby('freq').amplitude.mean().reset_index()
            
            # Find dominant frequencies
            dominant_freqs = avg_freqs.nlargest(10, 'amplitude')
            dominant_freqs['period_minutes'] = 1 / dominant_freqs['freq']
            
            return {
                'status': 'success',
                'avg_spectrum': avg_freqs,
                'dominant_freqs': dominant_freqs
            }
            
        except Exception as e:
            return {'status': 'error', 'reason': str(e)}

    def create_fft_visualization(self, fft_results: Dict) -> go.Figure:
        """Create visualization for FFT analysis results."""
        if fft_results['status'] != 'success':
            return None

        avg_spectrum = fft_results['avg_spectrum']
        dominant_freqs = fft_results['dominant_freqs']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Full Frequency Spectrum', 'Top 10 Dominant Frequencies (Cycles)'),
            vertical_spacing=0.15
        )
        
        # Plot 1: Full spectrum
        fig.add_trace(go.Scatter(
            x=avg_spectrum['freq'],
            y=avg_spectrum['amplitude'],
            mode='lines',
            name='Spectrum',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Plot 2: Dominant frequencies as bar chart
        fig.add_trace(go.Bar(
            x=[f"{p:.1f} min" for p in dominant_freqs['period_minutes']],
            y=dominant_freqs['amplitude'],
            text=[f"{p:.1f} min" for p in dominant_freqs['period_minutes']],
            textposition='outside',
            name='Dominant Cycles'
        ), row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="FFT Analysis: Cyclical Pattern Detection",
            showlegend=False
        )
        fig.update_xaxes(title_text="Frequency (cycles/minute)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Period of Dominant Cycle", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        
        return fig

    def create_correlation_network(self, correlation_matrix: pl.DataFrame, threshold: float = 0.5) -> go.Figure:
        """Create an interactive network graph of token correlations."""

        # Get token names and correlation values
        token_names = correlation_matrix['token'].to_list()
        corr_values = correlation_matrix.select([col for col in correlation_matrix.columns if col != 'token']).to_numpy()
        
        # Build the graph
        G = nx.Graph()
        for i, token_name in enumerate(token_names):
            G.add_node(token_name)
            
        for i in range(len(token_names)):
            for j in range(i + 1, len(token_names)):
                correlation = corr_values[i, j]
                if abs(correlation) >= threshold:
                    G.add_edge(token_names[i], token_names[j], weight=correlation)
                    
        if G.number_of_edges() == 0:
            return go.Figure(layout=dict(title='No correlations above threshold to display.'))

        # Get positions for the nodes
        pos = nx.spring_layout(G, k=0.8, iterations=50)

        # Create edge traces
        edge_x, edge_y = [], []
        edge_colors, edge_widths = [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = edge[2]['weight']
            edge_colors.append('red' if weight < 0 else 'green')
            edge_widths.append(1 + (abs(weight) - threshold) * 10)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
            
        # Create node traces
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=15,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        # Color nodes by number of connections
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Token Correlation Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text=f"Correlations > {threshold}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig


def load_tokens_for_correlation(data_paths: List[Path], 
                              limit: Optional[int] = None) -> Dict[str, pl.DataFrame]:
    """Load multiple token datasets for correlation analysis"""
    
    token_data = {}
    files_to_process = data_paths[:limit] if limit else data_paths
    
    print(f"Loading {len(files_to_process)} tokens for correlation analysis...")
    
    for path in files_to_process:
        try:
            token_name = path.stem
            df = pl.read_parquet(path)
            
            # Ensure required columns
            if 'price' in df.columns and 'datetime' in df.columns:
                # Basic filtering
                if len(df) >= 60:  # At least 1 hour of data
                    token_data[token_name] = df
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    print(f"Successfully loaded {len(token_data)} tokens")
    return token_data
