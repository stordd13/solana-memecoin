"""
Winsorizer for ML feature scaling - Better suited for cryptocurrency data
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorize features by capping extreme values at specified percentiles.
    Better than RobustScaler for cryptocurrency data with extreme outliers.
    
    Parameters
    ----------
    limits : tuple, default=(0.005, 0.995)
        Lower and upper percentiles for capping (0.5%, 99.5%)
    """
    
    def __init__(self, limits=(0.005, 0.995)):
        self.limits = limits
        # Backward compatibility
        self.lower_percentile = limits[0] 
        self.upper_percentile = limits[1]
        
    def fit(self, X, y=None):
        """Compute the percentile bounds for winsorization"""
        X = check_array(X, accept_sparse=False, ensure_all_finite='allow-nan')
        
        n_features = X.shape[1]
        self.lower_bounds_ = np.zeros(n_features)
        self.upper_bounds_ = np.zeros(n_features)
        
        # Create limits_ attribute for compatibility with tests
        self.limits_ = []
        
        for i in range(n_features):
            col = X[:, i]
            # Handle NaN values
            finite_mask = np.isfinite(col)
            if finite_mask.sum() > 0:
                finite_col = col[finite_mask]
                lower_bound = np.percentile(finite_col, self.lower_percentile * 100)
                upper_bound = np.percentile(finite_col, self.upper_percentile * 100)
                self.lower_bounds_[i] = lower_bound
                self.upper_bounds_[i] = upper_bound
                self.limits_.append((lower_bound, upper_bound))
            else:
                # If all NaN, set bounds to 0
                self.lower_bounds_[i] = 0
                self.upper_bounds_[i] = 0
                self.limits_.append((0, 0))
                
        return self
        
    def transform(self, X):
        """Apply winsorization to the features"""
        check_is_fitted(self, ['lower_bounds_', 'upper_bounds_'])
        X = check_array(X, accept_sparse=False, ensure_all_finite='allow-nan', copy=True)
        
        n_features = X.shape[1]
        
        for i in range(n_features):
            col = X[:, i]
            # Only transform finite values
            finite_mask = np.isfinite(col)
            if finite_mask.sum() > 0:
                # Cap values at bounds
                col[finite_mask & (col < self.lower_bounds_[i])] = self.lower_bounds_[i]
                col[finite_mask & (col > self.upper_bounds_[i])] = self.upper_bounds_[i]
                
        return X
        
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class RollingWinsorizer(BaseEstimator, TransformerMixin):
    """
    Per-token winsorizer that fits separately on each token's training data.
    This prevents data leakage across tokens while handling crypto volatility.
    """
    
    def __init__(self, lower_percentile=0.005, upper_percentile=0.995):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.token_winsorizers = {}
        
    def fit_token(self, X, token_id):
        """Fit winsorizer for a specific token"""
        winsorizer = Winsorizer(self.lower_percentile, self.upper_percentile)
        winsorizer.fit(X)
        self.token_winsorizers[token_id] = winsorizer
        return self
        
    def transform_token(self, X, token_id):
        """Transform features for a specific token"""
        if token_id not in self.token_winsorizers:
            # If we haven't seen this token, return unchanged
            return X
        return self.token_winsorizers[token_id].transform(X) 