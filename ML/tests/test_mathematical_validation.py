"""
Mathematical validation tests for ML module.
Ensures all metrics, model calculations, and financial computations are mathematically correct.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from ML.utils.metrics_helpers import (
    calculate_classification_metrics,
    calculate_financial_metrics, 
    calculate_regression_metrics
)
from ML.utils.winsorizer import Winsorizer


@pytest.mark.unit
@pytest.mark.mathematical
class TestClassificationMetricsCalculation:
    """Test classification metrics calculation mathematical correctness."""
    
    def test_accuracy_calculation_precision(self, ml_metrics_references):
        """Test accuracy calculation against sklearn reference."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_pred = ref_data['y_pred']
        expected_accuracy = ref_data['expected_metrics']['accuracy']
        
        # Test our implementation
        calculated_accuracy = calculate_classification_metrics(y_true, y_pred)['accuracy']
        
        assert abs(calculated_accuracy - expected_accuracy) < 1e-12, \
            f"Accuracy calculation mismatch: {calculated_accuracy} vs {expected_accuracy}"
    
    def test_precision_calculation_precision(self, ml_metrics_references):
        """Test precision calculation against sklearn reference."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_pred = ref_data['y_pred']
        expected_precision = ref_data['expected_metrics']['precision']
        
        calculated_precision = calculate_classification_metrics(y_true, y_pred)['precision']
        
        assert abs(calculated_precision - expected_precision) < 1e-12, \
            f"Precision calculation mismatch: {calculated_precision} vs {expected_precision}"
    
    def test_recall_calculation_precision(self, ml_metrics_references):
        """Test recall calculation against sklearn reference."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_pred = ref_data['y_pred']
        expected_recall = ref_data['expected_metrics']['recall']
        
        calculated_recall = calculate_classification_metrics(y_true, y_pred)['recall']
        
        assert abs(calculated_recall - expected_recall) < 1e-12, \
            f"Recall calculation mismatch: {calculated_recall} vs {expected_recall}"
    
    def test_f1_score_calculation_precision(self, ml_metrics_references):
        """Test F1 score calculation against sklearn reference."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_pred = ref_data['y_pred']
        expected_f1 = ref_data['expected_metrics']['f1']
        
        calculated_f1 = calculate_classification_metrics(y_true, y_pred)['f1']
        
        assert abs(calculated_f1 - expected_f1) < 1e-12, \
            f"F1 score calculation mismatch: {calculated_f1} vs {expected_f1}"
    
    def test_roc_auc_calculation_precision(self, ml_metrics_references):
        """Test ROC AUC calculation against sklearn reference."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_proba = ref_data['y_proba']
        expected_roc_auc = ref_data['expected_metrics']['roc_auc']
        
        calculated_roc_auc = calculate_classification_metrics(y_true, None, y_proba)['roc_auc']
        
        assert abs(calculated_roc_auc - expected_roc_auc) < 1e-12, \
            f"ROC AUC calculation mismatch: {calculated_roc_auc} vs {expected_roc_auc}"
    
    def test_confusion_matrix_properties(self, ml_metrics_references):
        """Test confusion matrix mathematical properties."""
        ref_data = ml_metrics_references
        y_true = ref_data['y_true']
        y_pred = ref_data['y_pred']
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        # Confusion matrix elements
        tp = metrics['confusion_matrix']['true_positive']
        tn = metrics['confusion_matrix']['true_negative']
        fp = metrics['confusion_matrix']['false_positive']
        fn = metrics['confusion_matrix']['false_negative']
        
        # Test mathematical properties
        total_samples = tp + tn + fp + fn
        assert total_samples == len(y_true), "Confusion matrix should sum to total samples"
        
        # Verify accuracy calculation from confusion matrix
        accuracy_from_cm = (tp + tn) / total_samples
        assert abs(accuracy_from_cm - metrics['accuracy']) < 1e-12, \
            "Accuracy from confusion matrix should match direct calculation"
        
        # Verify precision calculation from confusion matrix
        if tp + fp > 0:
            precision_from_cm = tp / (tp + fp)
            assert abs(precision_from_cm - metrics['precision']) < 1e-12, \
                "Precision from confusion matrix should match direct calculation"
        
        # Verify recall calculation from confusion matrix  
        if tp + fn > 0:
            recall_from_cm = tp / (tp + fn)
            assert abs(recall_from_cm - metrics['recall']) < 1e-12, \
                "Recall from confusion matrix should match direct calculation"


@pytest.mark.unit
@pytest.mark.mathematical
class TestFinancialMetricsCalculation:
    """Test financial metrics calculation mathematical correctness."""
    
    def test_sharpe_ratio_calculation_accuracy(self, financial_metrics_references):
        """Test Sharpe ratio calculation against reference implementation."""
        ref_data = financial_metrics_references
        returns = ref_data['returns']
        risk_free_rate = ref_data['risk_free_rate']
        expected_sharpe = ref_data['expected_metrics']['sharpe_ratio']
        
        calculated_sharpe = calculate_financial_metrics(returns, risk_free_rate)['sharpe_ratio']
        
        assert abs(calculated_sharpe - expected_sharpe) < 1e-12, \
            f"Sharpe ratio calculation mismatch: {calculated_sharpe} vs {expected_sharpe}"
    
    def test_sortino_ratio_calculation_accuracy(self, financial_metrics_references):
        """Test Sortino ratio calculation against reference implementation."""
        ref_data = financial_metrics_references
        returns = ref_data['returns']
        risk_free_rate = ref_data['risk_free_rate']
        expected_sortino = ref_data['expected_metrics']['sortino_ratio']
        
        calculated_sortino = calculate_financial_metrics(returns, risk_free_rate)['sortino_ratio']
        
        assert abs(calculated_sortino - expected_sortino) < 1e-12, \
            f"Sortino ratio calculation mismatch: {calculated_sortino} vs {expected_sortino}"
    
    def test_maximum_drawdown_calculation_accuracy(self, financial_metrics_references):
        """Test maximum drawdown calculation against reference implementation."""
        ref_data = financial_metrics_references
        returns = ref_data['returns']
        expected_max_dd = ref_data['expected_metrics']['max_drawdown']
        
        calculated_max_dd = calculate_financial_metrics(returns)['max_drawdown']
        
        assert abs(calculated_max_dd - expected_max_dd) < 1e-12, \
            f"Maximum drawdown calculation mismatch: {calculated_max_dd} vs {expected_max_dd}"
    
    def test_calmar_ratio_calculation_accuracy(self, financial_metrics_references):
        """Test Calmar ratio calculation against reference implementation."""
        ref_data = financial_metrics_references
        returns = ref_data['returns']
        risk_free_rate = ref_data['risk_free_rate']
        expected_calmar = ref_data['expected_metrics']['calmar_ratio']
        
        calculated_calmar = calculate_financial_metrics(returns, risk_free_rate)['calmar_ratio']
        
        assert abs(calculated_calmar - expected_calmar) < 1e-12, \
            f"Calmar ratio calculation mismatch: {calculated_calmar} vs {expected_calmar}"
    
    def test_volatility_calculation_accuracy(self, financial_metrics_references):
        """Test volatility calculation consistency."""
        ref_data = financial_metrics_references
        returns = ref_data['returns']
        expected_volatility = ref_data['expected_metrics']['volatility']
        
        calculated_volatility = calculate_financial_metrics(returns)['volatility']
        
        assert abs(calculated_volatility - expected_volatility) < 1e-12, \
            f"Volatility calculation mismatch: {calculated_volatility} vs {expected_volatility}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestRegressionMetricsCalculation:
    """Test regression metrics calculation mathematical correctness."""
    
    def test_r2_calculation_accuracy(self, ml_reference_data):
        """Test R² calculation against sklearn reference."""
        data = ml_reference_data
        y_true = data['y_regression'][:100]  # Take subset
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add small noise
        
        expected_r2 = r2_score(y_true, y_pred)
        calculated_r2 = calculate_regression_metrics(y_true, y_pred)['r2']
        
        assert abs(calculated_r2 - expected_r2) < 1e-12, \
            f"R² calculation mismatch: {calculated_r2} vs {expected_r2}"
    
    def test_mae_calculation_accuracy(self, ml_reference_data):
        """Test MAE calculation against sklearn reference."""
        data = ml_reference_data
        y_true = data['y_regression'][:100]
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        
        expected_mae = mean_absolute_error(y_true, y_pred)
        calculated_mae = calculate_regression_metrics(y_true, y_pred)['mae']
        
        assert abs(calculated_mae - expected_mae) < 1e-12, \
            f"MAE calculation mismatch: {calculated_mae} vs {expected_mae}"
    
    def test_rmse_calculation_accuracy(self, ml_reference_data):
        """Test RMSE calculation against sklearn reference."""
        data = ml_reference_data
        y_true = data['y_regression'][:100]
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        
        expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        calculated_rmse = calculate_regression_metrics(y_true, y_pred)['rmse']
        
        assert abs(calculated_rmse - expected_rmse) < 1e-12, \
            f"RMSE calculation mismatch: {calculated_rmse} vs {expected_rmse}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestWinsorizationCalculation:
    """Test winsorization calculation mathematical correctness."""
    
    def test_winsorization_accuracy(self, winsorization_reference_data):
        """Test winsorization against reference implementation."""
        ref_data = winsorization_reference_data
        original_data = ref_data['original_data']
        expected_winsorized = ref_data['expected_winsorized']
        
        winsorizer = Winsorizer(limits=(0.05, 0.95))
        winsorizer.fit(original_data.reshape(-1, 1))
        calculated_winsorized = winsorizer.transform(original_data.reshape(-1, 1)).flatten()
        
        assert np.allclose(calculated_winsorized, expected_winsorized, atol=1e-12), \
            "Winsorization calculation should match reference implementation"
    
    def test_winsorization_boundary_values(self, winsorization_reference_data):
        """Test winsorization boundary value accuracy."""
        ref_data = winsorization_reference_data
        original_data = ref_data['original_data']
        
        winsorizer = Winsorizer(limits=(0.05, 0.95))
        winsorizer.fit(original_data.reshape(-1, 1))
        
        # Check that computed percentiles match reference
        p5_calculated = winsorizer.limits_[0][0]
        p95_calculated = winsorizer.limits_[0][1]
        
        assert abs(p5_calculated - ref_data['percentile_5']) < 1e-12, \
            f"5th percentile mismatch: {p5_calculated} vs {ref_data['percentile_5']}"
        
        assert abs(p95_calculated - ref_data['percentile_95']) < 1e-12, \
            f"95th percentile mismatch: {p95_calculated} vs {ref_data['percentile_95']}"
    
    def test_winsorization_no_clipping_case(self):
        """Test winsorization when no clipping is needed."""
        # Data within normal range
        normal_data = np.random.normal(0, 1, 100)
        
        winsorizer = Winsorizer(limits=(0.01, 0.99))  # Very wide limits
        winsorizer.fit(normal_data.reshape(-1, 1))
        transformed = winsorizer.transform(normal_data.reshape(-1, 1)).flatten()
        
        # Should be very close to original (minimal clipping expected)
        assert np.allclose(transformed, normal_data, atol=1e-10), \
            "Winsorization with wide limits should barely change normal data"


@pytest.mark.unit
@pytest.mark.mathematical  
class TestScalingAccuracy:
    """Test scaling accuracy and mathematical properties."""
    
    def test_robust_scaler_accuracy(self, ml_reference_data):
        """Test RobustScaler mathematical properties."""
        data = ml_reference_data
        X = data['X']
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check that median is approximately 0 for each feature
        medians = np.median(X_scaled, axis=0)
        assert np.allclose(medians, 0, atol=1e-10), \
            "RobustScaler should center data around median (0)"
        
        # Check that IQR-based scaling is applied
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            q75, q25 = np.percentile(feature_data, [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:  # Avoid division by zero
                expected_scale = 1 / iqr
                actual_scale = scaler.scale_[i]
                assert abs(actual_scale - expected_scale) < 1e-12, \
                    f"RobustScaler scale factor mismatch for feature {i}"
    
    def test_scaling_inverse_transform_accuracy(self, ml_reference_data):
        """Test scaling inverse transform mathematical correctness."""
        data = ml_reference_data
        X = data['X']
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        
        # Inverse transform should return to original values
        assert np.allclose(X, X_inverse, atol=1e-12), \
            "Inverse transform should exactly recover original data"


@pytest.mark.integration
@pytest.mark.mathematical
class TestMLPipelineIntegration:
    """Test ML pipeline integration and mathematical consistency."""
    
    def test_temporal_split_validation(self, temporal_safety_data):
        """Test that temporal splits maintain mathematical properties."""
        tokens = temporal_safety_data
        
        for token_name, token_data in tokens.items():
            df = token_data.to_pandas()
            
            # Split temporal data (60% train, 20% val, 20% test)
            n = len(df)
            train_end = int(0.6 * n)
            val_end = int(0.8 * n)
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Test that temporal order is preserved
            assert train_df['datetime'].iloc[-1] <= val_df['datetime'].iloc[0], \
                "Training data should end before validation data starts"
            
            assert val_df['datetime'].iloc[-1] <= test_df['datetime'].iloc[0], \
                "Validation data should end before test data starts"
            
            # Test that there are no overlapping timestamps
            train_times = set(train_df['datetime'])
            val_times = set(val_df['datetime']) 
            test_times = set(test_df['datetime'])
            
            assert len(train_times & val_times) == 0, "No overlap between train and validation"
            assert len(train_times & test_times) == 0, "No overlap between train and test"
            assert len(val_times & test_times) == 0, "No overlap between validation and test"
    
    def test_per_token_scaling_independence(self, temporal_safety_data):
        """Test that per-token scaling maintains independence."""
        tokens = temporal_safety_data
        
        scalers = {}
        scaled_data = {}
        
        for token_name, token_data in tokens.items():
            df = token_data.to_pandas()
            
            # Fit scaler on each token independently
            scaler = RobustScaler()
            returns = df['returns'].dropna().values.reshape(-1, 1)
            
            if len(returns) > 0:
                scaler.fit(returns)
                scaled_returns = scaler.transform(returns)
                
                scalers[token_name] = scaler
                scaled_data[token_name] = scaled_returns
        
        # Test that scalers are indeed different for different tokens
        scaler_names = list(scalers.keys())
        if len(scaler_names) > 1:
            scaler1 = scalers[scaler_names[0]]
            scaler2 = scalers[scaler_names[1]]
            
            # Scalers should have different parameters (unless by coincidence)
            # This is a reasonable expectation for different financial time series
            scale_diff = abs(scaler1.scale_[0] - scaler2.scale_[0])
            center_diff = abs(scaler1.center_[0] - scaler2.center_[0])
            
            # At least one should be different (allowing for small coincidences)
            assert scale_diff > 1e-10 or center_diff > 1e-10, \
                "Independent scalers should have different parameters"