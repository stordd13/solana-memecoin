from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np


def classification_metrics(y_true, y_pred, y_prob=None):
    """Return standard classification metrics as a dict."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def financial_classification_metrics(y_true, y_pred, returns, y_prob=None):
    """
    Return both standard and financial-weighted classification metrics.
    
    Args:
        y_true: Binary labels (0/1)
        y_pred: Binary predictions (0/1) 
        returns: Actual return values (e.g., price change percentages)
        y_prob: Prediction probabilities (optional)
    
    Returns:
        Dictionary with standard + financial-weighted metrics
    """
    # Start with standard metrics
    metrics = classification_metrics(y_true, y_pred, y_prob)
    
    # Convert to numpy arrays for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    returns = np.array(returns)
    
    # ADDED: Winsorize returns to handle extreme memecoin gains/losses
    # Cap at 0.5% and 99.5% percentiles to remove outliers
    if len(returns) > 0:
        lower_bound = np.percentile(returns, 0.5)
        upper_bound = np.percentile(returns, 99.5)
        returns_winsorized = np.clip(returns, lower_bound, upper_bound)
        
        # Store both for comparison
        metrics['returns_winsorized_applied'] = True
        metrics['returns_raw_min'] = float(np.min(returns))
        metrics['returns_raw_max'] = float(np.max(returns))
        metrics['returns_winsorized_min'] = float(np.min(returns_winsorized))
        metrics['returns_winsorized_max'] = float(np.max(returns_winsorized))
    else:
        returns_winsorized = returns
        metrics['returns_winsorized_applied'] = False
    
    # Use winsorized returns for financial calculations
    returns = returns_winsorized
    
    # Financial-weighted metrics
    try:
        # Calculate return-weighted recall
        # Standard recall: TP / (TP + FN)
        # Return-weighted recall: Sum(returns for TP) / Sum(returns for all positives)
        
        true_positives_mask = (y_true == 1) & (y_pred == 1)
        all_positives_mask = (y_true == 1)
        
        if np.sum(all_positives_mask) > 0:
            # Sum of returns for true positives
            tp_returns_sum = np.sum(returns[true_positives_mask])
            # Sum of returns for all actual positives
            all_pos_returns_sum = np.sum(returns[all_positives_mask])
            
            if all_pos_returns_sum > 0:
                recall_by_return = tp_returns_sum / all_pos_returns_sum
            else:
                # If all positive returns sum to 0 or negative, use standard recall
                recall_by_return = metrics['recall']
        else:
            recall_by_return = 0.0
        
        # Calculate return-weighted precision (bonus metric)
        # Sum(returns for TP) / Sum(returns for all predictions = 1)
        all_predicted_positives_mask = (y_pred == 1)
        
        if np.sum(all_predicted_positives_mask) > 0:
            pred_pos_returns_sum = np.sum(returns[all_predicted_positives_mask])
            if pred_pos_returns_sum > 0:
                precision_by_return = tp_returns_sum / pred_pos_returns_sum
            else:
                precision_by_return = 0.0
        else:
            precision_by_return = 0.0
        
        # Hybrid F1: precision by count + recall by return
        precision_by_count = metrics['precision']
        if (precision_by_count + recall_by_return) > 0:
            hybrid_f1 = 2 * (precision_by_count * recall_by_return) / (precision_by_count + recall_by_return)
        else:
            hybrid_f1 = 0.0
        
        # Alternative Hybrid F1: precision by return + recall by count
        recall_by_count = metrics['recall']
        if (precision_by_return + recall_by_count) > 0:
            hybrid_f1_alt = 2 * (precision_by_return * recall_by_count) / (precision_by_return + recall_by_count)
        else:
            hybrid_f1_alt = 0.0
        
        # Add financial metrics
        metrics.update({
            'recall_by_return': recall_by_return,
            'precision_by_return': precision_by_return,
            'hybrid_f1_precision_count_recall_return': hybrid_f1,
            'hybrid_f1_precision_return_recall_count': hybrid_f1_alt,
            'total_positive_returns': all_pos_returns_sum,
            'captured_positive_returns': tp_returns_sum,
            'return_capture_rate': recall_by_return  # Alias for clarity
        })
        
        # Calculate average return per prediction type
        if np.sum(true_positives_mask) > 0:
            metrics['avg_return_per_tp'] = np.mean(returns[true_positives_mask])
        else:
            metrics['avg_return_per_tp'] = 0.0
            
        false_positives_mask = (y_true == 0) & (y_pred == 1)
        if np.sum(false_positives_mask) > 0:
            metrics['avg_return_per_fp'] = np.mean(returns[false_positives_mask])
        else:
            metrics['avg_return_per_fp'] = 0.0
        
        # Economic value metrics
        metrics['economic_precision'] = metrics['avg_return_per_tp'] if metrics['avg_return_per_tp'] > 0 else 0
        
        # Risk-adjusted metrics
        if np.sum(all_predicted_positives_mask) > 0:
            predicted_returns = returns[all_predicted_positives_mask]
            metrics['prediction_return_volatility'] = np.std(predicted_returns)
            if metrics['prediction_return_volatility'] > 0:
                metrics['prediction_sharpe'] = np.mean(predicted_returns) / metrics['prediction_return_volatility']
            else:
                metrics['prediction_sharpe'] = 0.0
        else:
            metrics['prediction_return_volatility'] = 0.0
            metrics['prediction_sharpe'] = 0.0
            
    except Exception as e:
        # If financial metrics fail, add zeros to avoid breaking
        metrics.update({
            'recall_by_return': 0.0,
            'precision_by_return': 0.0,
            'hybrid_f1_precision_count_recall_return': 0.0,
            'hybrid_f1_precision_return_recall_count': 0.0,
            'total_positive_returns': 0.0,
            'captured_positive_returns': 0.0,
            'return_capture_rate': 0.0,
            'avg_return_per_tp': 0.0,
            'avg_return_per_fp': 0.0,
            'economic_precision': 0.0,
            'prediction_return_volatility': 0.0,
            'prediction_sharpe': 0.0,
            'financial_metrics_error': str(e)
        })
    
    return metrics


def regression_metrics(y_true, y_pred):
    """Return common regression metrics as a dict."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred)
    }


def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics with mathematical validation.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels  
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary with classification metrics and confusion matrix
    """
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics using exact formulas
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        }
    }
    
    # Add ROC AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5  # Default for edge cases
    
    return metrics


def calculate_financial_metrics(returns, risk_free_rate=0.02):
    """
    Calculate financial performance metrics with mathematical validation.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Dictionary with financial metrics
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return {
            'mean_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }
    
    # Basic statistics
    mean_return = np.mean(returns)
    volatility = np.std(returns, ddof=1)
    
    # Risk-adjusted metrics
    daily_rf_rate = risk_free_rate / 252  # Convert annual to daily
    excess_returns = returns - daily_rf_rate
    mean_excess_return = np.mean(excess_returns)
    
    # Sharpe ratio
    sharpe_ratio = mean_excess_return / volatility if volatility > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_deviation = np.std(downside_returns, ddof=1)
    else:
        downside_deviation = volatility
    
    sortino_ratio = mean_excess_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    annualized_return = mean_return * 252
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics with mathematical validation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with regression metrics
    """
    # Calculate metrics using exact formulas
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R² calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }