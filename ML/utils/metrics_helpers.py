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