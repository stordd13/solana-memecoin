"""
Shared utilities for hyperparameter tuning across all models
"""

import json
import optuna
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def save_best_params(best_params: Dict[str, Any], study: optuna.Study, 
                     model_name: str, results_dir: Path):
    """Save best parameters and study results to files"""
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters as JSON
    params_file = results_dir / f'{model_name}_best_params.json'
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Best parameters saved to: {params_file}")
    
    # Save study results as CSV for analysis
    df = study.trials_dataframe()
    study_file = results_dir / f'{model_name}_tuning_trials.csv'
    df.to_csv(study_file, index=False)
    print(f"📊 Study trials saved to: {study_file}")
    
    # Save study object for later analysis
    study_pickle = results_dir / f'{model_name}_study.pkl'
    optuna.storages.serialize.save_study(study, study_pickle)
    print(f"🔬 Study object saved to: {study_pickle}")


def load_best_params(model_name: str, results_dir: Path) -> Dict[str, Any]:
    """Load best parameters from JSON file"""
    
    params_file = results_dir / f'{model_name}_best_params.json'
    
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"✅ Loaded best parameters from: {params_file}")
        return params
    else:
        print(f"❌ No saved parameters found at: {params_file}")
        return {}


def create_tuning_visualizations(study: optuna.Study, model_name: str, 
                                results_dir: Path):
    """Create comprehensive visualizations for hyperparameter tuning results"""
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Optimization history
    fig_history = go.Figure()
    
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    
    # Running best
    running_best = []
    best_so_far = float('-inf')
    for val in values:
        if val > best_so_far:
            best_so_far = val
        running_best.append(best_so_far)
    
    fig_history.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='markers',
        name='Trial Values',
        marker=dict(color='lightblue', size=6)
    ))
    
    fig_history.add_trace(go.Scatter(
        x=list(range(len(running_best))),
        y=running_best,
        mode='lines',
        name='Best So Far',
        line=dict(color='red', width=2)
    ))
    
    fig_history.update_layout(
        title=f'{model_name} - Optimization History',
        xaxis_title='Trial',
        yaxis_title='Objective Value',
        template='plotly_white'
    )
    
    history_file = results_dir / f'{model_name}_optimization_history.html'
    fig_history.write_html(history_file)
    print(f"📈 Optimization history saved to: {history_file}")
    
    # 2. Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        
        fig_importance = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            marker_color='lightgreen'
        ))
        
        fig_importance.update_layout(
            title=f'{model_name} - Parameter Importance',
            xaxis_title='Importance',
            yaxis_title='Parameter',
            template='plotly_white'
        )
        
        importance_file = results_dir / f'{model_name}_parameter_importance.html'
        fig_importance.write_html(importance_file)
        print(f"🎯 Parameter importance saved to: {importance_file}")
        
    except Exception as e:
        print(f"⚠️ Could not create parameter importance plot: {e}")
    
    # 3. Parallel coordinate plot for top trials
    try:
        df = study.trials_dataframe()
        df_complete = df[df['state'] == 'COMPLETE'].copy()
        
        if len(df_complete) > 20:
            # Show top 20 trials
            df_top = df_complete.nlargest(20, 'value')
        else:
            df_top = df_complete
        
        # Get parameter columns
        param_cols = [col for col in df_top.columns if col.startswith('params_')]
        
        if param_cols:
            fig_parallel = go.Figure(data=go.Parcoords(
                line=dict(color=df_top['value'],
                         colorscale='Viridis',
                         showscale=True,
                         colorbar=dict(title="Objective Value")),
                dimensions=[
                    dict(range=[df_top[col].min(), df_top[col].max()],
                         label=col.replace('params_', ''),
                         values=df_top[col])
                    for col in param_cols
                ]
            ))
            
            fig_parallel.update_layout(
                title=f'{model_name} - Parameter Relationships (Top Trials)',
                template='plotly_white'
            )
            
            parallel_file = results_dir / f'{model_name}_parameter_parallel.html'
            fig_parallel.write_html(parallel_file)
            print(f"🕸️ Parallel coordinate plot saved to: {parallel_file}")
    
    except Exception as e:
        print(f"⚠️ Could not create parallel coordinate plot: {e}")


def print_tuning_summary(study: optuna.Study, model_name: str):
    """Print a comprehensive summary of tuning results"""
    
    print(f"\n{'='*60}")
    print(f"🎯 HYPERPARAMETER TUNING SUMMARY - {model_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📊 Study Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if study.best_trial:
        print(f"\n🏆 Best Trial:")
        print(f"  Trial number: {study.best_trial.number}")
        print(f"  Objective value: {study.best_value:.6f}")
        
        print(f"\n⚙️ Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        if study.best_trial.user_attrs:
            print(f"\n📈 Best Trial Metrics:")
            for key, value in study.best_trial.user_attrs.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")


def get_default_optuna_config():
    """Get default Optuna configuration for consistent tuning"""
    return {
        'sampler': optuna.samplers.TPESampler(seed=42),
        'pruner': optuna.pruners.MedianPruner(
            n_startup_trials=10, 
            n_warmup_steps=10
        ),
        'direction': 'maximize'
    }


class TuningProgressCallback:
    """Custom callback to track tuning progress"""
    
    def __init__(self, model_name: str, save_interval: int = 20):
        self.model_name = model_name
        self.save_interval = save_interval
        self.trial_count = 0
        
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        self.trial_count += 1
        
        if self.trial_count % self.save_interval == 0:
            print(f"\n📊 Progress Update - {self.model_name}")
            print(f"  Completed {self.trial_count} trials")
            if study.best_trial:
                print(f"  Current best value: {study.best_value:.6f}")
                print(f"  Best trial: #{study.best_trial.number}")