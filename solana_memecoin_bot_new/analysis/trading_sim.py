# analysis/trading_sim.py - Unified trading simulation (RL + Transformer forecasts)
import polars as pl
import numpy as np
import torch
import os
import sys
from typing import Dict, Optional
from stable_baselines3 import PPO

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils import setup_logger, load_unified_models, get_unified_features, validate_feature_compatibility
from ml.rl_agent import UnifiedMemecoinEnv, evaluate_agent
import config

logger = setup_logger(__name__)

def simulate_unified_trading(df: pl.DataFrame, unified_models: Dict, rl_agent: PPO, 
                           num_tokens: Optional[int] = None, strategy: str = "hybrid") -> Dict:
    """
    Simulate trading using unified models and RL agent.
    
    Args:
        df: Unified dataset
        unified_models: Dictionary containing transformer and baseline models
        rl_agent: Trained PPO agent
        num_tokens: Number of tokens to simulate (None for all)
        strategy: Trading strategy ('hybrid', 'rl_only', 'transformer_only')
        
    Returns:
        Dictionary with simulation results and metrics
    """
    logger.info(f"Starting unified trading simulation with strategy: {strategy}")
    
    # Use test data for simulation
    test_df = df.filter(pl.col("split") == "test")
    tokens = test_df["token_id"].unique().to_list()
    
    if num_tokens:
        tokens = tokens[:num_tokens]
    
    logger.info(f"Simulating trades on {len(tokens)} tokens")
    
    all_trades = []
    all_profits = []
    token_results = []
    
    for i, token_id in enumerate(tokens):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing token {i + 1}/{len(tokens)}: {token_id}")
        
        token_df = test_df.filter(pl.col("token_id") == token_id).sort("datetime")
        
        if len(token_df) < 20:  # Skip very short sequences
            continue
        
        # Simulate trading for this token
        token_result = simulate_token_trading(token_df, unified_models, rl_agent, strategy)
        
        if token_result:
            token_results.append(token_result)
            all_trades.extend(token_result['trades'])
            all_profits.extend(token_result['profits'])
    
    # Calculate overall metrics
    metrics = calculate_trading_metrics(all_profits, all_trades)
    
    # Save results
    results = {
        'strategy': strategy,
        'num_tokens': len(tokens),
        'tokens_traded': len(token_results),
        'total_trades': len(all_trades),
        'metrics': metrics,
        'token_results': token_results,
        'profits': all_profits
    }
    
    # Log summary
    logger.info(f"\nTrading Simulation Results ({strategy}):")
    logger.info(f"Tokens processed: {len(tokens)}")
    logger.info(f"Tokens with trades: {len(token_results)}")
    logger.info(f"Total trades: {len(all_trades)}")
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    logger.info(f"Avg profit per trade: ${metrics['avg_profit_per_trade']:.2f}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Max drawdown: ${metrics['max_drawdown']:.2f}")
    
    return results

def simulate_token_trading(token_df: pl.DataFrame, unified_models: Dict, rl_agent: PPO, 
                          strategy: str) -> Optional[Dict]:
    """
    Simulate trading for a single token.
    
    Args:
        token_df: Token data
        unified_models: Models dictionary
        rl_agent: RL agent
        strategy: Trading strategy
        
    Returns:
        Dictionary with token trading results
    """
    token_id = token_df["token_id"][0]
    
    # Initialize trading state
    portfolio = 1000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    profits = []
    
    # Get transformer model
    transformer = unified_models.get('transformer', {}).get('model')
    window_size = unified_models.get('transformer', {}).get('window_size', 20)
    features = get_unified_features()
    
    # Validate features
    compatible, missing = validate_feature_compatibility(token_df.columns, features)
    if not compatible:
        logger.debug(f"Token {token_id}: Missing features {missing}, using available only")
        features = [f for f in features if f in token_df.columns]
    
    # Convert to numpy for faster access
    token_data = token_df.to_numpy()
    columns = token_df.columns
    
    for step in range(len(token_data)):
        row = token_data[step]
        price = row[columns.index("avg_price")]
        
        # Get predictions based on strategy
        action = 0  # Default: hold
        
        if strategy == "rl_only":
            # Use RL agent only
            env = UnifiedMemecoinEnv(token_df, unified_models, token_id=token_id)
            env.current_step = step
            env.position = position
            env.portfolio = portfolio
            env.entry_price = entry_price
            
            state = env._get_state()
            action, _ = rl_agent.predict(state, deterministic=True)
            
        elif strategy == "transformer_only":
            # Use transformer predictions only
            forecast = get_transformer_prediction(token_data, columns, features, transformer, 
                                                step, window_size)
            
            # Simple threshold-based actions
            if forecast > 0.7 and position == 0:
                action = 1  # Buy
            elif forecast < 0.3 and position > 0:
                action = 2  # Sell
            else:
                action = 0  # Hold
                
        elif strategy == "hybrid":
            # Combine RL and transformer
            env = UnifiedMemecoinEnv(token_df, unified_models, token_id=token_id)
            env.current_step = step
            env.position = position
            env.portfolio = portfolio
            env.entry_price = entry_price
            
            state = env._get_state()
            rl_action, _ = rl_agent.predict(state, deterministic=True)
            
            forecast = get_transformer_prediction(token_data, columns, features, transformer, 
                                                step, window_size)
            
            # Override RL action if transformer strongly disagrees
            if rl_action == 1 and forecast < 0.2:  # RL wants to buy but transformer says no
                action = 0
            elif rl_action == 2 and forecast > 0.8:  # RL wants to sell but transformer says hold
                action = 0
            else:
                action = rl_action
        
        # Execute action
        if action == 1 and position == 0:  # Buy
            stake = min(config.STAKE_RANGE[1], max(config.STAKE_RANGE[0], 
                       config.MAX_POSITION_FRACTION * portfolio))
            
            if portfolio >= stake * (1 + config.FEE_RATE):
                position = stake / price
                portfolio -= stake * (1 + config.FEE_RATE)
                entry_price = price
                
                trades.append({
                    'token_id': token_id,
                    'action': 'BUY',
                    'price': price,
                    'position': position,
                    'stake': stake,
                    'step': step
                })
        
        elif action == 2 and position > 0:  # Sell
            sell_value = position * price
            net_value = sell_value * (1 - config.FEE_RATE - config.SLIPPAGE_RATE)
            portfolio += net_value
            
            profit = net_value - (position * entry_price)
            profits.append(profit)
            
            trades.append({
                'token_id': token_id,
                'action': 'SELL',
                'price': price,
                'profit': profit,
                'return_pct': (price - entry_price) / entry_price,
                'step': step
            })
            
            position = 0.0
            entry_price = 0.0
        
        # Check trailing stop
        if position > 0 and price < entry_price * (1 - config.TRAILING_STOP):
            sell_value = position * price
            net_value = sell_value * (1 - config.FEE_RATE - config.SLIPPAGE_RATE)
            portfolio += net_value
            
            profit = net_value - (position * entry_price)
            profits.append(profit)
            
            trades.append({
                'token_id': token_id,
                'action': 'STOP_LOSS',
                'price': price,
                'profit': profit,
                'return_pct': (price - entry_price) / entry_price,
                'step': step
            })
            
            position = 0.0
            entry_price = 0.0
    
    # Close any remaining position
    if position > 0:
        final_price = token_data[-1, columns.index("avg_price")]
        sell_value = position * final_price
        net_value = sell_value * (1 - config.FEE_RATE - config.SLIPPAGE_RATE)
        portfolio += net_value
        
        profit = net_value - (position * entry_price)
        profits.append(profit)
        
        trades.append({
            'token_id': token_id,
            'action': 'FINAL_SELL',
            'price': final_price,
            'profit': profit,
            'return_pct': (final_price - entry_price) / entry_price,
            'step': len(token_data) - 1
        })
    
    if not trades:
        return None
    
    return {
        'token_id': token_id,
        'final_portfolio': portfolio,
        'total_profit': sum(profits),
        'num_trades': len(trades),
        'trades': trades,
        'profits': profits
    }

def get_transformer_prediction(token_data: np.ndarray, columns: list, features: list, 
                             transformer, step: int, window_size: int) -> float:
    """Get transformer prediction for current step"""
    if transformer is None:
        return 0.5
    
    try:
        # Get sequence window
        start_idx = max(0, step - window_size + 1)
        end_idx = step + 1
        
        if end_idx - start_idx < window_size:
            # Pad sequence if too short
            padding_size = window_size - (end_idx - start_idx)
            padded_data = np.zeros((window_size, len(features)))
            actual_data = token_data[start_idx:end_idx]
            
            for i, feat in enumerate(features):
                if feat in columns:
                    feat_idx = columns.index(feat)
                    padded_data[padding_size:, i] = actual_data[:, feat_idx]
            
            sequence = padded_data
        else:
            # Extract feature sequence
            sequence = np.zeros((window_size, len(features)))
            for i, feat in enumerate(features):
                if feat in columns:
                    feat_idx = columns.index(feat)
                    sequence[:, i] = token_data[start_idx:end_idx, feat_idx]
        
        # Make prediction
        with torch.no_grad():
            seq_tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
            prediction = transformer(seq_tensor).item()
            return prediction
    except:
        return 0.5

def calculate_trading_metrics(profits: list, trades: list) -> Dict:
    """Calculate trading performance metrics"""
    if not profits:
        return {
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'total_profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0
        }
    
    profits_array = np.array(profits)
    
    # Basic metrics
    win_rate = np.sum(profits_array > 0) / len(profits_array)
    avg_profit = np.mean(profits_array)
    total_profit = np.sum(profits_array)
    
    # Sharpe ratio (annualized, assuming daily trading)
    sharpe_ratio = avg_profit / (np.std(profits_array) + 1e-6) * np.sqrt(252)
    
    # Max drawdown
    cumulative_profits = np.cumsum(profits_array)
    running_max = np.maximum.accumulate(cumulative_profits)
    drawdowns = running_max - cumulative_profits
    max_drawdown = np.max(drawdowns)
    
    return {
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit,
        'total_profit': total_profit,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(profits)
    }

def main():
    """Main function for running trading simulation"""
    import argparse
    import json
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Run unified trading simulation')
    parser.add_argument('--interval', default='5m', choices=['1m', '5m'], help='Data interval')
    parser.add_argument('--strategy', default='hybrid', 
                       choices=['hybrid', 'rl_only', 'transformer_only'], 
                       help='Trading strategy')
    parser.add_argument('--num-tokens', type=int, default=50, help='Number of tokens to simulate')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results')
    args = parser.parse_args()
    
    # Load unified data
    data_file = f"processed_features_{args.interval}_unified.parquet"
    data_path = os.path.join(os.path.dirname(__file__), '..', data_file)
    
    if not os.path.exists(data_path):
        logger.error(f"Unified data file not found: {data_path}")
        logger.info("Run 'python scripts/run_pipeline3.py' first to generate unified data")
        return
    
    logger.info(f"Loading unified data from: {data_path}")
    df = pl.read_parquet(data_path)
    
    # Load unified models
    unified_models = load_unified_models(args.interval)
    
    if not unified_models:
        logger.error("No unified models found. Train models first.")
        return
    
    # Load RL agent
    rl_agent_path = os.path.join(os.path.dirname(__file__), '..', 'models', 
                                f'rl_agent_{args.interval}_unified.zip')
    
    if not os.path.exists(rl_agent_path):
        logger.error(f"RL agent not found: {rl_agent_path}")
        logger.info("Train RL agent first using 'python ml/rl_agent.py'")
        
        # For now, use baseline evaluation without RL
        if args.strategy == 'rl_only':
            logger.error("Cannot run RL-only strategy without trained RL agent")
            return
        else:
            logger.info("Running simulation without RL agent (transformer_only)")
            args.strategy = 'transformer_only'
            rl_agent = None
    else:
        logger.info(f"Loading RL agent from: {rl_agent_path}")
        rl_agent = PPO.load(rl_agent_path)
    
    # Run simulation
    results = simulate_unified_trading(df, unified_models, rl_agent, 
                                     args.num_tokens, args.strategy)
    
    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(os.path.dirname(__file__), 
                                   f"trading_results_{args.strategy}_{args.interval}_{timestamp}.json")
        
        # Convert non-serializable objects for saving
        save_results = results.copy()
        save_results.pop('token_results', None)  # Too large, skip detailed results
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"UNIFIED TRADING SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Strategy: {results['strategy']}")
    print(f"Interval: {args.interval}")
    print(f"Tokens processed: {results['num_tokens']}")
    print(f"Tokens with trades: {results['tokens_traded']}")
    print(f"Total trades: {results['total_trades']}")
    print(f"")
    print(f"Performance Metrics:")
    print(f"  Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"  Avg Profit/Trade: ${results['metrics']['avg_profit_per_trade']:.2f}")
    print(f"  Total Profit: ${results['metrics']['total_profit']:.2f}")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: ${results['metrics']['max_drawdown']:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()