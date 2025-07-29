# ml/rl_agent.py - Unified RL agent for cross-token pump prediction (no archetypes)
import gym
from gym import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import polars as pl
import os
import sys
import time
from typing import Dict, Optional, Tuple

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils import setup_logger, load_unified_models, get_unified_features, validate_feature_compatibility
import config

logger = setup_logger(__name__)

class UnifiedMemecoinEnv(gym.Env):
    """
    Unified trading environment for memecoin RL agent (no archetype separation).
    Uses unified transformer model for pump predictions and realistic trading constraints.
    """
    
    def __init__(self, df: pl.DataFrame, unified_models: Dict, token_id: Optional[str] = None, 
                 window_size: int = 20):
        super().__init__()
        
        # Filter data if specific token requested
        if token_id:
            self.df = df.filter(pl.col("token_id") == token_id).sort("datetime")
        else:
            # Use all tokens for training
            self.df = df.sort(["token_id", "datetime"])
        
        # Get feature list from unified models
        self.features = get_unified_features()
        
        # Validate feature compatibility
        compatible, missing = validate_feature_compatibility(self.df.columns, self.features)
        if not compatible:
            logger.warning(f"Missing features for RL env: {missing}")
            # Use available features only
            self.features = [f for f in self.features if f in self.df.columns]
        
        # State dimension: features + transformer forecasts (5 horizons) + metadata
        self.state_dim = len(self.features) + 7  # + 5 forecasts + position + portfolio_ratio
        self.action_space = spaces.Discrete(3)  # hold/buy/sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        
        # Trading state
        self.current_step = 0
        self.position = 0.0
        self.portfolio = 1000.0
        self.entry_price = 0.0
        self.window_size = window_size
        
        # Models
        self.unified_models = unified_models
        self.transformer = unified_models.get('transformer', {}).get('model')
        self.baseline = unified_models.get('baseline')
        
        # Convert DataFrame to numpy for faster access
        self.df_data = self.df.to_numpy()
        self.df_columns = self.df.columns
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0.0
        self.portfolio = 1000.0
        self.entry_price = 0.0
        return self._get_state()
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.df_data) - 1:
            return self._get_state(), 0, True, {}
        
        # Get current row data
        row = self.df_data[self.current_step]
        price = row[self.df_columns.index("avg_price")]
        returns = row[self.df_columns.index("returns")] if "returns" in self.df_columns else 0.0
        
        # Get transformer multi-horizon forecasts
        forecasts = self._get_transformer_prediction()
        
        # Use average forecast for decision making (can be enhanced later)
        avg_forecast = np.mean(forecasts)
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.position == 0:  # Only buy if not already holding
                stake = min(config.STAKE_RANGE[1], max(config.STAKE_RANGE[0], 
                           config.MAX_POSITION_FRACTION * self.portfolio))
                
                if self.portfolio >= stake * (1 + config.FEE_RATE):
                    self.position = stake / price
                    self.portfolio -= stake * (1 + config.FEE_RATE)
                    self.entry_price = price
                    reward -= stake * 0.01  # Small penalty for trading
        
        elif action == 2:  # Sell
            if self.position > 0:
                sell_value = self.position * price
                net_value = sell_value * (1 - config.FEE_RATE - config.SLIPPAGE_RATE)
                self.portfolio += net_value
                
                # Calculate profit/loss
                profit = (price - self.entry_price) * self.position
                reward += profit * 0.1  # Reward for profit
                
                self.position = 0.0
                self.entry_price = 0.0
        
        # Hold (action == 0) does nothing
        
        # Update position value and calculate unrealized P&L
        if self.position > 0:
            unrealized_pnl = (price - self.entry_price) * self.position
            reward += unrealized_pnl * 0.01  # Small reward for unrealized profit
            
            # Trailing stop loss
            if price < self.entry_price * (1 - config.TRAILING_STOP):
                # Force sell
                sell_value = self.position * price
                net_value = sell_value * (1 - config.FEE_RATE - config.SLIPPAGE_RATE)
                self.portfolio += net_value
                reward -= abs(unrealized_pnl) * 0.2  # Penalty for stop loss
                self.position = 0.0
                self.entry_price = 0.0
        
        # Risk penalty for portfolio drawdown
        portfolio_value = self.portfolio + (self.position * price if self.position > 0 else 0)
        if portfolio_value < 1000 * (1 - config.MAX_DRAWDOWN):
            reward -= (1000 - portfolio_value) * 0.01
        
        self.current_step += 1
        done = self.current_step >= len(self.df_data) - 1
        
        return self._get_state(forecasts), reward, done, {}
    
    def _get_transformer_prediction(self) -> np.ndarray:
        """Get transformer model multi-horizon price forecasts for current sequence"""
        if self.transformer is None:
            return np.zeros(5)  # Return 5 zeros for 5 horizons
        
        try:
            # Get sequence window
            start_idx = max(0, self.current_step - self.window_size + 1)
            end_idx = self.current_step + 1
            
            if end_idx - start_idx < self.window_size:
                # Pad sequence if too short
                padding_size = self.window_size - (end_idx - start_idx)
                padded_data = np.zeros((self.window_size, len(self.features)))
                actual_data = self.df_data[start_idx:end_idx]
                
                for i, feat in enumerate(self.features):
                    if feat in self.df_columns:
                        feat_idx = self.df_columns.index(feat)
                        padded_data[padding_size:, i] = actual_data[:, feat_idx]
                
                sequence = padded_data
            else:
                # Extract feature sequence
                sequence = np.zeros((self.window_size, len(self.features)))
                for i, feat in enumerate(self.features):
                    if feat in self.df_columns:
                        feat_idx = self.df_columns.index(feat)
                        sequence[:, i] = self.df_data[start_idx:end_idx, feat_idx]
            
            # Make prediction - returns 5 price change forecasts
            with torch.no_grad():
                seq_tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32)
                forecasts = self.transformer(seq_tensor).cpu().numpy()[0]  # Shape: (5,)
                return forecasts
                
        except Exception as e:
            logger.debug(f"Transformer prediction failed: {e}")
            return np.zeros(5)
    
    def _get_state(self, forecasts: np.ndarray = None) -> np.ndarray:
        """Get current environment state"""
        if forecasts is None:
            forecasts = np.zeros(5)
            
        if self.current_step >= len(self.df_data):
            # Return zero state if we're past the end
            return np.zeros(self.state_dim)
        
        row = self.df_data[self.current_step]
        
        # Extract feature values
        feat_values = []
        for feat in self.features:
            if feat in self.df_columns:
                feat_idx = self.df_columns.index(feat)
                feat_values.append(row[feat_idx])
            else:
                feat_values.append(0.0)
        
        # Add metadata: 5 forecasts, position, portfolio ratio
        portfolio_value = self.portfolio + (self.position * row[self.df_columns.index("avg_price")] 
                                          if self.position > 0 and "avg_price" in self.df_columns else 0)
        portfolio_ratio = portfolio_value / 1000.0  # Normalized to starting value
        
        state = np.array(feat_values + forecasts.tolist() + [self.position, portfolio_ratio])
        return state.astype(np.float32)

def train_unified_rl_agent(df: pl.DataFrame, unified_models: Dict, episodes: int = 50000, 
                          walk_steps: int = 5, interval: str = "5m") -> PPO:
    """
    Train unified RL agent using walk-forward validation across all tokens.
    
    Args:
        df: Unified dataset (all tokens)
        unified_models: Dictionary containing transformer and baseline models
        episodes: Total training episodes
        walk_steps: Number of walk-forward steps
        interval: Data interval for model saving
        
    Returns:
        Trained PPO agent
    """
    logger.info(f"Training unified RL agent with {episodes} episodes, {walk_steps} walk steps")
    
    # Filter to training data only
    train_df = df.filter(pl.col("split") == "train")
    logger.info(f"Training on {train_df.height} samples from {train_df['token_id'].n_unique()} tokens")
    
    best_agent = None
    best_reward = float('-inf')
    all_rewards = []
    
    for step in range(walk_steps):
        step_start = time.time()
        logger.info(f"Walk-forward step {step + 1}/{walk_steps}")
        
        # Walk-forward: temporal shift for each token
        step_rewards = []
        tokens = train_df["token_id"].unique().to_list()
        
        # Train on subset of tokens for this step
        tokens_per_step = len(tokens) // walk_steps
        start_token_idx = step * tokens_per_step
        end_token_idx = (step + 1) * tokens_per_step if step < walk_steps - 1 else len(tokens)
        step_tokens = tokens[start_token_idx:end_token_idx]
        
        logger.info(f"Step {step + 1}: Training on {len(step_tokens)} tokens")
        
        # Create environment for this step
        step_df = train_df.filter(pl.col("token_id").is_in(step_tokens))
        
        def make_env():
            return UnifiedMemecoinEnv(step_df, unified_models, 
                                    window_size=unified_models.get('transformer', {}).get('window_size', 20))
        
        # Optimize for M4 Max: Parallel environments + larger batches
        n_envs = 8 if torch.backends.mps.is_available() else 4  # Parallel environments
        batch_size_rl = 512 if torch.backends.mps.is_available() else 256  # Large batches for 64GB RAM
        device_rl = "mps" if torch.backends.mps.is_available() else "cpu"
        
        logger.info(f"🚀 RL Optimization: {n_envs} parallel envs, batch_size={batch_size_rl}, device={device_rl}")
        
        # Train agent with parallel environments
        env = make_vec_env(make_env, n_envs=n_envs)
        model = PPO("MlpPolicy", env, verbose=1, device=device_rl,
                   learning_rate=3e-4, batch_size=batch_size_rl, n_steps=2048)
        
        model.learn(total_timesteps=episodes // walk_steps)
        
        # Evaluate on test tokens
        eval_reward = evaluate_agent(model, df, unified_models, num_tokens=10)
        step_rewards.append(eval_reward)
        all_rewards.append(eval_reward)
        
        logger.info(f"Step {step + 1} completed - Reward: {eval_reward:.2f}, "
                   f"Time: {time.time() - step_start:.2f}s")
        
        # Keep best agent
        if eval_reward > best_reward:
            best_reward = eval_reward
            best_agent = model
            logger.info(f"New best agent with reward: {best_reward:.2f}")
        
        env.close()
    
    # Save best agent
    if best_agent:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 
                                 f'rl_agent_{interval}_unified.zip')
        best_agent.save(model_path)
        logger.info(f"Saved best RL agent to: {model_path}")
    
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    logger.info(f"Training completed - Avg reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    return best_agent

def evaluate_agent(agent: PPO, df: pl.DataFrame, unified_models: Dict, num_tokens: int = 20) -> float:
    """
    Evaluate RL agent on test data.
    
    Args:
        agent: Trained PPO agent
        df: Full dataset
        unified_models: Models dictionary
        num_tokens: Number of test tokens to evaluate on
        
    Returns:
        Average reward per episode
    """
    test_df = df.filter(pl.col("split") == "test")
    test_tokens = test_df["token_id"].unique().to_list()[:num_tokens]
    
    total_rewards = []
    
    for token_id in test_tokens:
        token_df = test_df.filter(pl.col("token_id") == token_id)
        if len(token_df) < 20:  # Skip very short sequences
            continue
        
        env = UnifiedMemecoinEnv(token_df, unified_models, token_id=token_id)
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards) if total_rewards else 0.0

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train unified RL agent')
    parser.add_argument('--interval', default='5m', choices=['1m', '5m'], help='Data interval')
    parser.add_argument('--episodes', type=int, default=50000, help='Training episodes')
    parser.add_argument('--walk-steps', type=int, default=5, help='Walk-forward steps')
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
        logger.error("No unified models found. Train transformer and baseline models first.")
        return
    
    # Train agent
    agent = train_unified_rl_agent(df, unified_models, args.episodes, args.walk_steps, args.interval)
    
    if agent:
        logger.info("RL agent training completed successfully!")
    else:
        logger.error("RL agent training failed")

if __name__ == "__main__":
    main()