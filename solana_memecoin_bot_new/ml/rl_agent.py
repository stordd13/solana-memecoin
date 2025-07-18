import gym
from gym import spaces
import numpy as np
import torch
from stable_baselines3 import PPO  # Assume torch policy
from stable_baselines3.common.env_util import make_vec_env
import polars as pl
from transformer_forecast import PumpTransformer  # Hybrid integration

class MemecoinEnv(gym.Env):
    """
    Custom Gym env for Solana memecoin trading sim: Low-liquidity, first 24h.
    State: Features + archetype + Transformer forecast + current position.
    Actions: 0=hold, 1=buy ($5-50), 2=sell.
    Reward: Net profit - slippage - fees - drawdown penalty; Kelly sizing.
    """
    def __init__(self, df: pl.DataFrame, archetype: int, transformer_model: PumpTransformer):
        super().__init__()
        self.df = df.filter(pl.col("archetype") == archetype).sort("timestamp")
        self.features = ["scaled_returns", "volatility", "imbalance", "avg_volume_5m", "liquidity"]
        self.state_dim = len(self.features) + 2  # + archetype + forecast
        self.action_space = spaces.Discrete(3)  # hold/buy/sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        self.current_step = 0
        self.position = 0.0  # Current holding
        self.portfolio = 1000.0  # Starting capital
        self.transformer = transformer_model
        
    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.portfolio = 1000.0
        return self._get_state()
    
    def step(self, action):
        row = self.df.row(self.current_step)
        price = row[self.df.columns.index("avg_price")]
        returns = row[self.df.columns.index("returns")]
        
        # Transformer forecast as state part
        seq = self.df.slice(self.current_step - 9, 10).select(self.features).to_numpy()  # Last 10min
        forecast = self.transformer(torch.tensor(seq[None, :, :], dtype=torch.float32)).item()
        
        reward = 0
        if action == 1:  # Buy
            stake = min(50, max(5, 0.005 * self.portfolio))  # Kelly-like, 0.5% max
            self.position += stake / price
            self.portfolio -= stake * (1 + 0.005)  # Fee
            reward -= stake * 0.03  # Slippage penalty
        
        elif action == 2:  # Sell
            if self.position > 0:
                sell_value = self.position * price * (1 - 0.03)  # Slippage
                self.portfolio += sell_value * (1 - 0.005)  # Fee
                reward += sell_value - (self.position * price)  # Profit
                self.position = 0
        
        # Step forward, apply market move
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward += returns * self.position * price  # Holding profit
        reward -= max(0, (1000 - self.portfolio) / 1000 * 10)  # Drawdown penalty (<5%)
        
        return self._get_state(forecast), reward, done, {}
    
    def _get_state(self, forecast=0.0):
        row = self.df.row(self.current_step)
        feat_values = np.array([row[self.df.columns.index(f)] for f in self.features])
        state = np.append(feat_values, [row[self.df.columns.index("archetype")], forecast])
        return state

def train_rl_per_archetype(df: pl.DataFrame, transformer_models: dict, episodes: int = 25000):
    """
    Trains PPO RL agent per archetype; hybrid with Transformer forecasts.
    Custom env sims high-frequency trades; targets >50% win rate after fees/slippage.
    """
    agents = {}
    for arch in df["archetype"].unique():
        env = make_vec_env(lambda: MemecoinEnv(df, arch, transformer_models[arch]), n_envs=1)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # Torch policy
        model.learn(total_timesteps=episodes)
        agents[arch] = model
        
        # Eval sim
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Archetype {arch} Total Reward: {total_reward} (proxy for win rate)")
    
    return agents

# Usage after transformers
rl_agents = train_rl_per_archetype(df, transformer_models)