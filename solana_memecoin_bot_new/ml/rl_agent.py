# ml/rl_agent.py (Per-token episodes with walk-forward; PyTorch PPO for RL)
import gym
from gym import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import polars as pl
from transformer_forecast import PumpTransformer
from utils import setup_logger

logger = setup_logger(__name__)

class MemecoinEnv(gym.Env):
    def __init__(self, df: pl.DataFrame, archetype: int, transformer_model: PumpTransformer, token_id: str = None):
        super().__init__()
        if token_id:
            self.df = df.filter(pl.col("token_id") == token_id & pl.col("archetype") == archetype).sort("datetime")
        else:
            self.df = df.filter(pl.col("archetype") == archetype).sort("datetime")
        self.features = ["scaled_returns", "volatility", "imbalance", "avg_volume_5m", "liquidity"]
        self.state_dim = len(self.features) + 2  # + archetype + forecast
        self.action_space = spaces.Discrete(3)  # hold/buy/sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        self.current_step = 0
        self.position = 0.0
        self.portfolio = 1000.0
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
        
        seq = self.df.slice(max(0, self.current_step - 9), 10).select(self.features).to_numpy()
        forecast = self.transformer(torch.tensor(seq[None, :, :], dtype=torch.float32)).item()
        
        reward = 0
        if action == 1:  # Buy
            stake = min(50, max(5, 0.005 * self.portfolio))
            self.position += stake / price
            self.portfolio -= stake * (1 + 0.005)
            reward -= stake * 0.03
        
        elif action == 2:  # Sell
            if self.position > 0:
                sell_value = self.position * price * (1 - 0.03)
                self.portfolio += sell_value * (1 - 0.005)
                reward += sell_value - (self.position * price)
                self.position = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward += returns * self.position * price
        reward -= max(0, (1000 - self.portfolio) / 1000 * 10)
        
        return self._get_state(forecast), reward, done, {}
    
    def _get_state(self, forecast=0.0):
        row = self.df.row(self.current_step)
        feat_values = np.array([row[self.df.columns.index(f)] for f in self.features])
        state = np.append(feat_values, [row[self.df.columns.index("archetype")], forecast])
        return state

def train_rl_per_archetype(df: pl.DataFrame, transformer_models: dict, episodes: int = 25000, walk_steps: int = 5):
    agents = {}
    for arch in df["archetype"].unique():
        subset = df.filter(pl.col("archetype") == arch)
        arch_rewards = []
        for step in range(walk_steps):
            step_start = time.time()
            # Walk-forward: Per-token rolling
            total_reward = 0
            for token in subset["token_id"].unique():
                token_df = subset.filter(pl.col("token_id") == token)
                shift = int(len(token_df) * 0.2 * step)
                train_len = int(len(token_df) * 0.7)
                train_df = token_df.slice(shift, train_len)
                if len(train_df) < 10: continue
                
                env = make_vec_env(lambda: MemecoinEnv(train_df, arch, transformer_models[arch]), n_envs=1)
                model = PPO("MlpPolicy", env, verbose=0, device="cpu")
                model.learn(total_timesteps=episodes // walk_steps)  # Split episodes
                
                # Eval on forward slice
                test_df = token_df.slice(shift + train_len, len(token_df) - (shift + train_len))
                obs = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
            arch_rewards.append(total_reward)
            logger.info(f"Archetype {arch}, Walk Step {step}: Reward {total_reward}, Time {time.time() - step_start:.2f}s")
        
        agents[arch] = model  # Last or best
        avg_reward = np.mean(arch_rewards)
        logger.info(f"Archetype {arch} Avg Walk-Forward Reward: {avg_reward}")
    return agents

# Usage after transformers
rl_agents = train_rl_per_archetype(df, transformer_models)