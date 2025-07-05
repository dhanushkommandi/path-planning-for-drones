import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from uav_env import UAVPathPlanningEnv

# ✅ Vectorized environment
env = make_vec_env(lambda: UAVPathPlanningEnv(), n_envs=1)

# ✅ Optimized PPO Model
model = PPO(
    "MlpPolicy", env, verbose=1,
    learning_rate=0.0001, gamma=0.99, batch_size=256,
    n_steps=16384, ent_coef=0.01, clip_range=0.2,
    tensorboard_log="./ppo_uav_logs/"
)

# ✅ Train the model
model.learn(total_timesteps=200000)
model.save("ppo_uav_path_planning")
print("✅ Training complete.")
