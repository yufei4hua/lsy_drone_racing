# test_ppo_drone.py
import re
import time
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy
import numpy as np
import torch
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.reinforcement_learning.rl_env_wrapper import RLDroneRacingWrapper
from lsy_drone_racing.utils import load_config
from rl_train_ppo import layer_init, make_env, Agent
    
# load model
def load_latest_model(log_dir: Path) -> Path:
    patt = re.compile(r"rl_drone_racing_(\d+)\.pth$")
    candidates = [(int(m.group(1)), f) for f in log_dir.glob("rl_drone_racing_*.pth")
                  if (m := patt.match(f.name))]
    if not candidates:
        raise FileNotFoundError("No saved model found in log_dir")
    latest_path = max(candidates, key=lambda t: t[0])[1]
    print(f"👉 Loading model: {latest_path}")
    return latest_path

# env
def make_eval_env(num_envs=1, device="cpu"):
    cfg = load_config(Path(__file__).parents[2] / "config/trainrl.toml")

    env = VecDroneRaceEnv(
        num_envs       = num_envs,
        freq           = cfg.env.freq,
        sim_config     = cfg.sim,
        track          = cfg.env.track,
        sensor_range   = cfg.env.sensor_range,
        control_mode   = cfg.env.control_mode,
        disturbances   = None,
        randomizations = None,
        seed           = cfg.env.seed,
        device         = device,
    )
    env   = JaxToNumpy(env)
    env   = RLDroneRacingWrapper(env)
    env   = gym.wrappers.vector.RecordEpisodeStatistics(env)
    return env

def main():
    log_dir = Path(__file__).parent / "log4"
    model_path = load_latest_model(log_dir)

    env = make_eval_env(num_envs=1)
    agent    = Agent(env).to("cpu")
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    EPISODES = 20
    for ep in range(EPISODES):
        obs, _ = env.reset()
        obs    = torch.tensor(obs, dtype=torch.float32)
        done   = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            act = agent.act(obs, deterministic=True).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(act)
            obs  = torch.tensor(obs, dtype=torch.float32)
            done = np.logical_or(terminated, truncated).item()
            env.render()
            ep_ret += reward.item()
            ep_len += 1
        print(f"[Episode {ep+1}] return = {ep_ret:.2f} | length = {ep_len}")

    env.close()

if __name__ == "__main__":
    main()
