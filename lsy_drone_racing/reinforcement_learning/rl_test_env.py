import gymnasium
from pathlib import Path
import numpy as np
import keyboard
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv

def make_env(seed):
    def _init():
        config = load_config(Path(__file__).parents[2] / "config/levelrl.toml")
        env = RLDroneRaceEnv(
            freq=config.env.freq,
            sim_config=config.sim,
            track=config.env.track,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
        )
        return env
    return _init


def test_models(model_paths, num_episodes=999, render=True):
    env_fns = [make_env(seed=0)]
    env = DummyVecEnv(env_fns)

    models = []
    for path in model_paths:
        model = PPO.load(path, env=env, device="cpu")
        models.append(model)

    current_model_idx = 0
    print(f"current model: {model_paths[current_model_idx].name}")

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            if keyboard.is_pressed('right'):
                current_model_idx = (current_model_idx + 1) % len(models)
                print(f"switch to model: {model_paths[current_model_idx].name}")
                time.sleep(0.3)

            elif keyboard.is_pressed('left'):
                current_model_idx = (current_model_idx - 1) % len(models)
                print(f"switch to model: {model_paths[current_model_idx].name}")
                time.sleep(0.3)

            model = models[current_model_idx]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            if render:
                env.envs[0].render()

        print(f"Episode {ep + 1}: reward={episode_reward}, steps={step}")

    env.close()

if __name__ == "__main__":
    model_paths = [
        # Path(__file__).parent / "log/ppo_final_model_hover_last",
        # Path(__file__).parent / "log/ppo_final_model_gate_center",
        # Path(__file__).parent / "log/ppo_final_model_pass_first_gate",
        # Path(__file__).parent / "log/ppo_final_model_scare_of_gate",
        # Path(__file__).parent / "log/ppo_final_model_pass_then_climb",
        Path(__file__).parent / "log/ppo_final_model_finish_race",
    ]
    test_models(model_paths)





# import gymnasium
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# from torch import nn
# from datetime import datetime
# import os
# from pathlib import Path

# from docs import conf
# from lsy_drone_racing.utils import load_config
# from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RenderCallback

# config = load_config(Path(__file__).parents[2] / "config" / "levelrl.toml")

# env = RLDroneRaceEnv = gymnasium.make(
#         config.env.id,
#         freq=config.env.freq,
#         sim_config=config.sim,
#         sensor_range=config.env.sensor_range,
#         control_mode=config.env.control_mode,
#         track=config.env.track,
#         disturbances=config.env.get("disturbances"),
#         randomizations=config.env.get("randomizations"),
#         seed=config.env.seed,
#     )

# obs, info = env.reset()
# i = 0
# fps = 60

# while True:
#     curr_time = i / config.env.freq

#     action = None
#     obs, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         break
#     # Synchronize the GUI.
#     if config.sim.gui:
#         if ((i * fps) % config.env.freq) < fps:
#             env.render()
#     i += 1

# # Close the environment
# env.close()