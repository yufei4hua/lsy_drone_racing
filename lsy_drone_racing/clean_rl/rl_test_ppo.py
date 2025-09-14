# test_ppo_drone.py
import re
import time
from pathlib import Path
import fire

import gymnasium as gym
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import numpy as np
import torch
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.clean_rl.rl_env_wrapper import RLDroneRacingWrapper
from lsy_drone_racing.utils import load_config
from rl_train_ppo import load_latest_model, layer_init, make_env, Agent, Args

class Args:
    random_init: bool = False

    # region Reward Coef
    k_alive:        float = 0.5   # alive reward for every step
    k_alive_anneal: float = 1.0   # anneal alive reward at every step
    k_pos:          float = 0.2   # position based reward coefficient
    k_ellip_norm:   float = 0.9   # ellipse norm axis length
    k_ellip_tang:   float = 0.3   # ellipse tang axis length
    k_gates:        float = 1.0   # gate passing reward coefficient
    k_center_d:     float = 0.5   # center velocity reward coefficient
    k_detour:       float = 0.6   # detour penalty coefficient
    k_detour_scale: float = 15.0  # detour penalty scaling factor: smaller -> wider range
    k_obst:         float = 0.0   # obstacle proximity penalty coefficient
    k_obst_d:       float = 0.0   # obstacle proximity derivative penalty coefficient
    k_act:          float = 0.1   # action regularization coefficient
    k_act_d:        float = 0.01  # action derivative regularization coefficient
    k_vel:          float = -0.0  # velocity regularization coefficient
    k_yaw:          float = 1.1   # yaw angle penalty coefficient
    k_crash:        float = 25.0  # crash penalty coefficient
    k_success:      float = 40.0  # gate passing reward coefficient
    k_finish:       float = 40.0  # finish line reward coefficient
    k_imit:         float = 0.0   # imitation learning reward coefficient

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
        disturbances   = cfg.env.get("disturbances"),
        randomizations = cfg.env.get("randomizations"),
        seed           = cfg.env.seed,
        device         = device,
    )
    env   = JaxToNumpy(env)
    env = RLDroneRacingWrapper(
        env,
        args = Args,
    ) # my custom wrapper
    env   = RecordEpisodeStatistics(env)
    return env

def main(n_runs: int = 10, gui: bool = True):
    log_dir = Path(__file__).parent / "log"
    model_path = load_latest_model(log_dir)
    # model_path = Path(__file__).parent / "log" / "checkpoint_iter_30.pth"

    ep_times = [None] * n_runs
    passed_gates = [0] * n_runs
    vel_max = [0] * n_runs
    vel_avg = [0] * n_runs

    
    if gui: # render
        env = make_eval_env(num_envs=1)
        # load agent
        agent    = Agent(env).to("cpu")
        agent.load_state_dict(torch.load(model_path))

        for ep in range(n_runs):
            obs, _ = env.reset()
            obs    = torch.Tensor(obs)
            done   = False
            velocity = [] # save velocity
            while not done:
                with torch.no_grad():
                    act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(act.cpu().numpy())
                velocity.append(np.linalg.norm(env.env.obs_env["vel"][0]))
                obs  = torch.tensor(obs, dtype=torch.float32)
                done = np.logical_or(terminated, truncated)[0]
                env.render()
                
                if "episode" in info:
                    ep_return = np.sum(info['episode']['r'][info['_episode']])
                    ep_length = np.sum(info['episode']['l'][info['_episode']])
                    lap_time = ep_length / env.env.env.env.freq
                    ep_times[ep] = lap_time if env.env.obs_env["target_gate"][0] == -1 else None
                    passed_gates[ep] = env.env.obs_env["target_gate"][0] if env.env.obs_env["target_gate"][0] >= 0 else 4
                    vel_max[ep] = np.max(np.array(velocity))
                    vel_avg[ep] = np.mean(np.array(velocity))
                    ep_pass = [x for x in ep_times if x is not None]
                    print(f"{len(ep_pass)}/{ep+1} | return = {ep_return:.2f} | length = {ep_length} | lap time = {lap_time:.2f}s")
    else: # run sim parellel
        env = make_eval_env(num_envs=n_runs)
        # load agent
        agent    = Agent(env).to("cpu")
        agent.load_state_dict(torch.load(model_path))
        
        obs, _ = env.reset()
        obs    = torch.Tensor(obs)
        done   = np.array([False]*n_runs)
        velocity = [[] for _ in range(n_runs)] # save velocity
        while np.sum(done) < n_runs:
            with torch.no_grad():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(act.cpu().numpy())
            for i in range(n_runs):
                if not done[i]:
                    velocity[i].append(np.linalg.norm(env.env.obs_env["vel"][i]))
            obs  = torch.tensor(obs, dtype=torch.float32)
            done_mask = np.logical_or(terminated, truncated)
            done = np.logical_or(done, done_mask)
            
            for i in np.arange(n_runs)[done_mask]:
                if done[i] and "episode" in info and info['episode']['r'][i] is not None:
                    print("=",end="")
                    ep_return = np.sum(info['episode']['r'][i])
                    ep_length = np.sum(info['episode']['l'][i])
                    lap_time = ep_length / env.env.env.env.freq
                    ep_times[i] = lap_time if env.env.obs_env["target_gate"][i] == -1 else None
                    passed_gates[i] = env.env.obs_env["target_gate"][i] if env.env.obs_env["target_gate"][i] >= 0 else 4
                    vel_max[i] = np.max(np.array(velocity[i]))
                    vel_avg[i] = np.mean(np.array(velocity[i]))

    env.close()
    ep_pass = [x for x in ep_times if x is not None]
    # print("=" * 60)
    print("")
    print(f"Success Rate: {int(len(ep_pass) / n_runs * 100)}%")
    print(f"Average Lap Time: {((sum(ep_pass)/len(ep_pass)) if len(ep_pass) > 0 else 0.0):.2f}")
    print("Lap Times:   \t|" + '\t|'.join(f"{t:.2f}" if t is not None else '----' for t in ep_times) + '\t|')
    print("Passed Gates:\t|" + '\t|'.join(f"{int(t)}" for t in passed_gates) + '\t|')
    print("Max Velocity:\t|" + '\t|'.join(f"{float(t):.2f}" for t in vel_max) + '\t|')
    print("Mean Velocity:\t|" + '\t|'.join(f"{float(t):.2f}" for t in vel_avg) + '\t|')


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
