from __future__ import annotations
import os, re, time
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.reinforcement_learning.rl_env_wrapper import RLDroneRacingWrapper
from lsy_drone_racing.utils import load_config


def build_env_fn(n_worlds: int) -> Callable[[], gym.Env]:
    """
    Factory that creates **one** wrapped vector environment.
    Each subprocess still simulates `jax_env_worlds` worlds in parallel on GPU.
    """
    def _init() -> gym.Env:
        config = load_config(Path(__file__).parents[2] / "config/trainrl.toml")

        base_env = VecDroneRaceEnv(
            num_envs       = n_worlds,   # e.g. 4 worlds / subprocess
            freq           = config.env.freq,
            sim_config     = config.sim,
            track          = config.env.track,
            sensor_range   = config.env.sensor_range,
            control_mode   = config.env.control_mode,
            disturbances   = config.env.get("disturbances"),
            randomizations = config.env.get("randomizations"),
            seed           = config.env.seed,
        )
        return RLDroneRacingWrapper(base_env)

    return _init


def latest_model(log_dir: Path, lesson: int, idx: int | None = None) -> Tuple[Path, int]:
    """Return last (or specified idx) checkpoint path for a lesson."""
    patt = re.compile(rf"ppo_final_model_{lesson}_(\d+)\.zip")
    files = [(f, int(m.group(1))) for f in log_dir.glob(f"ppo_final_model_{lesson}_*.zip")
             if (m := patt.match(f.name))]
    if not files:
        raise FileNotFoundError(f"No model for lesson {lesson} in {log_dir}")

    if idx is None:
        f, n = max(files, key=lambda t: t[1])
        return f, n + 1

    for f, n in files:
        if n == idx:
            return f, n + 1
    raise FileNotFoundError(f"ppo_final_model_{lesson}_{idx}.zip not found")


def main() -> None:
    n_subproc = 1                                      # CPU-level parallelism
    n_world = 20
    env_fns    = [build_env_fn(n_world) for i in range(n_subproc)]
    env        = SubprocVecEnv(env_fns)

    log_dir    = Path(__file__).parent / "log5"
    log_dir.mkdir(exist_ok=True)
    time_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # callbacks
    ckpt_cb = CheckpointCallback(save_freq=50_000,
                                 save_path=log_dir,
                                 name_prefix=f"model_{time_tag}")
    eval_cb = EvalCallback(env,
                           best_model_save_path=log_dir,
                           eval_freq=10_000,
                           n_eval_episodes=5)

    # PPO (small net, ReLU)
    policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.ReLU)
    model = PPO("MlpPolicy",
                env,
                tensorboard_log=log_dir,
                verbose=1,
                device="cpu",
                n_steps=2048,
                batch_size=64,
                gae_lambda=0.95,
                gamma=0.99,
                learning_rate=3e-4,
                ent_coef=0.0,
                policy_kwargs=policy_kwargs)

    # ── auto-resume ──
    lesson      = 1
    reset_idx   = 0   # set to None to resume latest
    try:
        ckpt_path, next_idx = latest_model(log_dir, lesson, idx=reset_idx)
        model = PPO.load(ckpt_path, env=env, device="cpu")
        print(f"▶ Resuming lesson {lesson}.{reset_idx if reset_idx is not None else next_idx-1}")
    except FileNotFoundError:
        next_idx = 0
        print(f"▶ No previous model, training lesson {lesson}.0 from scratch")

    # ── train ──
    model.learn(total_timesteps=12 * 400_000,
                callback=[ckpt_cb, eval_cb])

    # ── save ──
    final_path = log_dir / f"ppo_final_model_{lesson}_{next_idx}.zip"
    model.save(final_path)
    print(f"✅ Training done. Saved to {final_path}")


if __name__ == "__main__":
    main()