import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from datetime import datetime
import os
from pathlib import Path
import time
from docs import conf
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.reinforcement_learning.rl_drone_race import RLDroneRaceEnv, RenderCallback

# === 1. 创建训练环境 ===
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
# env = JaxToNumpy(env)
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

def main():
    num_envs = 1
    start_time = time.time()
    env_fns = [make_env(seed=i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # === 2. 设置训练保存目录和回调 ===
    log_dir = Path(__file__).parent / "log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=f"model_{timestamp}")
    eval_callback = EvalCallback(vec_env, best_model_save_path=log_dir, eval_freq=10000, n_eval_episodes=5)

    # === 3. 初始化 PPO 模型 ===
    # policy_kwargs = dict(
    #     net_arch=[128, 128],         # 两层，每层 128
    #     activation_fn=nn.ReLU        # 激活函数（默认是 Tanh，可以改为 ReLU）
    # )
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     policy_kwargs=policy_kwargs,
    #     n_steps=2048,
    #     batch_size=64,
    #     gae_lambda=0.95,
    #     gamma=0.99,
    #     learning_rate=3e-4,
    #     ent_coef=0.0,
    #     device="cpu",
    # )
    # 加载模型
    model = PPO.load(Path(__file__).parent / "log/ppo_final_model", env=vec_env, device="cpu")

    # === 4. 启动训练 ===
    if num_envs > 1:
        model.learn(total_timesteps=600000, callback=[checkpoint_callback, eval_callback])
    else: # for visualization
        render_callback = RenderCallback(render_freq=1)
        model.learn(total_timesteps=20000, callback=[render_callback])

    # === 5. 保存最终模型 ===
    model.save(f"{log_dir}/ppo_final_model")
    end_time = time.time()
    print(f"✅ 训练完成，耗时{end_time - start_time}s ，模型已保存至 {log_dir}")


if __name__ == "__main__":
    main()
