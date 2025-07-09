import gymnasium
from crazyflow import Sim
import lsy_drone_racing

from lsy_drone_racing.utils.utils import load_config
from pathlib import Path
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy

config = load_config(Path(__file__).parents[0] / "config/level2.toml")
# Load the controller module
# Create the racing environment
env = gymnasium.make_vec(
    config.env.id,
    num_envs=10_000,
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
    device="gpu"
)
env = JaxToNumpy(env)

obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(reward.shape)