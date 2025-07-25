"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

import time

logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
    log_dir : str = r'lsy_drone_racing/logs/roll_outs',
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    passed_gates = []
    vel_max = []
    vel_avg = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config, env)
        i = 0
        fps = 60
        
        velocity = []
        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            velocity.append(np.linalg.norm(obs["vel"]))
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            # Synchronize the GUI.
            if config.sim.gui:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
                    # time.sleep(1/config.env.freq)
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)
        passed_gates.append(obs["target_gate"] if obs["target_gate"] >= 0 else 4)
        vel_max.append(np.max(np.array(velocity)))
        vel_avg.append(np.mean(np.array(velocity)))
        ep_pass = [x for x in ep_times if x is not None]
        print(f"{len(ep_pass)}/{len(ep_times)}, Max Velocity: {vel_max[-1]:.2f}, Mean Velocity: {vel_avg[-1]:.2f}\n\n")

    # Close the environment
    env.close()
    ep_pass = [x for x in ep_times if x is not None]
    print(f"Success Rate: {int(len(ep_pass)/n_runs*100)}%")
    print(f"Average Lap Time: {((sum(ep_pass)/len(ep_pass)) if len(ep_pass) > 0 else 0.0):.2f}")
    print("Lap Times:   \t|" + '\t|'.join(f"{t:.2f}" if t is not None else '----' for t in ep_times) + '\t|')
    print("Passed Gates:\t|"  + '\t|'.join(f"{int(t)}" for t in passed_gates) + '\t|')
    print("Max Velocity:\t|"  + '\t|'.join(f"{float(t):.2f}" for t in vel_max) + '\t|')
    print("Mean Velocity:\t|" + '\t|'.join(f"{float(t):.2f}" for t in vel_avg) + '\t|')
    # Log all roll-out statistics to a csv file
    log_all_roll_outs(log_path=Path(log_dir), ep_times=ep_times, passed_gates=passed_gates, vel_max=vel_max, vel_avg=vel_avg)
    # ep_times = ep_pass

    return ep_times


def log_all_roll_outs(log_path : str, ep_times : list[float], passed_gates: list[int], vel_max: list[float], vel_avg: list[float]):
    """Log all roll-out statistics to a csv file."""
    os.makedirs(log_path,exist_ok=True)
    log_path = os.path.join(log_path, "roll_outs.csv")
    with open(log_path, "w") as f:
        f.write("Lap Times,Passed Gates,Max Velocity,Mean Velocity\n")
        for i in range(len(ep_times)):
            f.write(f"{(ep_times[i] if ep_times[i] is not None else -1.0):.2f},{passed_gates[i] if passed_gates[i] is not None else 0},{vel_max[i]:.2f},{vel_avg[i]:.2f}\n")

def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
