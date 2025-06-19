from posixpath import relpath
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from lsy_drone_racing.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv
from jax import Array
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils import draw_line

from lsy_drone_racing.tools import race_objects
from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space, build_observation_space

from jax import Array
from ml_collections import ConfigDict
from typing import Dict, Tuple

IMMITATION_LEARNING = True
if IMMITATION_LEARNING:
    from pathlib import Path
    from lsy_drone_racing.utils import load_config
    from lsy_drone_racing.control.mpcc import MPCC

class RLDroneRacingWrapper(gymnasium.Wrapper):
    def __init__(self, env: VecDroneRaceEnv):
        super().__init__(env)
        # create action & observation spaces
        self.num_envs = env.num_envs
        self.single_action_space = build_action_space(env.single_action_space.dtype)
        self.action_space = env.action_space
        state_dim = 36
        lim = np.full(state_dim, np.inf, dtype=np.float32) # set to infinite for now
        self.single_observation_space = spaces.Box(-lim, lim, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        # initialize internal state saving variables
        self._d_safe = 1.0
        self._act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_act = np.repeat(self._act_bias[None, :], self.num_envs, axis=0)
        self._prev_gate = np.zeros(self.num_envs, dtype=int)
        # region RL parameters
        self.k_living = 0.4
        self.k_obst = 0.4
        self.k_obst_d = 0.5
        self.k_gates = 1.0
        self.k_center = 0.3
        self.k_vel = -0.4
        self.k_act = 0.01
        self.k_act_d = 0.001
        self.k_yaw = 0.1
        self.k_crash = 20
        self.k_success = 30

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs_env, info = self.env.reset(**kwargs)           # dict → array(num_envs, ...)
        state = self._obs_to_state(obs_env,
                                   np.repeat(self._act_bias[None, :], self.num_envs, axis=0))
        self._prev_act = np.repeat(self._act_bias[None, :], self.num_envs, axis=0)
        self._prev_gate = np.zeros(self.num_envs, dtype=int)
        return state, info

    def step(self, action: Array) -> tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action for the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        action_exec = action + self.act_bias # always apply RL output on bias
        obs, _, terminated, truncated, info = self.env.step(action_exec)
        # obs = {k: v[0, 0] for k, v in obs.items()}
        obs_rl = self.obs_to_state(obs, action)
        reward = self.reward(obs, action)
        reward += -self.k_crash * int((terminated or truncated) and int(np.sum(obs['gates_visited'])) < self.num_gates)
        # info = {k: v[0, 0] for k, v in info.items()}
        return obs_rl, reward, terminated, truncated, info
    def step(self, action: np.ndarray):
        action_exec = action + self._act_bias
        obs_env, _, terminated, truncated, info = self.env.step(action_exec)
        state = self._obs_to_state(obs_env, action)
        reward = self._reward(obs_env, action)
        done = terminated | truncated

        self._prev_act = action
        self._prev_gate = obs_env["target_gate"].astype(int)
        return state, reward, done, truncated, info

    # region OBS
    def _obs_to_state(self, obs: dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """
        Args:
            obs    : Dict[str, np.ndarray]
                    pos            (N, 3)
                    vel            (N, 3)
                    quat           (N, 4)
                    ang_vel        (N, 3)
                    gates_pos      (N, n_gates, 3)
                    gates_quat     (N, n_gates, 4)
                    obstacles_pos  (N, n_obst, 3)
            action : np.ndarray, shape (N, 4)
        Returns:
            state  : np.ndarray, shape (N, 36)
        """
        pos      = obs["pos"]           # (N, 3)
        vel      = obs["vel"]           # (N, 3)
        quat     = obs["quat"]          # (N, 4)
        ang_vel  = obs["ang_vel"]       # (N, 3)
        N        = pos.shape[0]

        curr_gate_idx = obs['target_gate']
        gate_quat = obs["gates_quat"][np.arange(N), curr_gate_idx]   # (N, 4)
        gate_pos  = obs["gates_pos"][np.arange(N), curr_gate_idx]    # (N, 3)
        gate_rot_mat = R.from_quat(gate_quat).as_matrix()

        half_w, half_h = 0.2, 0.2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners = (gate_rot_mat @ corners_local.T).transpose(0, 2, 1) + gate_pos[:, None, :]
        draw_line(self, np.stack([gate_corners[0, 0], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 1], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 2], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 3], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        rel_pos_gate = gate_corners - pos[:, None, :]   # (N, 4, 3)

        obst_rel_xy = obs["obstacles_pos"][:, :, :2] - pos[:, None, :2]       # (N, n_obst, 2)
        obst_dists  = np.linalg.norm(obst_rel_xy, axis=-1)                    # (N, n_obst)
        closest_idx = obst_dists.argmin(axis=-1)                              # (N,)
        rel_xy_obst = obst_rel_xy[np.arange(N), closest_idx]                  # (N, 2)
        dist        = obst_dists[np.arange(N), closest_idx]                   # (N,)

        rel_xy_obst_gaus = rel_xy_obst * np.exp(-(dist / (0.5 * self._d_safe))**2)[:, None] \
                        / (dist[:, None] + 1e-6)                              # (N, 2)


        rot_mat  = R.from_quat(quat).as_matrix().reshape(N, -1)               # (N, 9)
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)                          # (N, 3)

        state = np.concatenate([
            pos,                        # (N, 3)
            vel,                        # (N, 3)
            rot_mat,                    # (N, 9)
            rpy_rates,                  # (N, 3)
            rel_pos_gate.reshape(N, -1),# (N, 12)
            rel_xy_obst_gaus,           # (N, 2)
            action                      # (N, 4)
        ], axis=-1).astype(np.float32)  # => (N, 36)

        # save to self just in case
        self.rel_pos_gate      = rel_pos_gate            # (N, 4, 3)
        self.rel_xy_obst_gaus  = rel_xy_obst_gaus        # (N, 2)

        return state
    
    # region Reward
    def _reward(self, obs: dict, obs_rl: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        Args:
            obs    : dict[str, np.ndarray]   # 字段均为 (N, …)
            obs_rl : np.ndarray, shape (N, 36)
            act    : np.ndarray, shape (N, 4)

        Returns:
            rewards: np.ndarray, shape (N,)
        """
        N = act.shape[0]
        rewards = np.full(N, self.k_living, dtype=float)
        # data preparation
        curr_gate   = obs["target_gate"].astype(int)          # (N,)
        drone_pos   = obs["pos"]                              # (N, 3)
        drone_vel   = obs["vel"]                              # (N, 3)
        gates_pos   = obs["gates_pos"]                        # (N, n_gates, 3)
        gates_quat  = obs["gates_quat"]                       # (N, n_gates, 4)
        rel_xy_obst = obs_rl[:, -6:-4]                        # (N, 2)
        obst_xy     = self.rel_xy_obst_gaus + drone_pos[:, :2]  # (N, 2)
        gate_pos = gates_pos[np.arange(N), curr_gate]         # (N, 3)
        gate_quat = gates_quat[np.arange(N), curr_gate]       # (N, 4)
        gates_norm = R.from_quat(gate_quat).as_matrix()[:, :, 1]  # (N, 3)
        rel_gate   = gate_pos - drone_pos                     # (N, 3)
        # reward: success passing gates | handle gate switching
        prev_gate_delta = (curr_gate != self._prev_gate)       # (N,) bool
        rewards[prev_gate_delta] += self.k_success            # gate pass reward
        self.prev_gate_pos[prev_gate_delta] = gate_pos[prev_gate_delta] # update changed gates position
        # reward: obstacle distance P&D
        r_obst   = -self.k_obst   * np.linalg.norm(rel_xy_obst, axis=1)
        dist_now   = np.linalg.norm(obst_xy - drone_pos[:, :2], axis=1)
        dist_prev  = np.linalg.norm(self.prev_obst_xy - self.prev_drone_pos[:, :2], axis=1)
        r_obst_d = -self.k_obst_d * np.linalg.norm(rel_xy_obst, axis=1) * (dist_prev - dist_now)
        # reward: gates distance D
        prev_gate_delta_pos = np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos, axis=1)
        curr_gate_delta_pos = np.linalg.norm(rel_gate, axis=1)
        r_gates = self.k_gates * (prev_gate_delta_pos - curr_gate_delta_pos)
        # reward: deviation from gate center line
        dot_rg = (rel_gate * gates_norm).sum(axis=1)
        r_center = -self.k_center * np.linalg.norm(rel_gate - gates_norm * dot_rg[:, None], axis=1) \
                / (np.linalg.norm(rel_gate, axis=1) + 1e-6)
        # reward: smooth action
        r_act = -self.k_act * np.linalg.norm(act, axis=1) \
                -self.k_act_d * np.linalg.norm(act - self.prev_act, axis=1)
        # reward: velocity related
        r_vel = self.k_vel * (1 + np.linalg.norm(rel_xy_obst, axis=1) - r_obst_d) \
                * np.linalg.norm(drone_vel, axis=1)
        # reward: yaw angle
        yaw = np.abs(R.from_quat(obs["quat"]).as_euler('zyx', degrees=False)[:, 0])
        r_yaw = -self.k_yaw * yaw

        # sum up
        rewards += r_obst + r_obst_d + r_gates + r_center + r_act + r_vel + r_yaw

        # reward: immitation learning
        if IMMITATION_LEARNING:
            self.teacher_controller.compute_control(obs, None)
            demo_action = np.array(
                [self.teacher_controller.compute_control(
                    {k: v[i] for k, v in obs.items()}, None) - self.act_bias
                for i in range(N)]
            )
            r_imit = -self.k_imit * np.linalg.norm(demo_action - act, axis=1)
            rewards += r_imit

        # reward debug
        # if self._tick % 10 == 0:
        #     i = 0
        #     print(
        #         f"[env {i}] obst:{r_obst[i]:+.3f} | obst_d:{r_obst_d[i]:+.3f} | "
        #         f"gates:{r_gates[i]:+.3f} | center:{r_center[i]:+.3f} | "
        #         f"act:{r_act[i]:+.3f} | vel:{r_vel[i]:+.3f} | yaw:{r_yaw[i]:+.3f} | "
        #         f"total:{rewards[i]:+.3f}"
        #     )
        
        # update saving
        self.prev_act        = act
        self._prev_gate      = curr_gate
        self.prev_gate_pos   = gate_pos
        self.prev_obst_xy    = obst_xy
        self.prev_drone_pos  = drone_pos

        return rewards.astype(np.float32)
