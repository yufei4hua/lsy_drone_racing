from posixpath import relpath
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.jax_to_numpy import jax_to_numpy
from lsy_drone_racing.envs.drone_race import DroneRaceEnv, VecDroneRaceEnv
from jax import Array
import jax.numpy as jp
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils import draw_line

from jax import Array
from typing import Dict, Tuple

IMMITATION_LEARNING = True
if IMMITATION_LEARNING:
    from pathlib import Path
    from lsy_drone_racing.utils import load_config
    from rl_teacher_policy_att_pid import AttitudeController
RAND_INIT = False

class RLDroneRacingWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, 
                 env: VecDroneRaceEnv,
                 args):
        super().__init__(env)
        # turn off autoreset
        env.unwrapped.autoreset = False
        self.marked_for_reset = np.zeros(env.num_envs, dtype=bool)
        # create action & observation spaces
        self._num_envs = env.num_envs
        self.action_space = env.action_space
        state_dim = 40
        lim = np.full(state_dim, np.inf, dtype=np.float32) # set to infinite for now
        self.single_observation_space = spaces.Box(-lim, lim, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self._num_envs)
        # initialize internal state saving variables
        self.obs_env = None
        self._d_safe = 1.0
        self._act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_drone_pos = np.zeros((self._num_envs, 3), dtype=np.float32)         # (N, 3)
        self._prev_obst_xy   = np.zeros((self._num_envs, 2), dtype=np.float32)         # (N, 2)
        self._prev_act       = np.repeat(self._act_bias[None, :], self._num_envs, axis=0)   # (N, A)
        self._prev_gate      = np.zeros(self._num_envs, dtype=int)                          # (N,)
        self._prev_gate_pos  = np.zeros((self._num_envs, 3), dtype=np.float32)         # (N, 3)
        self._steps = np.zeros(self._num_envs, dtype=int)
        # fetch properties and methods from core env
        self.sim = self.find_attr(env, 'sim')
        self._reset_env_data = self.find_attr(env, '_reset_env_data')
        self.obs = self.find_attr(env, 'obs')
        self.info = self.find_attr(env, 'info')
        # assign all rl coef to self
        for k, v in vars(args).items():
            if k.startswith("k_"):
                setattr(self, k, v)

    # region Reset
    @staticmethod
    def find_attr(env, attr_name):
        while hasattr(env, 'unwrapped'):
            if hasattr(env, attr_name):
                return getattr(env, attr_name)
            env = env.unwrapped
        raise AttributeError(f"Attribute '{attr_name}' not found.")

    def _reset(self, seed=None, options=None, mask=None):
        # random initialization
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset(mask=mask)

        if RAND_INIT:
            mask = mask if mask is not None else jp.ones(self.unwrapped.unwrapped.data.steps.shape, dtype=bool)
            num_reset = mask.sum()
            # manually recorded init points
            self.rand_init_list = [
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0}, # emphasize takeoff point
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                # {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                {'pos': jp.array([1.0, 1.5, 0.07]), 'vel': jp.array([0.0, 0.0, 0.0]), 'quat': jp.array([0.0, 0.0, 0.0, 1.0]), 'f_thrust': 0.3, 'target_gate': 0},
                {'pos': jp.array([0.9081, 1.1422, 0.2201]), 'vel': jp.array([-0.2142, -0.7419, 0.2087]), 'quat': jp.array([0.1611, -0.0436, 0.0031, 0.9860]), 'f_thrust': 0.3179, 'target_gate': 0},
                {'pos': jp.array([0.7550, 0.6635, 0.3080]), 'vel': jp.array([-0.2109, -0.7631, 0.1146]), 'quat': jp.array([0.0452, 0.0307, -0.0066, 0.9985]), 'f_thrust': 0.2883, 'target_gate': 0},
                {'pos': jp.array([0.2309, -1.1061, 1.0188]), 'vel': jp.array([0.1798, -0.5673, 0.4537]), 'quat': jp.array([-0.0357, 0.0800, 0.0031, 0.9965]), 'f_thrust': 0.2255, 'target_gate': 1},
                {'pos': jp.array([0.5624, -1.2678, 1.1197]), 'vel': jp.array([1.0049, 0.1084, 0.1169]), 'quat': jp.array([-0.0709, 0.0366, -0.0009, 0.9968]), 'f_thrust': 0.2705, 'target_gate': 1},
                {'pos': jp.array([1.1311, -0.8747, 1.1062]), 'vel': jp.array([0.0588, 1.0162, -0.1100]), 'quat': jp.array([-0.0605, -0.1642, -0.0146, 0.9845]), 'f_thrust': 0.2624, 'target_gate': 2},
                {'pos': jp.array([0.6138, -0.0001, 0.8368]), 'vel': jp.array([-0.5123, 0.6669, -0.3205]), 'quat': jp.array([-0.0417, -0.0282, 0.0048, 0.9987]), 'f_thrust': 0.2299, 'target_gate': 2},
                {'pos': jp.array([0.0045, 0.9539, 0.4696]), 'vel': jp.array([-0.1742, 0.8196, -0.0696]), 'quat': jp.array([0.1123, 0.0797, -0.0008, 0.9905]), 'f_thrust': 0.2878, 'target_gate': 2},
                {'pos': jp.array([-0.0996, 0.9104, 0.5883]), 'vel': jp.array([-0.3977, -0.8926, 0.0738]), 'quat': jp.array([0.0938, -0.0228, -0.0006, 0.9953]), 'f_thrust': 0.2662, 'target_gate': 3},
                {'pos': jp.array([-0.2380, 0.5384, 0.7121]), 'vel': jp.array([-0.3728, -1.1330, 0.7930]), 'quat': jp.array([-0.0511, -0.0051, -0.0002, 0.9987]), 'f_thrust': 0.3220, 'target_gate': 3},
            ]
            # randomly pick one init points for envs to be reset
            rand_indices = np.random.randint(len(self.rand_init_list), size=int(num_reset))
            init_pos = jp.stack([self.rand_init_list[i]['pos'] for i in rand_indices])
            init_vel = jp.stack([self.rand_init_list[i]['vel'] for i in rand_indices])
            init_quat = jp.stack([self.rand_init_list[i]['quat'] for i in rand_indices])
            target_gate = jp.array([self.rand_init_list[i]['target_gate'] for i in rand_indices])
            
            self.sim.data = self.sim.data.replace(
                states=self.sim.data.states.replace(
                    pos=self.sim.data.states.pos.at[mask].set(init_pos[:,None,:]),
                    vel=self.sim.data.states.vel.at[mask].set(init_vel[:,None,:]),
                    quat=self.sim.data.states.quat.at[mask].set(init_quat[:,None,:]),
                )
            )
        self.unwrapped.unwrapped.data = self._reset_env_data(self.unwrapped.unwrapped.data, self.sim.data.states.pos, mask) # NOTE: self.unwrapped.unwrapped.data and self.sim.data are different
        
        if RAND_INIT:
            # correct self.unwrapped.unwrapped.data after _reset_env_data()
            self.unwrapped.unwrapped.data = self.unwrapped.unwrapped.data.replace(
                target_gate=self.unwrapped.unwrapped.data.target_gate.at[mask].set(target_gate[:,None])
            )
            pass

        return self.obs(), self.info()


    def reset(self, *, seed: int | None = None, options: dict | None = None, mask: Array | None = None) -> Tuple[np.ndarray, Dict]:
        mask = mask if mask is not None else np.ones(self._num_envs, dtype=bool)
        # call lower level reset
        obs, info = self._reset(seed=seed, mask=mask)
        self.obs_env = {k: jax_to_numpy(v[:, 0]) for k, v in obs.items()}
        info = {k: jax_to_numpy(v[:, 0]) for k, v in info.items()}
        state = self._obs_to_state(self.obs_env,
                                   np.repeat(self._act_bias[None, :], self._num_envs, axis=0))
        # reset storage
        self._prev_drone_pos[mask] = self.obs_env["pos"][mask]
        self._prev_obst_xy[mask]   = self.obs_env["obstacles_pos"][mask, 0, :2]
        self._prev_act[mask]       = self._act_bias
        self._prev_gate[mask]      = int(0)
        self._prev_gate_pos[mask]  = self.obs_env["gates_pos"][mask, 0]  
        self._steps[mask] = int(0)
        if mask is None or mask[0]: # if the first world is reset
            self.traj_record = self.obs_env['pos'][0, :] # debug trajectory
        # setup teacher policy
        if IMMITATION_LEARNING:
            config = load_config(Path(__file__).parents[2] / "config/level0.toml")
            self.teacher_controller = AttitudeController(self.obs_env, info, config, self)
        return state, info

    # region Step
    def step(self, action: np.ndarray):
        if IMMITATION_LEARNING: # test teacher policy
            action = self.teacher_controller.compute_control(self.obs_env, None) - self._act_bias
        action_exec = action + self._act_bias
        self.obs_env, _, terminated, truncated, info = self.env.step(action_exec)
        state = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, state, action)
        # self handle autoreset
        if self.marked_for_reset.any():
            # add crash & finish reward
            r_crash = -self.k_crash * (self.marked_for_reset & (self.obs_env["target_gate"] >= 0))
            r_finish = self.k_finish * (self.marked_for_reset & (self.obs_env["target_gate"] < 0))
            reward += r_crash + r_finish
            # reset specific world
            state, info = self.reset(mask=self.marked_for_reset)
            terminated = terminated & ~self.marked_for_reset
            truncated = truncated & ~self.marked_for_reset
        done = terminated | truncated
        self.marked_for_reset = done # update mask after reset

        self.traj_record = np.vstack([self.traj_record, self.obs_env['pos'][0, :]]) # debug trajectory
        self._steps += 1
        try:
            draw_line(self, self.traj_record[0:-1:5], rgba=np.array([0.0, 1.0, 0.0, 0.2]))
        except:
            pass

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
        progress_onehot = np.zeros((N, 4), dtype=np.float32)
        progress_onehot[np.arange(N), curr_gate_idx] = 1.0 # (N, 4)

        half_w, half_h = 0.2, 0.2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners = (gate_rot_mat @ corners_local.T).transpose(0, 2, 1) + gate_pos[:, None, :]
        draw_line(self, np.stack([gate_corners[0, 0], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 1], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 2], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners[0, 3], pos[0]]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        rel_pos_gate = gate_corners - pos[:, None, :]   # (N, 4, 3)

        obst_rel_xy = obs["obstacles_pos"][:, :, :2] - pos[:, None, :2]       # (N, n_obst, 2)
        obst_dists  = np.linalg.norm(obst_rel_xy, axis=-1)                    # (N, n_obst)
        closest_idx = obst_dists.argmin(axis=-1)                              # (N,)
        rel_xy_obst = obst_rel_xy[np.arange(N), closest_idx]                  # (N, 2)
        # dist        = obst_dists[np.arange(N), closest_idx]                   # (N,)
        # rel_xy_obst_gaus = rel_xy_obst * np.exp(-(dist / (0.5 * self._d_safe))**2)[:, None] \
        #                 / (dist[:, None] + 1e-6)                              # (N, 2) # depricated

        rot_mat  = R.from_quat(quat).as_matrix().reshape(N, -1)               # (N, 9)
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)                          # (N, 3)

        # EXP: manually normalize obs
        obs_norm_coef = {
            "pos": 0.6,
            "vel": 0.2,
            "rot_mat": 1.0,
            "rpy_rates": 0.3,
            "rel_pos_gate": 0.5,
            "rel_xy_obst": 0.5,
            "progress_onehot": 1.0,
            "action": np.array([2.0, 1.5, 1.5, 2.0]),
        }

        state = np.concatenate([
            pos * obs_norm_coef["pos"], # (N, 3)
            vel * obs_norm_coef["vel"], # (N, 3)
            rot_mat * obs_norm_coef["rot_mat"], # (N, 9)
            rpy_rates * obs_norm_coef["rpy_rates"], # (N, 3)
            rel_pos_gate.reshape(N, -1) * obs_norm_coef["rel_pos_gate"], # (N, 12)
            rel_xy_obst * obs_norm_coef["rel_xy_obst"], # (N, 2)
            progress_onehot * obs_norm_coef["progress_onehot"], # (N, 4)
            action * obs_norm_coef["action"] # (N, 4)
        ], axis=-1).astype(np.float32) # => (N, 40)

        # save to self just in case
        self.rel_pos_gate = rel_pos_gate     # (N, 4, 3)
        self.rel_xy_obst  = rel_xy_obst      # (N, 2)

        return state
    
    # region Reward
    def _reward(self, obs: dict, obs_rl: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        Args:
            obs    : dict[str, np.ndarray]
            obs_rl : np.ndarray, shape (N, 36)
            act    : np.ndarray, shape (N, 4)

        Returns:
            rewards: np.ndarray, shape (N,)
        """
        N = act.shape[0]
        rewards = np.full(N, self.k_alive, dtype=float)
        rewards = rewards * (self.k_alive_anneal ** self._steps)
        # data preparation
        curr_gate   = obs["target_gate"].astype(int)                # (N,)
        drone_pos   = obs["pos"]                                    # (N, 3)
        drone_vel   = obs["vel"]                                    # (N, 3)
        gates_pos   = obs["gates_pos"]                              # (N, n_gates, 3)
        gates_quat  = obs["gates_quat"]                             # (N, n_gates, 4)
        rel_xy_obst = obs_rl[:, 30:32]                              # (N, 2)
        dist_obst = np.linalg.norm(rel_xy_obst, axis=1, keepdims=True)
        rel_xy_obst_gaus = rel_xy_obst * np.exp(-(dist_obst / (0.5 * self._d_safe))**2) \
                        / (dist_obst + 1e-6)               # (N, 2)
        obst_xy     = self.rel_xy_obst + drone_pos[:, :2]           # (N, 2)
        gate_pos = gates_pos[np.arange(N), curr_gate]               # (N, 3)
        gate_quat = gates_quat[np.arange(N), curr_gate]             # (N, 4)
        gates_norm = R.from_quat(gate_quat).as_matrix()[:, :, 1]    # (N, 3)
        rel_gate   = gate_pos - drone_pos                           # (N, 3)
        proj_norm = gates_norm * (rel_gate * gates_norm).sum(axis=1, keepdims=True) # vector projected to gate normal
        proj_tang = rel_gate - proj_norm # vector from drone to gate center line
        ## A. gate related
        # 1. success passing gates | handle gate switching
        prev_gate_delta = (curr_gate != self._prev_gate)   # (N,) bool
        rewards[prev_gate_delta] += self.k_success         # gate pass reward
        # 2. position based ellipsoid shape reward
        norm2 =  (proj_norm ** 2).sum(axis=1)
        tang2 =  (proj_tang ** 2).sum(axis=1)
        r_pos = self.k_pos * np.exp(-(norm2 / (self.k_ellip_norm**2) + tang2 / (self.k_ellip_tang**2)))
        # 3. velocity based rewards
        #   3.1 gates approaching velocity
        rel_gate_norm = rel_gate / (np.linalg.norm(rel_gate, axis=1, keepdims=True) + 1e-6) # (N, 3)
        r_gates = self.k_gates * (drone_vel * rel_gate_norm).sum(axis=1) # velocity projecting to gate direction
        #   3.2 deviation from gate center line
        r_center_d = self.k_center_d * (drone_vel * proj_tang).sum(axis=1)
        #   3.3 detour at gate sides
        ratio = (proj_norm ** 2).sum(axis=1) / (np.sum(rel_gate**2, axis=1) + 1e-6) # sin(theta)^2
        exp_factor = np.exp(-self.k_detour_scale * ratio)
        r_detour = self.k_detour * (-(drone_vel * gates_norm).sum(axis=1)) * exp_factor
        ## B. obstacle related
        # 1. distance based penalty
        r_obst   = -self.k_obst * np.linalg.norm(rel_xy_obst_gaus, axis=1)
        # 2. velocity based penalty
        r_obst_d = -self.k_obst_d * (drone_vel[:, :2] * rel_xy_obst_gaus).sum(axis=1)
        ## C. other rewards
        # 1. action smoothness
        r_act = -self.k_act * np.linalg.norm(act, axis=1) \
                -self.k_act_d * np.linalg.norm(act - self._prev_act, axis=1)
        # 2. velocity magnitude
        r_vel = self.k_vel * np.linalg.norm(drone_vel, axis=1)
        # 3. yaw angle penalty
        yaw = np.abs(R.from_quat(obs["quat"]).as_euler('zyx', degrees=False))[:, 0]
        r_yaw = -self.k_yaw * yaw

        # sum up
        rewards += r_pos + r_gates + r_center_d + r_detour + r_obst + r_obst_d + r_act + r_vel + r_yaw

        # reward: immitation learning
        if IMMITATION_LEARNING:
            demo_action = self.teacher_controller.compute_control(self.obs_env, None) - self._act_bias
            r_imit = -self.k_imit * np.linalg.norm(demo_action - act, axis=1)
            rewards += r_imit

        # # reward debug
        # i = 0
        # print(
        #     f"alive:{self.k_alive * self.k_alive_anneal ** self._steps[i]:+.3f} | "
        #     # f"obst:{r_obst[i]:+.3f} | obst_d:{r_obst_d[i]:+.3f} | "
        #     f"pos:{r_pos[i]:+.3f} | gates:{r_gates[i]:+.3f} | center_d:{r_center_d[i]:+.3f} | "
        #     f"detour:{r_detour[i]:+.3f} | "
        #     f"pass:{(self.k_success if prev_gate_delta[i] else 0.0):+.3f} | act:{r_act[i]:+.3f} | vel:{r_vel[i]:+.3f} | yaw:{r_yaw[i]:+.3f}"
        #     + (f" | imit:{r_imit[i]:+.3f}" if IMMITATION_LEARNING else "")
        #     + f"\n |position:{r_pos[i]:+.3f} | velocity:{r_gates[i]+r_center_d[i]+r_detour[i]:+.3f} | total:{rewards[i]:+.3f}"
        # )
        
        # update saving
        self._prev_act        = act
        self._prev_gate       = curr_gate
        self._prev_obst_xy    = obst_xy
        self._prev_drone_pos  = drone_pos

        return rewards.astype(np.float32)
