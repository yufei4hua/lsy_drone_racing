"""Single drone racing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jp
import gymnasium
from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from packaging.version import Version
from lsy_drone_racing.envs.utils import gate_passed, load_track
from ml_collections import ConfigDict

from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.envs.race_core import RaceCoreEnv, build_action_space
from crazyflow.constants import GRAVITY, MASS
from crazyflow.sim.physics import ang_vel2rpy_rates
from lsy_drone_racing.utils import draw_line
from lsy_drone_racing.control.attitude_controller import AttitudeController
from lsy_drone_racing.control.easy_controller import EasyController

if TYPE_CHECKING:
    from jax import Array
    from ml_collections import ConfigDict
AutoresetMode = None
if Version(gymnasium.__version__) >= Version("1.1"):
    from gymnasium.vector import AutoresetMode

IMMITATION_LEARNING = False
if IMMITATION_LEARNING:
    from pathlib import Path
    from lsy_drone_racing.utils import load_config

class RLDroneRaceEnv(RaceCoreEnv, Env):
    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            n_envs=1,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(control_mode)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [5.0]*2 + [np.pi]*3)
        self.observation_space = spaces.Box(low=-lim, high=lim, shape=(36,), dtype=np.float32)
        self.autoreset = False

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate = 0
        self.prev_gate_pos = None
        self.prev_obst_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        self._tick = 0
        self.d_safe = 1.0
        # parameters setting
        self.k_gates = 1.0
        self.k_center = 0.2
        self.k_vel = -0.01
        self.k_act = 2e-4
        self.k_act_d = 1e-4
        self.k_yaw = 0.02
        self.k_crash = 20
        self.k_success = 25
        self.k_imit = 0.0
        # public variables
        self.obs_env = None
        self.obs_rl = None

    def step(self, action):
        # if IMMITATION_LEARNING: # test
        #     action = self.teacher_controller.compute_control(self.obs_env, None) - self.act_bias
        #     self.teacher_controller._tick += 1
        self._tick += 1
        action_exec = action + self.act_bias
        self.obs_env, _, terminated, truncated, info = self._step(action_exec)
        self.obs_env = {k: np.array(v[0, 0]) for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.traj_record = np.vstack([self.traj_record, self.obs_env['pos'][None, :]])
        self.obs_rl = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, self.obs_rl, action)
        if (terminated or truncated) and self.obs_env['target_gate'] >= 0:
            reward -= self.k_crash
        return self.obs_rl, reward, bool(terminated[0, 0]), bool(truncated[0, 0]), info

    def _obs_to_state(self, obs: dict[str, NDArray], action: Array) -> NDArray:
        # define rl input states: [pos(3), vel(3), rot_mat(9), rpy_rates(3), rel_pos_gate(4*3), rel_xy_obst(2), prev_act(4)] (dim=36)
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        # calc vectors pointing to four gate corners
        curr_gate = obs['target_gate']
        self.gates_size = [0.4, 0.4] # [width, height]
        self.gate_rot_mat = np.array(R.from_quat(obs['gates_quat'][curr_gate]).as_matrix())
        half_w, half_h = self.gates_size[0] / 2, self.gates_size[1] / 2
        corners_local = np.array([
            [-half_w, 0.0,  half_h],
            [ half_w, 0.0,  half_h],
            [-half_w, 0.0, -half_h],
            [ half_w, 0.0, -half_h],
        ])
        gate_corners_pos = (self.gate_rot_mat @ corners_local.T).T + obs['gates_pos'][curr_gate]  # shape: (4, 3)
        draw_line(self, np.stack([gate_corners_pos[0], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners_pos[1], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners_pos[2], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        draw_line(self, np.stack([gate_corners_pos[3], pos]), rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        if self.traj_record.shape[0] >= 2:
            draw_line(self, self.traj_record, rgba=np.array([0.0, 1.0, 0.0, 0.2]))
        
        self.rel_pos_gate = gate_corners_pos - pos[None, :]  # shape: (4, 3)
        # self.rel_pos_gate_norm = self.rel_pos_gate / np.linalg.norm(self.rel_pos_gate, axis=1, keepdims=True) # normalize

        obst_rel_xy_list = np.array(obs['obstacles_pos'])[:,:2] - pos[:2]
        obst_dists = np.linalg.norm(obst_rel_xy_list, axis=-1)
        closest_obst_idx = np.argmin(obst_dists)
        self.rel_xy_obst = obst_rel_xy_list[closest_obst_idx]
        dist = obst_dists[closest_obst_idx]
        self.rel_xy_obst = self.rel_xy_obst * np.exp(-(dist/(0.5*self.d_safe))**2) / (dist+1e-6)
        try:
            draw_line(self, np.stack([np.concatenate([self.rel_xy_obst, np.array([0])])+pos, pos]), rgba=np.array([1.0, 0.0, 1.0, 0.5]))
        except:
            pass
        # self.rel_xy_obst = np.zeros(2) # for lesson 1 without pillar
        
        # calc euler
        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        
        # ang_vel to rpy_rates
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)
        
        state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1), self.rel_xy_obst, action])
        # state = np.concatenate([pos, vel, rot_mat, rpy_rates, self.rel_pos_gate.reshape(-1), action]) # old version
        return state

    def reset_prev(self):
        self._tick = 0
        self.prev_gate = 0
        self.prev_gate_pos = self.obs_env['gates_pos'][0]
        self.prev_obst_pos = self.obs_env['obstacles_pos'][0]
        self.prev_drone_pos = self.obs_env['pos']
        self.prev_act = self.act_bias
        self.traj_record = np.array([self.obs_env['pos']])

    def reset(self, seed=None, options=None):
        # parameters setting # 放到这儿好调参
        self.k_obst = 0.4
        self.k_obst_d = 1
        self.k_gates = 1.0
        self.k_center = 0.3
        self.k_vel = -0.3
        self.k_act = 0.002
        self.k_act_d = 0.0001
        self.k_yaw = 0.1
        self.k_crash = 30
        self.k_success = 40
        # TODO: random reset at different racing process
        self.obs_env, info = self._reset(seed=seed, options=options)
        self.obs_env = {k: np.array(v[0, 0]) for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.reset_prev()
        self.tick_nearest = 0
        self.obs_rl = self._obs_to_state(self.obs_env, self.act_bias)
        # if IMMITATION_LEARNING:
        #     config = load_config(Path(__file__).parents[2] / "config/level0.toml")
        #     # self.teacher_controller = AttitudeController(self.obs_env, info, config)
        #     self.teacher_controller = EasyController(self.obs_env, info, config)
        #     self.prev_ref_waypoint = self.teacher_controller.get_trajectory_waypoints()[0]
        return self.obs_rl, info
    
    def _reward(self, obs, obs_rl, act):
        curr_gate = obs['target_gate']
        gate_pos = obs['gates_pos'][curr_gate]
        gates_norm = np.array(R.from_quat(obs['gates_quat']).as_matrix())[:,:,1].squeeze()
        obst_pos = obs['obstacles_pos'][0]
        drone_pos = obs['pos']
        drone_vel = obs['vel']
        rel_xy_obst = obs_rl[-6:-4] # gaussian length
        rel_gate = gate_pos - drone_pos
        r = 0.25
        if curr_gate != self.prev_gate: # handle gate switching
            self.prev_gate_pos = gate_pos
            r += self.k_success
        r_obst = -self.k_obst * np.linalg.norm(rel_xy_obst)
        r_obst_d = -self.k_obst_d * (np.linalg.norm(rel_xy_obst)) * (np.linalg.norm(self.prev_obst_pos - self.prev_drone_pos) - np.linalg.norm(obst_pos - drone_pos))
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(rel_gate))
        r_center = -self.k_center * (1 - np.abs(np.dot(rel_gate, gates_norm))/np.linalg.norm(rel_gate))
        r_center = -self.k_center * np.linalg.norm(rel_gate - gates_norm*np.dot(rel_gate, gates_norm))/np.linalg.norm(rel_gate) # err dist to center line
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        r_vel = self.k_vel * (1+np.linalg.norm(rel_xy_obst)-r_obst_d) * np.linalg.norm(drone_vel)
        r_yaw = -self.k_yaw * np.fabs(R.from_quat(obs['quat']).as_euler('zyx', degrees=False)[0])

        print(
            f"obst: {r_obst:.4f} | obst_d: {r_obst_d:.4f} | gates: {r_gates:.4f} | center: {r_center:.4f} | "
            f"act: {r_act:.4f} | vel: {r_vel:.4f} | yaw: {r_yaw:.4f} | "
            f"total: {(r_obst + r_gates + r_center + r_act + r_vel + r_yaw):.4f}"
        )

        self.prev_act = act
        self.prev_gate = curr_gate
        self.prev_gate_pos = gate_pos
        self.prev_obst_pos = obst_pos
        self.prev_drone_pos = drone_pos
        r += r_gates + r_center + r_act + r_vel + r_yaw + r_obst + r_obst_d
        return r
    
        # if IMMITATION_LEARNING:
        #     k_imit_p, k_imit_d = 0.3, 1.0
        #     # tracking the waypoint a bit ahead current position
        #     waypoints = self.teacher_controller.get_trajectory_waypoints()
        #     ref_tick, ref_waypoint = self._find_leading_waypoint(waypoints, drone_pos, 0.2, curr_gate)
        #     r_imit_p = -k_imit_p * np.linalg.norm(ref_waypoint - drone_pos)
        #     r_imit_d = k_imit_d * (np.linalg.norm(self.prev_ref_waypoint - self.prev_drone_pos) - np.linalg.norm(ref_waypoint - drone_pos))
        #     r_imit = self.k_imit * (r_imit_p + r_imit_d)
        #     # draw_line(self, np.stack([ref_waypoint, drone_pos]), rgba=np.array([int(r_imit_d<0), int(r_imit_d>0), 0.0, 1.0]))
        #     self.prev_ref_waypoint = ref_waypoint
        #     # # action diff from teacher action (incompatible with state controller)
        #     self.teacher_controller.set_tick(ref_tick)
        #     self.teacher_controller.compute_control(obs, None) # only to update trajectory
        #     # demo_action = self.teacher_controller.compute_control(obs, None) - self.act_bias
        #     # r_imit = -self.k_imit * np.linalg.norm(demo_action - act)
        #     r += r_imit
    
# for curriculum learning
class RLDroneHoverEnv(RLDroneRaceEnv):
    def __init__(
        self,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.action_space = build_action_space(control_mode)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [5.0]*2 + [np.pi]*3)
        self.observation_space = spaces.Box(low=-lim, high=lim, shape=(36,), dtype=np.float32)
        self.autoreset = False

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate = 0
        self.prev_gate_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        self._tick = 0
        # public variables
        self.obs_env = None
        self.obs_rl = None

        # Lesson 1: passing random gates
        

    def rand_fake_gate(self, pos, max_dis, max_z, max_ang):
        theta = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.0, max_dis)
        dx, dy = distance * np.cos(theta), distance * np.sin(theta)
        dz = np.random.uniform(-max_z, max_z)

        gate_pos = np.array(pos) + np.array([dx, dy, dz])

        yaw = theta-np.pi/2
        yaw = yaw + 2*np.pi if yaw < 0 else yaw

        yaw += np.random.uniform(-max_ang, max_ang)

        gate_rpy = [0.0, 0.0, yaw]

        return gate_pos.tolist(), gate_rpy, R.from_euler("xyz", gate_rpy).as_quat().astype(np.float32)
    
    def rand_fake_obstacles(self, pos, gate_pos, ang_range, safe_dis):
        pos = np.array(pos)
        gate_pos = np.array(gate_pos)
        vec = gate_pos[:2] - pos[:2]
        total_dist = np.linalg.norm(vec)

        if total_dist <= 2 * safe_dis:
            safe_dis = total_dist/2 - 0.1

        angle_center = np.arctan2(vec[1], vec[0])

        angle_offset = np.random.uniform(-ang_range, ang_range)
        radius = np.random.uniform(safe_dis, total_dist - safe_dis)

        angle = angle_center + angle_offset
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)

        obstacle_xy = pos[:2] + [dx, dy]
        obstacle_z = gate_pos[2] + 1

        return [obstacle_xy[0], obstacle_xy[1], obstacle_z]

    def _reset(self, seed=None, options=None, mask=None, init_pos:Array=None):
        if seed is not None:
            self.sim.seed(seed)

        self.sim.reset(mask=mask)

        if init_pos is not None:
            pos = self.sim.data.states.pos.at[...].set(jp.array(init_pos))
            self.sim.data = self.sim.data.replace(
                states=self.sim.data.states.replace(pos=pos)
            )

        gate_pos, gate_rpy, gate_quat = self.rand_fake_gate(init_pos, max_dis=2.0, max_z=0.75, max_ang=0.7) # generate random gates
        new_mocap_pos = self.sim.data.mjx_data.mocap_pos.at[:, self.gates["mj_ids"][0]].set(jp.array(gate_pos))
        new_mocap_quat = self.sim.data.mjx_data.mocap_quat.at[:, self.gates["mj_ids"][0]].set(jp.array([gate_quat[3], gate_quat[0], gate_quat[1], gate_quat[2]]))  # to MuJoCo order

        obstacle_pos = self.rand_fake_obstacles(init_pos, gate_pos, ang_range=0.3, safe_dis=0.7)
        new_mocap_pos = new_mocap_pos.at[:, self.obstacles["mj_ids"][0]].set(jp.array(obstacle_pos))

        self.sim.data = self.sim.data.replace(
            mjx_data=self.sim.data.mjx_data.replace(
                mocap_pos=new_mocap_pos,
                mocap_quat=new_mocap_quat
            )
        )

        self.data = self._reset_env_data(self.data, self.sim.data.states.pos, mask)
        return self.obs(), self.info()
    
    def reset(self, seed=None, options=None, init_pos:NDArray=None):
        # parameters setting # 放到这儿好调参
        self.k_obst = 0.4
        self.k_obst_d = 1
        self.k_gates = 1.0
        self.k_center = 0.3
        self.k_vel = -0.3
        self.k_act = 0.002
        self.k_act_d = 0.0001
        self.k_yaw = 0.1
        self.k_crash = 30
        self.k_success = 40
        # TODO: random reset position
        init_pos = np.array([0.0, 0.0, 1.5])
        self.obs_env, info = self._reset(seed=seed, options=options, init_pos=init_pos)
        self.obs_env = {k: v[0, 0] for k, v in self.obs_env.items()}
        info = {k: v[0, 0] for k, v in info.items()}
        self.reset_prev()
        self.obs_rl = self._obs_to_state(self.obs_env, self.act_bias)
        return self.obs_rl, info


    def _reward(self, obs, obs_rl, act):
        curr_gate = obs['target_gate']
        gate_pos = obs['gates_pos'][curr_gate]
        gates_norm = np.array(R.from_quat(obs['gates_quat']).as_matrix())[:,:,1][0]
        obst_pos = obs['obstacles_pos'][0]
        drone_pos = obs['pos']
        drone_vel = obs['vel']
        rel_xy_obst = obs_rl[-6:-4] # gaussian length
        rel_gate = gate_pos - drone_pos
        r = 0.25
        if curr_gate != self.prev_gate:
            self.prev_gate_pos = gate_pos
            r += self.k_success
        r_obst = -self.k_obst * np.linalg.norm(rel_xy_obst)
        r_obst_d = -self.k_obst_d * (np.linalg.norm(rel_xy_obst)) * (np.linalg.norm(self.prev_obst_pos - self.prev_drone_pos) - np.linalg.norm(obst_pos - drone_pos))
        r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos - self.prev_drone_pos) - np.linalg.norm(rel_gate))
        r_center = -self.k_center * (1 - np.abs(np.dot(rel_gate, gates_norm))/np.linalg.norm(rel_gate))
        r_center = -self.k_center * np.linalg.norm(rel_gate - gates_norm*np.dot(rel_gate, gates_norm))/np.linalg.norm(rel_gate) # err dist to center line
        r_act = -self.k_act * np.linalg.norm(act) - self.k_act_d * np.linalg.norm(act - self.prev_act)
        r_vel = self.k_vel * (1+np.linalg.norm(rel_xy_obst)-r_obst_d) * np.linalg.norm(drone_vel)
        r_yaw = -self.k_yaw * np.fabs(R.from_quat(obs['quat']).as_euler('zyx', degrees=False)[0])

        # print(
        #     f"obst: {r_obst:.4f} | obst_d: {r_obst_d:.4f} | gates: {r_gates:.4f} | center: {r_center:.4f} | "
        #     f"act: {r_act:.4f} | vel: {r_vel:.4f} | yaw: {r_yaw:.4f} | "
        #     f"total: {(r_obst + r_gates + r_center + r_act + r_vel + r_yaw):.4f}"
        # )

        self.prev_act = act
        self.prev_gate = curr_gate
        self.prev_gate_pos = gate_pos
        self.prev_obst_pos = obst_pos
        self.prev_drone_pos = drone_pos
        r += r_gates + r_center + r_act + r_vel + r_yaw + r_obst + r_obst_d
        return r


from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=50, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            try:
                self.training_env.env_method("render", indices=0)
                # self.training_env.env_method("render", indices=1)
            except Exception as e:
                print(f"Render error: {e}")
        return True
    



#### not used for now
class VecRLDroneRaceEnv(RaceCoreEnv, VectorEnv):
    """Vectorized single-agent drone racing environment."""

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP if AutoresetMode else None}

    def __init__(
        self,
        num_envs: int,
        freq: int,
        sim_config: ConfigDict,
        track: ConfigDict,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        seed: int = 1337,
        max_episode_steps: int = 1500,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        """Initialize the vectorized single-agent drone racing environment.

        Args:
            num_envs: Number of worlds in the vectorized environment.
            freq: Environment step frequency.
            sim_config: Simulation configuration.
            track: Track configuration.
            sensor_range: Sensor range.
            control_mode: Control mode for the drones. See `build_action_space` for details.
            disturbances: Disturbance configuration.
            randomizations: Randomization configuration.
            seed: Random seed.
            max_episode_steps: Maximum number of steps per episode.
            device: Device used for the environment and the simulation.
        """
        super().__init__(
            n_envs=num_envs,
            n_drones=1,
            freq=freq,
            sim_config=sim_config,
            track=track,
            sensor_range=sensor_range,
            control_mode=control_mode,
            disturbances=disturbances,
            randomizations=randomizations,
            seed=seed,
            max_episode_steps=max_episode_steps,
            device=device,
        )
        self.num_envs = num_envs
        self.single_action_space = build_action_space(control_mode)
        self.action_space = batch_space(self.single_action_space, num_envs)
        lim = np.array([np.inf]*18 + [1.0]*12 + [1.0] + [np.pi]*3)
        self.single_observation_space = spaces.Box(low=-lim, high=lim, shape=(34,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.act_bias = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=np.float32)
        # record previous states for reward calculation
        self.prev_gate_pos = None
        self.prev_drone_pos = None
        self.prev_act = self.act_bias
        self.num_gates = 4
        self.gates_size = [0.4, 0.4] # [width, height]
        # parameters setting
        self.k_gates = 1.0
        self.k_act = 2e-4
        self.k_act_d = 1e-4
        self.k_crash = 10

    def reset(self, seed=None, options=None):
        self.obs_env, info = self._reset(seed=seed, options=options)
        self.prev_gate_pos = self.obs_env['gates_pos'][:, 0, :]
        self.prev_drone_pos = self.obs_env['pos'][:, 0, :]
        self.prev_act = np.tile(self.act_bias, (self.num_envs, 1))
        self.obs_rl = self._obs_to_state(self.obs_env, self.prev_act)
        return self.obs_rl, info

    def step(self, action):
        action_exec = action + self.act_bias
        self.obs_env, _, terminated, truncated, info = self._step(action_exec)
        self.obs_env = {k: v[:, 0] for k, v in self.obs_env.items()}
        info = {k: v[:, 0] for k, v in info.items()}
        self.obs_rl = self._obs_to_state(self.obs_env, action)
        reward = self._reward(self.obs_env, action)
        done = (terminated | truncated) & (self.obs_env['target_gate'] >= 0)
        reward -= self.k_crash * done.astype(float)
        return self.obs_rl, reward, terminated, truncated, info

    def _obs_to_state(self, obs, action): # handel vec env obs
        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"]
        ang_vel = obs["ang_vel"]

        state_list = []
        for i in range(self.num_envs):
            curr_gate = obs['target_gate'][i]
            gate_rot_mat = R.from_quat(obs['gates_quat'][i, curr_gate]).as_matrix()
            half_w, half_h = 0.4 / 2, 0.4 / 2
            corners_local = np.array([
                [-half_w, 0.0,  half_h],
                [ half_w, 0.0,  half_h],
                [-half_w, 0.0, -half_h],
                [ half_w, 0.0, -half_h],
            ])
            gate_corners_pos = (gate_rot_mat @ corners_local.T).T + obs['gates_pos'][i, curr_gate]
            rel_pos_gate = (gate_corners_pos - pos[i]) 
            rel_pos_gate = rel_pos_gate / np.linalg.norm(rel_pos_gate, axis=1, keepdims=True)
            rel_pos_gate = rel_pos_gate.reshape(-1)

            rot_mat = R.from_quat(quat[i]).as_matrix().reshape(-1)
            rpy_rates = ang_vel2rpy_rates(ang_vel[i], quat[i])

            state = np.concatenate([pos[i], vel[i], rot_mat, rpy_rates, rel_pos_gate, action[i]])
            state_list.append(state)
        return np.stack(state_list)

    def _reward(self, obs, act):
        reward = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            curr_gate = obs['target_gate'][i]
            gate_pos = obs['gates_pos'][i, curr_gate]
            drone_pos = obs['pos'][i]
            r_gates = self.k_gates * (np.linalg.norm(self.prev_gate_pos[i] - self.prev_drone_pos[i]) - np.linalg.norm(gate_pos - drone_pos))
            r_act = -self.k_act * np.linalg.norm(act[i]) - self.k_act_d * np.linalg.norm(act[i] - self.prev_act[i])
            reward[i] = r_gates + r_act
            self.prev_act[i] = act[i]
            self.prev_gate_pos[i] = gate_pos
            self.prev_drone_pos[i] = drone_pos
        return reward
