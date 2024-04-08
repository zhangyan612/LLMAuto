
# Given point cloud and gripper location, object target location
# Get robot path trajectory to reach target location

# full code of vox poser

import os
import pickle
import json
import hashlib
import yaml
from LMP import LMP
import numpy as np
from planners import PathPlanner
from scipy.ndimage import distance_transform_edt
import transforms3d
from transforms3d.quaternions import mat2quat
import copy
import time
import openai
from time import sleep
from openai.error import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from arguments import get_config
from envs.rlbench_env import VoxPoserRLBench
from rlbench import tasks
import plotly.graph_objects as go
import datetime
import open3d as o3d
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode
from rlbench.environment import Environment
import rlbench.tasks as tasks
from pyrep.const import ObjectType
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_config(env=None, config_path=None):
    assert env is None or config_path is None, 'env and config_path cannot be both specified'
    if config_path is None:
        assert env.lower() == 'rlbench'
        config_path = './configs/rlbench_config.yaml'
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_config(config_path)
    # wrap dict such that we can access config through attribute
    class ConfigDict(dict):
        def __init__(self, config):
            """recursively build config"""
            self.config = config
            for key, value in config.items():
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]
        def __getstate__(self):
            return self.config
        def __setstate__(self, state):
            self.config = state
            self.__init__(state)
    config = ConfigDict(config)
    return config


# creating some aliases for end effector and table in case LLMs refer to them differently
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']

class Controller:
    def __init__(self, env, config):
        self.config = config
        self.env = env
        self.dynamics_model = PushingDynamicsModel()
    
    def _calculate_ee_rot(self, pushing_dir):
        """
        Given a pushing direction, calculate the rotation matrix for the end effector
        It is offsetted such that it doesn't exactly point towards the direction but slanted towards table, so it's safer
        """
        pushing_dir = normalize_vector(pushing_dir)
        desired_dir = pushing_dir + np.array([0, 0, -np.linalg.norm(pushing_dir)])
        desired_dir = normalize_vector(desired_dir)
        left = np.cross(pushing_dir, desired_dir)
        left = normalize_vector(left)
        up = np.array(desired_dir, dtype=np.float32)
        up = normalize_vector(up)
        forward = np.cross(left, up)
        forward = normalize_vector(forward)
        rotmat = np.eye(3).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up
        quat_wxyz = mat2quat(rotmat)
        return quat_wxyz
    
    def _apply_mpc_control(self, control, target_velocity=1):
        """
        apply control to the object; depending on different control type
        """
        # calculate start and final ee pose
        contact_position = control[:3]  # [3]
        pushing_dir = control[3:6]  # [3]
        pushing_dist = control[6]  # [1]
        # calculate a safe end effector rotation
        ee_quat = self._calculate_ee_rot(pushing_dir)
        # calculate translation
        start_dist = 0.08
        t_start = contact_position - pushing_dir * start_dist
        t_interact = contact_position + pushing_dir * pushing_dist
        t_rest = contact_position - pushing_dir * start_dist * 0.8

        # apply control
        self.env.close_gripper()
        # move to start pose
        self.env.move_to_pose(np.concatenate([t_start, ee_quat]), speed=target_velocity)
        print('[controllers.py] moved to start pose', end='; ')
        # move to interact pose
        self.env.move_to_pose(np.concatenate([t_interact, ee_quat]), speed=target_velocity * 0.2)
        print('[controllers.py] moved to final pose', end='; ')
        # back to rest pose
        self.env.move_to_pose(np.concatenate([t_rest, ee_quat]), speed=target_velocity * 0.33)
        print('[controllers.py] back to release pose', end='; ')
        self.env.reset_to_default_pose()
        print('[controllers.py] back togenerate_random_control default pose', end='')
        print()

    def execute(self, movable_obs, waypoint):
        info = dict()
        target_xyz, target_rotation, target_velocity, target_gripper = waypoint
        object_centric = (not movable_obs['name'].lower() in EE_ALIAS)
        # move to target pose directly
        if not object_centric:
            target_pose = np.concatenate([target_xyz, target_rotation])
            result = self.env.apply_action(np.concatenate([target_pose, [target_gripper]]))
            info['mp_info'] = result
        # optimize through dynamics model to obtain robot actions
        else:
            start = time.time()
            # sample control sequence using MPC
            movable_obs = {key: value for key, value in movable_obs.items() if key in ['_point_cloud_world']}
            best_control, self.mpc_info = self.random_shooting_MPC(movable_obs, target_xyz)  # [7]
            print('[controllers.py] mpc search completed in {} seconds with {} samples'.format(time.time() - start, self.config.num_samples))
            # apply first control in the sequence
            self.mpc_velocity = target_velocity
            self._apply_mpc_control(best_control[0])
            print(f'[controllers.py] applied control (pos: {best_control[0][:3].round(4)}, dir: {best_control[0][3:6].round(4)}, dist: {best_control[0][6:].round(4)})')
            info['mpc_info'] = self.mpc_info
            info['mpc_control'] = best_control[0]
        return info

    def random_shooting_MPC(self, start_obs, target):
        # Initialize empty list to store the control sequence and corresponding cost
        obs_sequences = []
        controls_sequences = []
        costs = []
        info = dict()
        # repeat the observation for the number of samples (non-batched -> batched)
        batched_start_obs = {}
        for key, value in start_obs.items():
            batched_start_obs[key] = np.repeat(value[None, ...], self.config.num_samples, axis=0)
        obs_sequences.append(batched_start_obs)
        # Generate random control sequences
        for t in range(self.config.horizon_length):
            curr_obs = copy.deepcopy(obs_sequences[-1])
            controls = self.generate_random_control(curr_obs, target)
            # Simulate the system using the model and the generated control sequence
            pred_next_obs = self.forward_step(curr_obs, controls)
            # record the control sequence and the resulting observation
            obs_sequences.append(pred_next_obs)  # obs_sequences: [T, N, obs_dim]
            controls_sequences.append(controls)  # controls_sequences: [T, N, control_dim]
        # Calculate the cost of the generated control sequence
        costs = self.calculate_cost(obs_sequences, controls_sequences, target)  # [N]
        # Find the control sequence with the lowest cost
        best_traj_idx = np.argmin(costs)
        best_controls_sequence = np.array([control_per_step[best_traj_idx] for control_per_step in controls_sequences])  # [T, control_dim]
        # log info
        info['best_controls_sequence'] = best_controls_sequence
        info['best_cost'] = costs[best_traj_idx]
        info['costs'] = costs
        info['controls_sequences'] = controls_sequences
        info['obs_sequences'] = obs_sequences
        return best_controls_sequence, info

    def forward_step(self, obs, controls):
        """
        obs: dict including point cloud [B, N, obs_dim]
        controls: batched control sequences [B, control_dim]
        returns: resulting point cloud [B, N, obs_dim]
        """
        # forward dynamics
        pcs = obs['_point_cloud_world']  # [B, N, 3]
        contact_position = controls[:, :3]  # [B, 3]
        pushing_dir = controls[:, 3:6]  # [B, 3]
        pushing_dist = controls[:, 6:]  # [B, 1]
        inputs = (pcs, contact_position, pushing_dir, pushing_dist)
        next_pcs = self.dynamics_model.forward(inputs)  # [B, N, 3]
        # assemble next_obs
        next_obs = copy.deepcopy(obs)
        next_obs['_point_cloud_world'] = next_pcs
        return next_obs

    def generate_random_control(self, obs, target):
        pcs = obs['_point_cloud_world']  # [B, N, 3]
        num_samples, num_points, _ = pcs.shape
        # sample contact position randomly on point cloud
        points_idx = np.random.randint(0, num_points, num_samples)
        contact_positions = pcs[np.arange(num_samples), points_idx]  # [B, 3]
        # sample pushing_dir
        pushing_dirs = target - contact_positions  # [B, 3]
        pushing_dirs = normalize_vector(pushing_dirs)
        # sample pushing_dist
        pushing_dist = np.random.uniform(-0.02, 0.09, size=(num_samples, 1))  # [B, 1]
        # assemble control sequences
        controls = np.concatenate([contact_positions, pushing_dirs, pushing_dist], axis=1)  # [B, 7]
        return controls

    def calculate_cost(self, obs_sequences, controls_sequences, target_xyz):
        """
        Calculate the cost of the generated control sequence

        inputs:
        obs_sequences: batched observation sequences [T, B, N, 3]
        controls_sequences: batched control sequences [T, B, 7]

        returns: cost [B, 1]
        """
        num_samples, _, _ = obs_sequences[0]['_point_cloud_world'].shape
        last_obs = obs_sequences[-1]
        costs = []
        for i in range(num_samples):
            last_pc = last_obs['_point_cloud_world'][i]  # [N, 3]
            last_position = np.mean(last_pc, axis=0)  # [3]
            cost = np.linalg.norm(last_position - target_xyz)
            costs.append(cost)
        costs = np.array(costs)  # [B]
        return costs


class PushingDynamicsModel:
    """
    Heuristics-based pushing dynamics model.
    Translates the object by gripper_moving_distance in gripper_direction.
    """
    def __init__(self):
        pass

    def forward(self, inputs, max_per_batch=2000):
        """split inputs into multiple batches if exceeds max_per_batch"""
        num_batch = int(np.ceil(inputs[0].shape[0] / max_per_batch))
        output = []
        for i in range(num_batch):
            start = i * max_per_batch
            end = (i + 1) * max_per_batch
            output.append(self._forward_batched([x[start:end] for x in inputs]))
        output = np.concatenate(output, axis=0)
        return output

    def _forward_batched(self, inputs):
        (pcs, contact_position, gripper_direction, gripper_moving_distance) = inputs
        # to float16
        pcs = pcs.astype(np.float16)
        contact_position = contact_position.astype(np.float16)
        gripper_direction = gripper_direction.astype(np.float16)
        gripper_moving_distance = gripper_moving_distance.astype(np.float16)
        # find invalid push (i.e., outward push)
        obj_center = np.mean(pcs, axis=1)  # B x 3
        is_outward = np.sum((obj_center - contact_position) * gripper_direction, axis=1) < 0  # B
        moving_dist = gripper_moving_distance.copy()
        moving_dist[is_outward] = 0
        # translate pc by gripper_moving_distance in gripper_direction
        output = pcs + moving_dist[:, np.newaxis, :] * gripper_direction[:, np.newaxis, :]  # B x N x 3
        return output


# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

class LMP_interface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench'):
    self._env = env
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self._planner = PathPlanner(planner_config, map_size=self._map_size)
    self._controller = Controller(self._env, controller_config)

    # calculate size of each voxel (resolution)
    self._resolution = (self._env.workspace_bounds_max - self._env.workspace_bounds_min) / self._map_size
    print('#' * 50)
    print(f'## voxel resolution: {self._resolution}')
    print('#' * 50)
    print()
  
  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())
  
  def detect(self, obj_name):
    """return an observation dict containing useful information about the object"""
    if obj_name.lower() in EE_ALIAS:
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self._env.get_ee_pos()
    elif obj_name.lower() in TABLE_ALIAS:
      offset_percentage = 0.1
      x_min = self._env.workspace_bounds_min[0] + offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      x_max = self._env.workspace_bounds_max[0] - offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      y_min = self._env.workspace_bounds_min[1] + offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      y_max = self._env.workspace_bounds_max[1] - offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['normal'] = np.array([0, 0, 1])
      obs_dict['aabb'] = np.array([self._world_to_voxel(table_min_world), self._world_to_voxel(table_max_world)])
    else:
      obs_dict = dict()
      obj_pc, obj_normal = self._env.get_3d_obs_by_name(obj_name)
      voxel_map = self._points_to_voxel_map(obj_pc)
      aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
      aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
      obs_dict['occupancy_map'] = voxel_map  # in voxel frame
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
      obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
      obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
      obs_dict['_point_cloud_world'] = obj_pc  # in world frame
      obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))

    object_obs = Observation(obs_dict)
    return object_obs
  
  def execute(self, movable_obs_func, affordance_map=None, avoidance_map=None, rotation_map=None,
              velocity_map=None, gripper_map=None):
    # initialize default voxel maps if not specified
    if rotation_map is None:
      rotation_map = self._get_default_voxel_map('rotation')
    if velocity_map is None:
      velocity_map = self._get_default_voxel_map('velocity')
    if gripper_map is None:
      gripper_map = self._get_default_voxel_map('gripper')
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle')
    object_centric = (not movable_obs_func()['name'] in EE_ALIAS)
    execute_info = []
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(self._cfg['max_plan_iter']):
        step_info = dict()
        # evaluate voxel maps such that we use latest information
        movable_obs = movable_obs_func()
        _affordance_map = affordance_map()
        _avoidance_map = avoidance_map()
        _rotation_map = rotation_map()
        _velocity_map = velocity_map()
        _gripper_map = gripper_map()
        # preprocess avoidance map
        _avoidance_map = self._preprocess_avoidance_map(_avoidance_map, _affordance_map, movable_obs)
        # start planning
        start_pos = movable_obs['position']
        start_time = time.time()
        # optimize path and log
        path_voxel, planner_info = self._planner.optimize(start_pos, _affordance_map, _avoidance_map,
                                                        object_centric=object_centric)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'
        step_info['path_voxel'] = path_voxel
        step_info['planner_info'] = planner_info
        # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
        traj_world = self._path2traj(path_voxel, _rotation_map, _velocity_map, _gripper_map)
        traj_world = traj_world[:self._cfg['num_waypoints_per_plan']]
        step_info['start_pos'] = start_pos
        step_info['plan_iter'] = plan_iter
        step_info['movable_obs'] = movable_obs
        step_info['traj_world'] = traj_world
        step_info['affordance_map'] = _affordance_map
        step_info['rotation_map'] = _rotation_map
        step_info['velocity_map'] = _velocity_map
        step_info['gripper_map'] = _gripper_map
        step_info['avoidance_map'] = _avoidance_map

        # visualize
        if self._cfg['visualize']:
          assert self._env.visualizer is not None
          step_info['start_pos_world'] = self._voxel_to_world(start_pos)
          step_info['targets_world'] = self._voxel_to_world(planner_info['targets_voxel'])
          self._env.visualizer.visualize(step_info)

        # execute path
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] start executing path via controller ({len(traj_world)} waypoints){bcolors.ENDC}')
        controller_infos = dict()
        for i, waypoint in enumerate(traj_world):
          # check if the movement is finished
          if np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]) <= 0.01:
            print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_obs['_position_world']}, target={traj_world[-1][0]} (distance: {np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]):.3f})){bcolors.ENDC}")
            break
          # skip waypoint if moving to this point is going in opposite direction of the final target point
          # (for example, if you have over-pushed an object, no need to move back)
          if i != 0 and i != len(traj_world) - 1:
            movable2target = traj_world[-1][0] - movable_obs['_position_world']
            movable2waypoint = waypoint[0] - movable_obs['_position_world']
            if np.dot(movable2target, movable2waypoint).round(3) <= 0:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] skip waypoint {i+1} because it is moving in opposite direction of the final target{bcolors.ENDC}')
              continue
          # execute waypoint
          controller_info = self._controller.execute(movable_obs, waypoint)
          # loggging
          movable_obs = movable_obs_func()
          dist2target = np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0])
          if not object_centric and controller_info['mp_info'] == -1:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}')
          else:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
          controller_info['controller_step'] = i
          controller_info['target_waypoint'] = waypoint
          controller_infos[i] = controller_info
        step_info['controller_infos'] = controller_infos
        execute_info.append(step_info)
        # check whether we need to replan
        curr_pos = movable_obs['position']
        if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached target; terminating {bcolors.ENDC}')
          break
    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')

    # make sure we are at the final target position and satisfy any additional parametrization
    if not object_centric:
      try:
        # traj_world: world_xyz, rotation, velocity, gripper
        ee_pos_world = traj_world[-1][0]
        ee_rot_world = traj_world[-1][1]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = traj_world[-1][2]
        gripper_state = traj_world[-1][3]
      except:
        # evaluate latest voxel map
        _rotation_map = rotation_map()
        _velocity_map = velocity_map()
        _gripper_map = gripper_map()
        # get last ee pose
        ee_pos_world = self._env.get_ee_pos()
        ee_pos_voxel = self.get_ee_pos()
        ee_rot_world = _rotation_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = _velocity_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
        gripper_state = _gripper_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
      # move to the final target
      self._env.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))

    return execute_info
  
  def cm2index(self, cm, direction):
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)

  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
  
  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  def get_empty_affordance_map(self):
    return self._get_default_voxel_map('target')()  

  def get_empty_avoidance_map(self):
    return self._get_default_voxel_map('obstacle')()  
  
  def get_empty_rotation_map(self):
    return self._get_default_voxel_map('rotation')()  
  
  def get_empty_velocity_map(self):
    return self._get_default_voxel_map('velocity')() 
  
  def get_empty_gripper_map(self):
    return self._get_default_voxel_map('gripper')() 
  
  def reset_to_default_pose(self):
     self._env.reset_to_default_pose()
  
  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_scene_collision_voxel_map(self):
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel

  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'gripper':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 4))
        voxel_map[:, :, :] = self._env.get_ee_quat()
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  
  def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
    """
    convert path (generated by planner) to trajectory (used by controller)
    path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
    """
    # convert path to trajectory
    traj = []
    for i in range(len(path)):
      # get the current voxel position
      voxel_xyz = path[i]
      # get the current world position
      world_xyz = self._voxel_to_world(voxel_xyz)
      voxel_xyz = np.round(voxel_xyz).astype(int)
      # get the current rotation (in world frame)
      rotation = rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current velocity
      velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current on/off
      gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
      if (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
        # get indices of the less common values
        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
        less_common_indices = np.where(gripper_map == less_common_value)
        less_common_indices = np.array(less_common_indices).T
        # get closest distance from voxel_xyz to any of the indices that have less common value
        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
        # if the closest distance is less than threshold, then set gripper to less common value
        if closest_distance <= 3:
          gripper = less_common_value
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}')
      # add to trajectory
      traj.append((world_xyz, rotation, velocity, gripper))
    # append the last waypoint a few more times for the robot to stabilize
    for _ in range(2):
      traj.append((world_xyz, rotation, velocity, gripper))
    return traj
  
  def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    scene_collision_map = self._get_scene_collision_voxel_map()
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    try:
      ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
      scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    except KeyError:
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1)
    return avoidance_map

def setup_LMP(env, general_config, debug=False):
  controller_config = general_config['controller']
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']
  lmps_config = general_config['lmp_config']['lmps']
  env_name = general_config['env_name']
  # LMP env wrapper
  lmp_env = LMP_interface(env, lmp_env_config, controller_config, planner_config, env_name=env_name)
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'euler2quat': transforms3d.euler.euler2quat,
      'quat2euler': transforms3d.euler.quat2euler,
      'qinverse': transforms3d.quaternions.qinverse,
      'qmult': transforms3d.quaternions.qmult,
  }  # external library APIs
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }  # our custom APIs exposed to LMPs

  # allow LMPs to access other LMPs
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config']]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)

  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name
  )

  lmps = {
      'plan_ui': task_planner,
      'composer_ui': composer,
  }
  lmps.update(low_level_lmps)

  return lmps, lmp_env


def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map

class DiskCache:
    def __init__(self, cache_dir='cache', load_cache=True):
        self.cache_dir = cache_dir
        self.data = {}
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            if load_cache:
                self._load_cache()

    def _generate_filename(self, key):
        key_str = json.dumps(key)
        key_hash = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        return f"{key_hash}.pkl"

    def _load_cache(self):
        for filename in os.listdir(self.cache_dir):
            with open(os.path.join(self.cache_dir, filename), 'rb') as file:
                key, value = pickle.load(file)
                self.data[json.dumps(key)] = value

    def _save_to_disk(self, key, value):
        filename = self._generate_filename(key)
        with open(os.path.join(self.cache_dir, filename), 'wb') as file:
            pickle.dump((key, value), file)

    def __setitem__(self, key, value):
        str_key = json.dumps(key)
        self.data[str_key] = value
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        str_key = json.dumps(key)
        return self.data[str_key]
    
    def __contains__(self, key):
        str_key = json.dumps(key)
        return str_key in self.data

    def __repr__(self):
        return repr(self.data)

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4']]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query
            # handle given context (this was written originally for completion endpoint)
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                # remove obj_context from user1
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                # add obj_context to user2
                user2 = obj_context.strip() + '\n' + user2
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            kwargs['messages'] = messages
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                print(**kwargs)
                ret = openai.ChatCompletion.create(**kwargs)['choices'][0]['message']['content']
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                ret = openai.Completion.create(**kwargs)['choices'][0]['text'].strip()
                self._cache[kwargs] = ret
                return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)
        print(prompt)
        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]

def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    
def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    
"""Greedy path planner."""
class PathPlanner:
    """
    A greedy path planner that greedily chooses the next voxel with the lowest cost.
    Then apply several postprocessing steps to the path.
    """
    def __init__(self, planner_config, map_size):
        self.config = planner_config
        self.map_size = map_size

    def optimize(self, start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray, object_centric=False):
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] start')
        info = dict()
        # make copies
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map
        # smoothing
        target_map = distance_transform_edt(1 - target_map)
        target_map = normalize_map(target_map)
        obstacle_map = gaussian_filter(obstacle_map, sigma=self.config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        # combine target_map and obstacle_map
        costmap = target_map * self.config.target_map_weight + obstacle_map * self.config.obstacle_map_weight
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        # get stop criteria
        stop_criteria = self._get_stop_criteria()
        # initialize path
        path, current_pos = [start_pos], start_pos
        # optimize
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] start optimizing, start_pos: {start_pos}')
        for i in range(self.config.max_steps):
            # calculate all nearby voxels around current position
            all_nearby_voxels = self._calculate_nearby_voxel(current_pos, object_centric=object_centric)
            # calculate the score of all nearby voxels
            nearby_score = _costmap[all_nearby_voxels[:, 0], all_nearby_voxels[:, 1], all_nearby_voxels[:, 2]]
            # Find the minimum cost voxel
            steepest_idx = np.argmin(nearby_score)
            next_pos = all_nearby_voxels[steepest_idx]
            # increase cost at current position to avoid going back
            _costmap[current_pos[0].round().astype(int),
                     current_pos[1].round().astype(int),
                     current_pos[2].round().astype(int)] += 1
            # update path and current position
            path.append(next_pos)
            current_pos = next_pos
            # check stop criteria
            if stop_criteria(current_pos, _costmap, self.config.stop_threshold):
                break
        raw_path = np.array(path)
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] optimization finished; path length: {len(raw_path)}')
        # postprocess path
        processed_path = self._postprocess_path(raw_path, raw_target_map, object_centric=object_centric)
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] after postprocessing, path length: {len(processed_path)}')
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] last waypoint: {processed_path[-1]}')
        # save info
        info['start_pos'] = start_pos
        info['target_map'] = target_map
        info['obstacle_map'] = obstacle_map
        info['costmap'] = costmap
        info['costmap_altered'] = _costmap
        info['raw_start_pos'] = raw_start_pos
        info['raw_target_map'] = raw_target_map
        info['raw_obstacle_map'] = raw_obstacle_map
        info['planner_raw_path'] = raw_path.copy()
        info['planner_postprocessed_path'] = processed_path.copy()
        info['targets_voxel'] = np.argwhere(raw_target_map == 1)
        return processed_path, info
    
    def _get_stop_criteria(self):
        def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
            """
            Do not stop if there is a nearby voxel with cost less than current cost + stop_threshold.
            """
            assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
            current_pos_discrete = current_pos.round().clip(0, self.map_size - 1).astype(int)
            current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1], current_pos_discrete[2]]
            nearby_locs = self._calculate_nearby_voxel(current_pos, object_centric=False)
            nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1], nearby_locs[:, 2]] < current_cost + stop_threshold)
            if nearby_equal:
                return False
            return True
        return no_nearby_equal_criteria

    def _calculate_nearby_voxel(self, current_pos, object_centric=False):
        # create a grid of nearby voxels
        half_size = int(2 * self.map_size / 100)
        offsets = np.arange(-half_size, half_size + 1)
        # our heuristics-based dynamics model only supports planar pushing -> only xy path is considered
        if object_centric:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, [0])).T.reshape(-1, 3)
            # Remove the [0, 0, 0] offset, which corresponds to the current position
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        else:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
            # Remove the [0, 0, 0] offset, which corresponds to the current position
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        # Calculate all nearby voxel coordinates
        all_nearby_voxels = np.clip(current_pos + offsets_grid, 0, self.map_size - 1)
        # Remove duplicates, if any, caused by clipping
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0)
        return all_nearby_voxels
    
    def _postprocess_path(self, path, raw_target_map, object_centric=False):
        """
        Apply various postprocessing steps to the path.
        """
        # smooth the path
        savgol_window_size = min(len(path), self.config.savgol_window_size)
        savgol_polyorder = min(self.config.savgol_polyorder, savgol_window_size - 1)
        path = savgol_filter(path, savgol_window_size, savgol_polyorder, axis=0)
        # early cutoff if curvature is too high
        curvature = calc_curvature(path)
        if len(curvature) > 5:
            high_curvature_idx = np.where(curvature[5:] > self.config.max_curvature)[0]
            if len(high_curvature_idx) > 0:
                high_curvature_idx += 5
                path = path[:int(0.9 * high_curvature_idx[0])]  
        # skip waypoints such that they reach target spacing
        path_trimmed = path[1:-1]
        skip_ratio = None
        if len(path_trimmed) > 1:
            target_spacing = int(self.config['target_spacing'] * self.map_size / 100)
            length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
            if length > target_spacing:
                curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
                skip_ratio = np.round(target_spacing / curr_spacing).astype(int)
                if skip_ratio > 1:
                    path_trimmed = path_trimmed[::skip_ratio]
        path = np.concatenate([path[0:1], path_trimmed, path[-1:]])
        # force last position to be one of the target positions
        last_waypoint = path[-1].round().clip(0, self.map_size - 1).astype(int)
        if raw_target_map[last_waypoint[0], last_waypoint[1], last_waypoint[2]] == 0:
            # find the closest target position
            target_pos = np.argwhere(raw_target_map == 1)
            closest_target_idx = np.argmin(np.linalg.norm(target_pos - last_waypoint, axis=1))
            closest_target = target_pos[closest_target_idx]
            # for object centric motion, we assume we can only push in the xy plane
            if object_centric:
                closest_target[2] = last_waypoint[2]
            path = np.append(path, [closest_target], axis=0)
        # space out path more if task is object centric (so that we can push faster)
        if object_centric:
            k = self.config['pushing_skip_per_k']
            path = np.concatenate([path[k:-1:k], path[-1:]])
        path = path.clip(0, self.map_size-1)
        return path


def set_lmp_objects(lmps, objects):
    if isinstance(lmps, dict):
        lmps = lmps.values()
    for lmp in lmps:
        lmp._context = f'objects = {objects}'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_prompt(prompt_fname):
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # get full path to file
    if '/' in prompt_fname:
        prompt_fname = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *prompt_fname)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    # read file
    with open(full_path, 'r') as f:
        contents = f.read().strip()
    return contents

def normalize_vector(x, eps=1e-6):
    """normalize a vector to unit length"""
    x = np.asarray(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    elif x.ndim == 2:
        norm = np.linalg.norm(x, axis=1)  # (N,)
        normalized = np.zeros_like(x)
        normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
        return normalized

def normalize_map(map):
    """normalization voxel maps to [0, 1] without producing nan"""
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def calc_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    dz = np.gradient(path[:, 2])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    curvature = np.sqrt((ddy * dx - ddx * dy)**2 + (ddz * dx - ddx * dz)**2 + (ddz * dy - ddy * dz)**2) / np.power(dx**2 + dy**2 + dz**2, 3/2)
    # convert any nan to 0
    curvature[np.isnan(curvature)] = 0
    return curvature

class IterableDynamicObservation:
    """acts like a list of DynamicObservation objects, initialized with a function that evaluates to a list"""
    def __init__(self, func):
        assert callable(func), 'func must be callable'
        self.func = func
        self._validate_func_output()

    def _validate_func_output(self):
        evaluated = self.func()
        assert isinstance(evaluated, list), 'func must evaluate to a list'

    def __getitem__(self, index):
        def helper():
            evaluated = self.func()
            item = evaluated[index]
            # assert isinstance(item, Observation), f'got type {type(item)} instead of Observation'
            return item
        return helper

    def __len__(self):
        return len(self.func())

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __call__(self):
        static_list = self.func()
        return static_list

class DynamicObservation:
    """acts like dict observation but initialized with a function such that it uses the latest info"""
    def __init__(self, func):
        try:
            assert callable(func) and not isinstance(func, dict), 'func must be callable or cannot be a dict'
        except AssertionError as e:
            print(e)
            import pdb; pdb.set_trace()
        self.func = func
    
    def __get__(self, key):
        evaluated = self.func()
        if isinstance(evaluated[key], np.ndarray):
            return evaluated[key].copy()
        return evaluated[key]
    
    def __getattr__(self, key):
        return self.__get__(key)
    
    def __getitem__(self, key):
        return self.__get__(key)

    def __call__(self):
        static_obs = self.func()
        if not isinstance(static_obs, Observation):
            static_obs = Observation(static_obs)
        return static_obs

class Observation(dict):
    def __init__(self, obs_dict):
        super().__init__(obs_dict)
        self.obs_dict = obs_dict
    
    def __getattr__(self, key):
        return self.obs_dict[key]
    
    def __getitem__(self, key):
        return self.obs_dict[key]

    def __getstate__(self):
        return self.obs_dict
    
    def __setstate__(self, state):
        self.obs_dict = state

def pointat2quat(pointat):
    """
    calculate quaternion from pointat vector
    """
    up = np.array(pointat, dtype=np.float32)
    up = normalize_vector(up)
    rand_vec = np.array([1, 0, 0], dtype=np.float32)
    rand_vec = normalize_vector(rand_vec)
    # make sure that the random vector is close to desired direction
    if np.abs(np.dot(rand_vec, up)) > 0.99:
        rand_vec = np.array([0, 1, 0], dtype=np.float32)
        rand_vec = normalize_vector(rand_vec)
    left = np.cross(up, rand_vec)
    left = normalize_vector(left)
    forward = np.cross(left, up)
    forward = normalize_vector(forward)
    rotmat = np.eye(3).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    quat_wxyz = mat2quat(rotmat)
    return quat_wxyz

def visualize_points(point_cloud, point_colors=None, show=True):
    """visualize point clouds using plotly"""
    if point_colors is None:
        point_colors = point_cloud[:, 2]
    fig = go.Figure(data=[go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
                                    mode='markers', marker=dict(size=3, color=point_colors, opacity=1.0))])
    if show:
        fig.show()
    else:
        # save to html
        fig.write_html('temp_pc.html')
        print(f'Point cloud saved to temp_pc.html')

def _process_llm_index(indices, array_shape):
    """
    processing function for returned voxel maps (which are to be manipulated by LLMs)
    handles non-integer indexing
    handles negative indexing with manually designed special cases
    """
    if isinstance(indices, int) or isinstance(indices, np.int64) or isinstance(indices, np.int32) or isinstance(indices, np.int16) or isinstance(indices, np.int8):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, float) or isinstance(indices, np.float64) or isinstance(indices, np.float32) or isinstance(indices, np.float16):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = np.round(start).astype(int)
        if stop is not None:
            stop = np.round(stop).astype(int)
        if step is not None:
            step = np.round(step).astype(int)
        # only convert the case where the start is negative and the stop is positive/negative
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, tuple) or isinstance(indices, list):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        print("[IndexingWrapper] Warning: numpy array indexing was converted to list")
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        print(f"[IndexingWrapper] {indices} (type: {type(indices)}) not supported")
        raise TypeError("Indexing type not supported")
    # give warning if index was negative
    if processed != indices:
        print(f"[IndexingWrapper] Warning: index was changed from {indices} to {processed}")
    # print(f"[IndexingWrapper] {idx} -> {processed}")
    return processed

"""Plotly-Based Visualizer"""
class ValueMapVisualizer:
    """
    A Plotly-based visualizer for 3D value map and planned path.
    """
    def __init__(self, config):
        self.scene_points = None
        self.save_dir = config['save_dir']
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.quality = config['quality']
        self.update_quality(self.quality)
        self.map_size = config['map_size']
    
    def update_bounds(self, lower, upper):
        self.workspace_bounds_min = lower
        self.workspace_bounds_max = upper
        self.plot_bounds_min = lower - 0.15 * (upper - lower)
        self.plot_bounds_max = upper + 0.15 * (upper - lower)
        xyz_ratio = 1 / (self.workspace_bounds_max - self.workspace_bounds_min)
        scene_scale = np.max(xyz_ratio) / xyz_ratio
        self.scene_scale = scene_scale

    def update_quality(self, quality):
        self.quality = quality
        if self.quality == 'low':
            self.downsample_ratio = 4
            self.max_scene_points = 150000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == 'medium':
            self.downsample_ratio = 2
            self.max_scene_points = 300000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == 'high':
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.07 * 0.6
            self.costmap_surface_count = 50
        else:
            raise ValueError(f'Unknown quality: {self.quality}; should be one of [low, medium, high]')

    def update_scene_points(self, points, colors=None):
        points = points.astype(np.float16)
        assert colors.dtype == np.uint8
        self.scene_points = (points, colors)

    def visualize(self, info, show=False, save=True):
        """visualize the path and relevant info using plotly"""
        planner_info = info['planner_info']
        waypoints_world = np.array([p[0] for p in info['traj_world']])
        start_pos_world = info['start_pos_world']
        assert len(start_pos_world.shape) == 1
        waypoints_world = np.concatenate([start_pos_world[None, ...], waypoints_world], axis=0)
        
        fig_data = []
        # plot path
        # add marker to path waypoints
        fig_data.append(go.Scatter3d(x=waypoints_world[:, 0], y=waypoints_world[:, 1], z=waypoints_world[:, 2], mode='markers', name='waypoints', marker=dict(size=4, color='red')))
        # add lines between waypoints
        for i in range(waypoints_world.shape[0] - 1):
            fig_data.append(go.Scatter3d(x=waypoints_world[i:i+2, 0], y=waypoints_world[i:i+2, 1], z=waypoints_world[i:i+2, 2], mode='lines', name='path', line=dict(width=10, color='orange')))
        if planner_info is not None:
            # plot costmap
            if 'costmap' in planner_info:
                costmap = planner_info['costmap'][::self.downsample_ratio, ::self.downsample_ratio, ::self.downsample_ratio]
                skip_ratio = (self.workspace_bounds_max - self.workspace_bounds_min) / (self.map_size / self.downsample_ratio)
                x, y, z = np.mgrid[self.workspace_bounds_min[0]:self.workspace_bounds_max[0]:skip_ratio[0],
                                self.workspace_bounds_min[1]:self.workspace_bounds_max[1]:skip_ratio[1],
                                self.workspace_bounds_min[2]:self.workspace_bounds_max[2]:skip_ratio[2]]
                grid_shape = costmap.shape
                x = x[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
                y = y[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
                z = z[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
                fig_data.append(go.Volume(x=x.flatten(), y=y.flatten(), z=z.flatten(), value=costmap.flatten(), isomin=0, isomax=1, opacity=self.costmap_opacity, surface_count=self.costmap_surface_count, colorscale='Jet', showlegend=True, showscale=False))
            # plot start position
            if 'start_pos' in planner_info:
                fig_data.append(go.Scatter3d(x=[start_pos_world[0]], y=[start_pos_world[1]], z=[start_pos_world[2]], mode='markers', name='start', marker=dict(size=6, color='blue')))
            # plot target as dots extracted from target_map
            if 'raw_target_map' in planner_info:
                targets_world = info['targets_world']
                fig_data.append(go.Scatter3d(x=targets_world[:, 0], y=targets_world[:, 1], z=targets_world[:, 2], mode='markers', name='target', marker=dict(size=6, color='green', opacity=0.7)))

        # visualize scene points
        if self.scene_points is None:
            print('no scene points to overlay, skipping...')
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(scene_points.shape[0], min(scene_points.shape[0], self.max_scene_points), replace=False)
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points
            fig_data.append(go.Scatter3d(x=scene_points[:, 0], y=scene_points[:, 1], z=scene_points[:, 2],
                                        mode='markers', marker=dict(size=3, color=scene_point_colors, opacity=1.0)))
        
        fig = go.Figure(data=fig_data)
 
        # set bounds and ratio
        fig.update_layout(scene=dict(xaxis=dict(range=[self.plot_bounds_min[0], self.plot_bounds_max[0]], autorange=False),
                                    yaxis=dict(range=[self.plot_bounds_min[1], self.plot_bounds_max[1]], autorange=False),
                                    zaxis=dict(range=[self.plot_bounds_min[2], self.plot_bounds_max[2]], autorange=False)),
                        scene_aspectmode='manual',
                        scene_aspectratio=dict(x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]))

        # do not show grid and axes
        fig.update_layout(scene=dict(xaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
                                    yaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
                                    zaxis=dict(showgrid=False, showticklabels=False, title='', visible=False)))

        # set background color as white
        fig.update_layout(template='none')

        # save and show
        if save and self.save_dir is not None:
            curr_time = datetime.datetime.now()
            log_id = f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'
            save_path = os.path.join(self.save_dir, log_id + '.html')
            latest_save_path = os.path.join(self.save_dir, 'latest.html')
            print('** saving visualization to', save_path, '...')
            fig.write_html(save_path)
            print('** saving visualization to', latest_save_path, '...')
            fig.write_html(latest_save_path)
            print(f'** save to {save_path}')
        if show:
            fig.show()
        return fig


class CustomMoveArmThenGripper(MoveArmThenGripper):
    """
    A potential workaround for the default MoveArmThenGripper as we frequently run into zero division errors and failed path.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_arm_action = None

    def action(self, scene, action):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        # if the arm action is the same as the previous action, skip it
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action)
        else:
            try:
                self.arm_action_mode.action(scene, arm_action)
            except Exception as e:
                print(f'{bcolors.FAIL}[rlbench_env.py] Ignoring failed arm action; Exception: "{str(e)}"{bcolors.ENDC}')
            self.gripper_action_mode.action(scene, ee_action)
        self._prev_arm_action = arm_action.copy()

class VoxPoserRLBench():
    def __init__(self, visualizer=None):
        action_mode = CustomMoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(),
                                        gripper_action_mode=Discrete())
        self.rlbench_env = Environment(action_mode)
        self.rlbench_env.launch()
        self.task = None

        self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self.camera_names = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        # calculate lookat vector for all cameras (for normal estimation)
        name2cam = {
            'front': self.rlbench_env._scene._cam_front,
            'left_shoulder': self.rlbench_env._scene._cam_over_shoulder_left,
            'right_shoulder': self.rlbench_env._scene._cam_over_shoulder_right,
            'overhead': self.rlbench_env._scene._cam_overhead,
            'wrist': self.rlbench_env._scene._cam_wrist,
        }
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_name in self.camera_names:
            extrinsics = name2cam[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        # load file containing object names for each task
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        with open(path, 'r') as f:
            self.task_object_names = json.load(f)

        self._reset_task_variables()

    def get_object_names(self):
        name_mapping = self.task_object_names[self.task.get_name()]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names

    def load_task(self, task):
        self._reset_task_variables()
        if isinstance(task, str):
            task = getattr(tasks, task)
        self.task = self.rlbench_env.get_task(task)
        self.arm_mask_ids = [obj.get_handle() for obj in self.task._robot.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_mask_ids = [obj.get_handle() for obj in self.task._robot.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_mask_ids = self.arm_mask_ids + self.gripper_mask_ids
        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json"')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if ignore_robot:
            robot_mask = np.isin(masks, self.robot_mask_ids)
            points = points[~robot_mask]
            colors = colors[~robot_mask]
            masks = masks[~robot_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        assert self.task is not None, "Please load a task first"
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        return descriptions, obs

    def apply_action(self, action):
        assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.task.step(action)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        grasped_objects = self.rlbench_env._scene.robot.gripper.get_grasped_objects()
        if len(grasped_objects) > 0:
            self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action
    

config = get_config('rlbench')

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)
lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']

# below are the tasks that have object names added to the "task_object_names.json" file
env.load_task(tasks.PutRubbishInBin)

descriptions, obs = env.reset()
set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer

instruction = np.random.choice(descriptions)
voxposer_ui(instruction)
