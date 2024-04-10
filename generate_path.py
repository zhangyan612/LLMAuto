
# Given a point cloud and gripper location, object target location
# Get robot path trajectory to reach target location

import datetime
import time
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import savgol_filter
from types import SimpleNamespace
from ValueMapVisualizer import ValueMapVisualizer
# from VoxelIndexingWrapper import VoxelIndexingWrapper
# from rlbench_env import VoxPoserRLBench

EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

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


_cfg = {'map_size': 100, 'num_waypoints_per_plan': 10000, 'max_plan_iter': 1, 'visualize': True}
_map_size = _cfg['map_size']

config = {
    "stop_threshold": 0.001,
    "savgol_polyorder": 3,
    "savgol_window_size": 20,
    "obstacle_map_weight": 1,
    "max_steps": 300,
    "obstacle_map_gaussian_sigma": 10,
    "target_map_weight": 2,
    "stop_criteria": "no_nearby_equal",
    "target_spacing": 1,
    "max_curvature": 3,
    "pushing_skip_per_k": 5
}
config = SimpleNamespace(**config)

visualizer_config = {
  "save_dir": "./visualizations",
  "quality": "low",
  "map_size": 100
}

latest_action = None

visualizer = ValueMapVisualizer(visualizer_config)
# env = VoxPoserRLBench(visualizer=visualizer)


def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'
    
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

# def get_ee_pose():
#     assert latest_obs is not None, "Please reset the environment first"
#     return latest_obs.gripper_pose

# def get_ee_pos():
#     return get_ee_pose()[:3]

# def get_ee_quat():
#     return get_ee_pose()[3:]

# def get_last_gripper_action():
#     """
#     Returns the last gripper action.

#     Returns:
#         float: The last gripper action.
#     """
#     if latest_action is not None:
#         return latest_action[-1]
#     else:
#         return init_obs.gripper_open


# def _get_default_voxel_map(type='target'):
#     """returns default voxel map (defaults to current state)"""
#     def fn_wrapper():
#       if type == 'target':
#         voxel_map = np.zeros((_cfg['map_size'], _cfg['map_size'], _cfg['map_size']))
#       elif type == 'obstacle':  # for LLM to do customization
#         voxel_map = np.zeros((_cfg['map_size'], _cfg['map_size'], _cfg['map_size']))
#       elif type == 'velocity':
#         voxel_map = np.ones((_cfg['map_size'], _cfg['map_size'], _cfg['map_size']))
#       elif type == 'gripper':
#         voxel_map = np.ones((_cfg['map_size'], _cfg['map_size'], _cfg['map_size'])) * env.get_last_gripper_action()
#       elif type == 'rotation':
#         voxel_map = np.zeros((_cfg['map_size'], _cfg['map_size'], _cfg['map_size'], 4))
#         voxel_map[:, :, :] = env.get_ee_quat()
#       else:
#         raise ValueError('Unknown voxel map type: {}'.format(type))
#       voxel_map = VoxelIndexingWrapper(voxel_map)
#       return voxel_map
#     return fn_wrapper

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def _voxel_to_world(voxel_xyz):
    _voxels_bounds_robot_min = [-0.27499999, -0.65500004,  0.75199986] #env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = [0.77499999, 0.65500004, 1.75199986] #env.workspace_bounds_max.astype(np.float32)
    _map_size = _map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

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

# def _points_to_voxel_map(points):
#     """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
#     _points = points.astype(np.float32)
#     _voxels_bounds_robot_min = env.workspace_bounds_min.astype(np.float32)
#     _voxels_bounds_robot_max = env.workspace_bounds_max.astype(np.float32)
#     _map_size = _map_size
#     return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

# def _get_scene_collision_voxel_map(self):
#     collision_points_world, _ = env.get_scene_3d_obs(ignore_robot=True)
#     collision_voxel = _points_to_voxel_map(collision_points_world)
#     return collision_voxel

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

# def _preprocess_avoidance_map(avoidance_map, affordance_map, movable_obs):
#     # collision avoidance
#     scene_collision_map = _get_scene_collision_voxel_map()
#     # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
#     ignore_mask = distance_transform_edt(1 - affordance_map)
#     scene_collision_map[ignore_mask < int(0.15 * _map_size)] = 0
#     # anywhere within 15/100 indices of the start is ignored
#     try:
#       ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
#       scene_collision_map[ignore_mask < int(0.15 * _map_size)] = 0
#     except KeyError:
#       start_pos = movable_obs['position']
#       ignore_mask = np.ones_like(avoidance_map)
#       ignore_mask[start_pos[0] - int(0.1 * _map_size):start_pos[0] + int(0.1 * _map_size),
#                   start_pos[1] - int(0.1 * _map_size):start_pos[1] + int(0.1 * _map_size),
#                   start_pos[2] - int(0.1 * _map_size):start_pos[2] + int(0.1 * _map_size)] = 0
#       scene_collision_map *= ignore_mask
#     avoidance_map += scene_collision_map
#     avoidance_map = np.clip(avoidance_map, 0, 1)
#     return avoidance_map

def normalize_map(map):
    """normalization voxel maps to [0, 1] without producing nan"""
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def _calculate_nearby_voxel(current_pos, object_centric=False):
        # create a grid of nearby voxels
        half_size = int(2 * _map_size / 100)
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
        all_nearby_voxels = np.clip(current_pos + offsets_grid, 0, _map_size - 1)
        # Remove duplicates, if any, caused by clipping
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0)
        return all_nearby_voxels

def _get_stop_criteria():
    def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
        """
        Do not stop if there is a nearby voxel with cost less than current cost + stop_threshold.
        """
        assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
        current_pos_discrete = current_pos.round().clip(0, _map_size - 1).astype(int)
        current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1], current_pos_discrete[2]]
        nearby_locs = _calculate_nearby_voxel(current_pos, object_centric=False)
        nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1], nearby_locs[:, 2]] < current_cost + stop_threshold)
        if nearby_equal:
            return False
        return True
    return no_nearby_equal_criteria


def _postprocess_path(path, raw_target_map, object_centric=False):
        """
        Apply various postprocessing steps to the path.
        """
        # smooth the path
        savgol_window_size = min(len(path), config.savgol_window_size)
        savgol_polyorder = min(config.savgol_polyorder, savgol_window_size - 1)
        path = savgol_filter(path, savgol_window_size, savgol_polyorder, axis=0)
        # early cutoff if curvature is too high
        curvature = calc_curvature(path)
        if len(curvature) > 5:
            high_curvature_idx = np.where(curvature[5:] > config.max_curvature)[0]
            if len(high_curvature_idx) > 0:
                high_curvature_idx += 5
                path = path[:int(0.9 * high_curvature_idx[0])]  
        # skip waypoints such that they reach target spacing
        path_trimmed = path[1:-1]
        skip_ratio = None
        if len(path_trimmed) > 1:
            target_spacing = int(config['target_spacing'] * _map_size / 100)
            length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
            if length > target_spacing:
                curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
                skip_ratio = np.round(target_spacing / curr_spacing).astype(int)
                if skip_ratio > 1:
                    path_trimmed = path_trimmed[::skip_ratio]
        path = np.concatenate([path[0:1], path_trimmed, path[-1:]])
        # force last position to be one of the target positions
        last_waypoint = path[-1].round().clip(0, _map_size - 1).astype(int)
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
            k = config['pushing_skip_per_k']
            path = np.concatenate([path[k:-1:k], path[-1:]])
        path = path.clip(0, _map_size-1)
        return path

def optimize(start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray, object_centric=False):
        """
        config:
            start_pos: (3,) np.ndarray, start position
            target_map: (map_size, map_size, map_size) np.ndarray, target_map
            obstacle_map: (map_size, map_size, map_size) np.ndarray, obstacle_map
            object_centric: bool, whether the task is object centric (entity of interest is an object/part instead of robot)
        Returns:
            path: (n, 3) np.ndarray, path
            info: dict, info
        """

        print(f'[planners.py | {get_clock_time(milliseconds=True)}] start')
        info = dict()
        # make copies
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map
        # smoothing
        target_map = distance_transform_edt(1 - target_map)
        target_map = normalize_map(target_map)
        obstacle_map = gaussian_filter(obstacle_map, sigma=config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        # combine target_map and obstacle_map
        costmap = target_map * config.target_map_weight + obstacle_map * config.obstacle_map_weight
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        # get stop criteria
        stop_criteria = _get_stop_criteria()
        # initialize path
        path, current_pos = [start_pos], start_pos
        # optimize
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] start optimizing, start_pos: {start_pos}')
        for i in range(config.max_steps):
            # calculate all nearby voxels around current position
            all_nearby_voxels = _calculate_nearby_voxel(current_pos, object_centric=object_centric)
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
            if stop_criteria(current_pos, _costmap, config.stop_threshold):
                break
        raw_path = np.array(path)
        print(f'[planners.py | {get_clock_time(milliseconds=True)}] optimization finished; path length: {len(raw_path)}')
        # postprocess path
        processed_path = _postprocess_path(raw_path, raw_target_map, object_centric=object_centric)
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


def execute(affordance_map, avoidance_map, rotation_map, velocity_map, gripper_map):
    movable_obs = {'name': 'gripper', 'position': [52.0, 49.0, 71.0], 'aabb': [[52.0, 49.0, 71.0],[52.0, 49.0, 71.0]], '_position_world': [ 0.27849087, -0.00815093,  1.47194481]}

    # initialize default voxel maps if not specified
    object_centric = (not movable_obs['name'] in EE_ALIAS)
    execute_info = []
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(_cfg['max_plan_iter']):
        step_info = dict()

        # start planning
        start_pos = movable_obs['position']
        start_time = time.time()
        # optimize path and log
        path_voxel, planner_info = optimize(start_pos, _affordance_map, _avoidance_map,
                                                        object_centric=object_centric)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'
        step_info['path_voxel'] = path_voxel
        step_info['planner_info'] = planner_info
        # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
        traj_world = _path2traj(path_voxel, _rotation_map, _velocity_map, _gripper_map)
        traj_world = traj_world[:_cfg['num_waypoints_per_plan']]
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
        if _cfg['visualize']:
          assert visualizer is not None
          step_info['start_pos_world'] = _voxel_to_world(start_pos)
          step_info['targets_world'] = _voxel_to_world(planner_info['targets_voxel'])
          visualizer.visualize(step_info)



if __name__ == "__main__":
    _affordance_map = np.load("affordance_map.npy")
    _avoidance_map = np.load("avoidance_map.npy")
    _rotation_map  = np.load("rotation_map.npy")
    _velocity_map = np.load("velocity_map.npy")
    _gripper_map = np.load("gripper_map.npy")
    execute(_affordance_map, _avoidance_map, _rotation_map, _velocity_map, _gripper_map)