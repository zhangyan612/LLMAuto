from planners import PathPlanner
import numpy as np
from types import SimpleNamespace

map_size = 100
start_pos = np.array([52, 49, 71])

_affordance_map = np.load("affordance_map.npy")

_avoidance_map = np.load("avoidance_map.npy")
planner_config = {
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

config = SimpleNamespace(**planner_config)

print(start_pos)
_planner = PathPlanner(config, map_size=map_size)

path_voxel, planner_info = _planner.optimize(start_pos, _affordance_map, _avoidance_map, object_centric=False)

print(path_voxel)
print(planner_info)
