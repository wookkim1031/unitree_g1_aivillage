import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import mjlab.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
import utils
import config_loader
from pathlib import Path

"""
No-State-Estimation drops "motion_anchor_pos_b" and "base_lin_vel" from the actor group 

G1's LowState_ gives the joint encoders and an IMU. IMU provides orientation and angular velocity, so base_ang_vel survives

base_lin_vel: root linear velocity in world frame. Not measurable without external tracking or a learned state estimator. 
motion_anchor_pos_b: the reference anchor's position relative to the robot. measuring this requires knowing the robot's own global root position

Everything else 160-dim actor obs
- command (58, reference joint targets you carry with the policy) 
- motion_achor_ori_b(6, orientation only - IMU) 
- base_ang_vel (3, IMU)
- joint_pos / joint_vel (29 each, encoders)
- actions (29, last output)
"""

TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
config = utils.PlayConfig(TASK)

config = config_loader.load_and_overwrite_play_config(
    config, Path(__file__).resolve().parent / "configs/play_config.yaml")

print(config)
utils.run_play(task_id=TASK, cfg=config)