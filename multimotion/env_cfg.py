"""Multi-clip tracking env config."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.tasks.registry import load_env_cfg

from .mdp import MultiMotionCommandCfg, clip_timeout

BASE_TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"


@dataclass
class MultiClipSettings:
  """Lands in the env.yaml dumped beside every checkpoint."""
  motion_file: str = ""
  split_file: str = ""
  split: str = "train"
  episode_length_s: float = 10.0
  njmax: int = 512
  w_root_pos: float = 1.0
  w_action_rate: float = -0.1
  ee_body_names: tuple[str, ...] = (
    "left_ankle_roll_link", "right_ankle_roll_link")
  eval: bool = False
  eval_disturbed: bool = False


def make_multiclip_cfg(s: MultiClipSettings = MultiClipSettings(),
                       play: bool = False):
  cfg = load_env_cfg(BASE_TASK, play=play)

  cfg.episode_length_s = s.episode_length_s
  cfg.sim.njmax = s.njmax

  # Clip end terminates the episode; the command holds at the last frame.
  TermCfg = type(cfg.terminations["time_out"])
  cfg.terminations["clip_timeout"] = TermCfg(func=clip_timeout, time_out=True)

  # Wrists stay out of the end-effector termination: the retargeting emits no
  # wrist motion, so reference wrist poses sit far off on many resets.
  if "ee_body_pos" in cfg.terminations:
    cfg.terminations["ee_body_pos"].params["body_names"] = s.ee_body_names

  if s.eval:
    # A fall must not truncate the measurement; only clip_timeout ends it.
    for t in ("anchor_pos", "anchor_ori", "ee_body_pos"):
      cfg.terminations.pop(t, None)
    cfg.episode_length_s = 1e9
    if not s.eval_disturbed:
      for ev in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
        cfg.events.pop(ev, None)

  base = cfg.commands["motion"]
  cfg.commands["motion"] = MultiMotionCommandCfg(
    **{**vars(base), "sampling_mode": "uniform"},
    split_file=s.split_file or None,
    split=s.split,
    min_remaining_frames=1 if s.eval else 100,
    random_start_phase=not s.eval,
    deterministic_clips=s.eval,
  )
  cfg.commands["motion"].motion_file = s.motion_file
  cfg.commands["motion"].resampling_time_range = (1e9, 1e9)

  cfg.rewards["motion_global_root_pos"].weight = s.w_root_pos
  cfg.rewards["action_rate_l2"].weight = s.w_action_rate

  cfg.episode_length_s = 40.0

  return cfg