"""Registers the multi-clip tracking task."""

from mjlab.tasks.registry import register_mjlab_task, load_rl_cfg

from .env_cfg import MultiClipSettings, make_multiclip_cfg

BASE_TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

MOTION_FILE = "/opt/nb/johan/data/motion_file/phuma_track_v2.npz"
SPLIT_FILE = "/opt/nb/johan/data/motion_file/phuma_track_v2.split.json"


def _settings(**kw) -> MultiClipSettings:
    base = dict(motion_file=MOTION_FILE, split_file=SPLIT_FILE)
    base.update(kw)
    return MultiClipSettings(**base)


register_mjlab_task(
    task_id="G1-MultiClip",
    env_cfg=make_multiclip_cfg(_settings(split="train")),
    play_env_cfg=make_multiclip_cfg(_settings(split="test", eval=True), play=True),
    rl_cfg=load_rl_cfg(BASE_TASK),
)