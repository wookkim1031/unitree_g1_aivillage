from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

import config_loader

from mjlab.utils.torch import configure_torch_backends

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper

@dataclass
class EvalConfig:
    task_id: str = "G1-MultiClip"
    checkpoint_file: str = ""
    motion_file: str = ""
    split_file: str = ""
    split: str = "test"

    out_csv: str = "eval_results.csv"
    device:str = "cuda:0"

    min_clip_frames:int = 150
    max_steps: int | None = None

    succ_threshold: float = 0.5

    # Leave failure terminations on
    keep_terminations: bool = True

    # Zero the Eventmanager. Disabling terminations alone does not stop push_robot 
    # or the startup doamin randomisation
    disable_events: bool = True

    num_envs: int | None = None

    metric_keys: list[str] = field(
        default_factory=lambda: [
            "error_joint_pos",
            "error_body_pos",
            "error_anchor_pos",
            "error_anchor_rot",
            "error_joint_vel",
        ]
    )

class ClipMetrics:
    # Per env accumulators, masked at each env's first termination
    def __init__(self, num_envs: int, keys: list[str], device: str):
        z = lambda: torch.zeros(num_envs, device=device)
        self.keys = keys
        self.sums = {k: z() for k in keys}
        self.counts = z()
        self.peak_3d = z()
        self.peak_z = z()
        self.steps = z()
        self.alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.completed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, metrics:dict, err_3d, err_z):
        m = self.alive.float()
        for k in self.keys:
            if k in metrics:
                self.sums[k] += metrics[k].detach() * m
        self.counts += m
        self.steps += m
        self.peak_3d = torch.maximum(self.peak_3d, err_3d.detach() * m)
        self.peak_z = torch.maximum(self.peak_z, err_z.detach() * m)

    def retire(self, terminated, truncated):
        """truncated (clip_timeout / time_out) means the clip finished."""
        self.completed |= truncated.bool() & self.alive
        self.alive &= ~(terminated.bool() | truncated.bool())

    def finalize(self, threshold: float) -> dict:
        n = self.counts.clamp(min=1)
        out = {k: (self.sums[k] / n).cpu() for k in self.keys}
        out["peak3d"] = self.peak_3d.cpu()
        out["peak_z"] = self.peak_z.cpu()
        out["succ_3d"] = (self.peak_3d < threshold).float().cpu()
        out["succ_z"] = (self.peak_z < threshold).float().cpu()
        out["completed"] = self.completed.float().cpu()
        out["steps"] = self.steps.cpu()
        return out
        
def load_config() -> EvalConfig:
    config = EvalConfig()
    config = config_loader.load_and_overwrite_config(
        config, Path(__file__).resolve().parent / "configs/eval_config.yaml"
    )
    return config

def build_eval_env(cfg: EvalConfig, num_envs:int, device:str):
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    agent_cfg = load_rl_cfg(cfg.task_id)
    motion_cmd = env_cfg.commands["motion"]

    motion_cmd.motion_file = cfg.motion_file
    motion_cmd.split_file = cfg.split_file
    motion_cmd.split = cfg.split
    motion_cmd.min_clip_frames = cfg.min_clip_frames
    motion_cmd.deterministic_clips = True
    motion_cmd.random_start_phase = False
    motion_cmd.sampling_mode = "uniform"
    motion_cmd.debug_vis = False

    if cfg.disable_events:
        env_cfg.events = {}
        print("[eval] events disabled")
        
    # No observation noise
    env_cfg.observations["actor"].enable_corruption = False
    
    if not cfg.keep_terminations:
        for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
            env_cfg.terminations.pop(name, None)
        print("[eval] failure terminations removed")

    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # noqa: F821
    return env, agent_cfg

    

def load_policy(cfg: EvalConfig, env, agent_cfg, device):
    sd = torch.load(cfg.checkpoint_file, map_location="cpu")["actor_state_dict"]
    dims = [sd[k].shape[0] for k in sorted(sd) if k.endswith("weight")][:-1]
    agent_cfg.actor.hidden_dims = dims
    agent_cfg.critic.hidden_dims = dims
    ckpt_obs = sd["mlp.0.weight"].shape[1]
    print(f"[eval] checkpoint obs {ckpt_obs}, dims {dims}")

    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    obs = unwrap_obs(env.get_observations())
    env_obs = obs["actor"].shape[-1]
    if env_obs != ckpt_obs:
        raise SystemExit(f"obs mismatch: env {env_obs}, checkpoint {ckpt_obs}")
    return policy, obs

def unwrap_obs(ret):
    """get_observations returns either (obs, extras) or the obs container.
 
    Never unpack this blindly: if it returns a TensorDict, `obs, _ = ret`
    iterates over its KEYS and silently binds strings instead of tensors
    whenever the key count happens to match.
    """
    if isinstance(ret, tuple):
        ret = ret[0]
    if not hasattr(ret, "__getitem__"):
        raise SystemExit(f"unexpected observation type: {type(ret)}")
    return ret

def dones_split(env, dones): 
    """
    RslRlVecEnvWrapper collapses terminated and truncated into one dones flag, but the two mean opposite things here: truncation is the clip
    finishing, termination is a failure.
    """
    tm = env.unwrapped.termination_manager
    return tm.terminated, tm.time_outs

def anchor_errors(env):
    """
    3D and z-only anchor position error
    """
    cmd = env.unwrapped.command_manager.get_term("motion")
    err_3d = cmd.metrics["error_anchor_pos"]
    err_z = (cmd.robot_anchor_pos_w[:, 2] - cmd.anchor_pos_w[:, 2]).abs()
    return err_3d, err_z

def category_of(clip_name: str) -> str:
    for cat in ("LAFAN1", "LocoMuJoCo", "aist", "fitness", "humanml",
                "perform", "idea400", "humman", "GRAB", "music", "dance"):
        if cat.lower() in clip_name.lower():
            return cat
    return "other"
    
def main():
    cfg = load_config()
    configure_torch_backends()  # noqa: F821
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(cfg.split_file) as f:
        split_idx = json.load(f)[cfg.split]
    n_clips = len(split_idx)
    num_envs = n_clips if cfg.num_envs is None else cfg.num_envs

    env, agent_cfg = build_eval_env(cfg, num_envs, device)
    policy, obs = load_policy(cfg, env, agent_cfg, device) 

    cmd = env.unwrapped.command_manager.get_term("motion")
    clip_names = list(cmd.clip_names)
    max_steps = cfg.max_steps or int(cmd.clip_lengths.max().item())

    acc = ClipMetrics(num_envs, cfg.metric_keys, device)
    
    with torch.inference_mode():
        for step in range(max_steps):
            obs, _, dones, _ = env.step(policy(obs))
            terminated, truncated = dones_split(env, dones)
            err_3d, err_z = anchor_errors(env)
            acc.update(cmd.metrics, err_3d, err_z)
            acc.retire(terminated, truncated)

            
            if not acc.alive.any():
                print(f"[eval] all envs finished at step {step}")
                break
            if step % 200 == 0:
                print(f"  step {step:5d}  alive {int(acc.alive.sum())}/{num_envs}")

    results = acc.finalize(cfg.succ_threshold)

    rows = []
    for i in range(num_envs):
        clip = clip_names[i] if i < len(clip_names) else f"env{i}"
        row = {"clip": clip, "category": category_of(clip)}
        row.update({k: float(v[i]) for k, v in results.items()})
        rows.append(row)
    
    with open(cfg.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[eval] wrote {cfg.out_csv}")

    summarize(rows, cfg)

    env.close()
    del env
    torch.cuda.empty_cache()

def summarize(rows: list[dict], cfg: EvalConfig):
    def block(label: str, subset: list[dict]):
        n = len(subset)
        if n == 0:
            return
        avg = lambda k: sum(r[k] for r in subset) / n  # noqa: E731
        print(
            f"{label:<14} n={n:<5} joint {avg('error_joint_pos'):.3f}  "
            f"peak3d {avg('peak3d'):.3f}  succ_z {100 * avg('succ_z'):.1f}%  "
            f"succ_3d {100 * avg('succ_3d'):.1f}%  "
            f"completed {100 * avg('completed'):.1f}%"
        )

    print(f"\nthreshold {cfg.succ_threshold} m\n" + "-" * 82)
    block("ALL", rows)
    print("-" * 82)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        block(cat, by_cat[cat])
    print("-" * 82)
    print(
        "Single-run differences below ~5 percentage points sit inside seed "
        "noise. Any claim needs several seeds with mean and spread."
    )


    
    
if __name__ == "__main__":
    main()