"""Pick fail_threshold from a trained checkpoint.

Builds the eval-configured env (no failure terminations, no episode cap, no
disturbances), runs the sweep twice against a checkpoint, and prints the
peak_dev distribution so you can choose a threshold that puts a useful
fraction of clips on the failing side.

Run:  uv run calibrate_threshold.py logs/rsl_rl/g1_multiclip/<run>/model_XXXX.pt
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import config_loader
import multimotion  # registers G1-MultiClip
import utils
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from multimotion import _settings
from multimotion.env_cfg import make_multiclip_cfg
from multimotion.dynamic_sampling import SamplingWeightsCfg, SweepEvaluator

TASK = "G1-MultiClip"
DEVICE = "cuda:0"

ckpt = Path(sys.argv[1])
assert ckpt.exists(), ckpt

# --- config, same as training -------------------------------------------- #
cfg = utils.TrainConfig.from_task(TASK)
cfg = config_loader.load_and_overwrite_train_config(
    cfg, Path("configs/ppo_training.yaml")
)
motion_cmd = cfg.env.commands["motion"]
motion_path = Path(cfg.motion_file).expanduser().resolve()
motion_cmd.motion_file = str(motion_path)

print(f"[cal] checkpoint : {ckpt}")
print(f"[cal] motion     : {motion_path.name}")
print(f"[cal] split      : {motion_cmd.split} / {motion_cmd.split_file}")

# --- eval env: this is what the sweep should always run in ---------------- #
sweep_cfg = make_multiclip_cfg(
    _settings(
        motion_file=str(motion_path),
        split_file=motion_cmd.split_file,
        split=motion_cmd.split,
        eval=True,
    )
)
sweep_cfg.scene.num_envs = 512
sweep_cfg.seed = cfg.env.seed
sweep_env = ManagerBasedRlEnv(cfg=sweep_cfg, device=DEVICE)
cmd = sweep_env.command_manager.get_term("motion")
print(f"[cal] clips      : {cmd.num_clips}")

# --- policy: runner needs a wrapped env of the same obs layout ------------ #
wrapped = RslRlVecEnvWrapper(sweep_env, clip_actions=cfg.agent.clip_actions)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(wrapped, asdict(cfg.agent), None, DEVICE)
runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
policy = runner.get_inference_policy(DEVICE)

sd = torch.load(str(ckpt), map_location="cpu")["actor_state_dict"]
print(f"[cal] ckpt obs   : {sd['mlp.0.weight'].shape[1]}")

# --- sweep twice: identical results prove the measurement is deterministic - #
ds = cfg.dynamic_sampling
ds.log_dir = None
print(f"[cal] sweep frames {ds.max_sweep_frames}  dev_threshold {ds.dev_threshold}")

results = []
for trial in range(2):
    res = sweep_env.command_manager and SweepEvaluator(sweep_env).run(policy, ds)
    results.append(res)
    pd = res["peak_dev"]
    print(f"\n[trial {trial}] peak_dev mean {pd.mean():.4f}  min {pd.min():.4f}  max {pd.max():.4f}")

# --- reproducibility: the binary failed set is what drives the weights ----- #

mr = results[0]["max_run"]
fa = torch.isfinite(results[0]["first_fail"])
print("max_run:", mr.shape, "finite:", fa.sum().item(), "of", fa.numel())
    
a, b = results[0]["first_fail"], results[1]["first_fail"]
fa, fb = torch.isfinite(a), torch.isfinite(b)
print(f"\nfailed set identical: {(fa == fb).all().item()}   "
      f"({fa.sum().item()} vs {fb.sum().item()} of {len(a)} clips)")
if not (fa == fb).all():
    flip = (fa != fb).nonzero().flatten().tolist()
    print(f"  {len(flip)} clips flipped between trials, e.g.:")
    for i in flip[:5]:
        print(f"    {cmd.clip_names[i]}  {a[i]:.3f} vs {b[i]:.3f}")

pdv = results[0]["peak_dev"]
print(f"peak_dev max |diff| = {(pdv - results[1]['peak_dev']).abs().max():.3f}  "
      f"(magnitude is expected to vary; the failed set is not)")

# --- first_fail: fraction of the clip completed before tracking broke ------ #
ff = a
survived = ~torch.isfinite(ff)
print(f"\nsurvived to end : {survived.sum().item()} / {len(ff)} "
      f"({survived.float().mean():.1%})")
print(f"failed          : {(~survived).sum().item()} "
      f"({(~survived).float().mean():.1%})  <- this is fail_frac")

f = ff[~survived]
if f.numel():
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q = torch.quantile(f, torch.tensor(qs, device=f.device))
    print("\nwhen failing clips break (fraction of clip elapsed):")
    for p, v in zip(qs, q.tolist()):
        print(f"  {p:4.0%}  {v:6.3f}")

print("\nfail fraction by dev_threshold — rerun with a different "
      "dev_threshold to move this; the sweep uses "
      f"{ds.dev_threshold} m")

# --- earliest failures (hardest) and survivors ----------------------------- #
order = torch.argsort(ff)
print("\nbreaks earliest (hardest 20):")
for i in order[:20].tolist():
    print(f"  {ff[i]:6.3f}  peak {pdv[i]:7.2f}  {cmd.clip_names[i]}")

print(f"\nsurvivors (first 10 of {survived.sum().item()}):")
for i in survived.nonzero().flatten()[:10].tolist():
    print(f"  peak {pdv[i]:7.2f}  {cmd.clip_names[i]}")

sweep_env.close()
