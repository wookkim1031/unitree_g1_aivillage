"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import torch
from typing import Literal, cast

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
"""
PlayConfig
"""
@dataclass
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser", "none"] = "auto"
    no_terminations: bool = False
    """Disable all termination conditions (useful for viewing motions with dummy agents)."""
    
    # Internal flag used by demo script.
    _demo_mode: bool = False

"""
tryo turns this dataclass into a CLI. because env itself is a nested dataclass

TrainConfig.from_task pulls the defaults from the registry
"""
# @dataclass(frozen=True)
@dataclass(frozen=False)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    motion_file: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    
    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()
    
    cfg.agent.seed = seed
    cfg.env.seed = seed
    
    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
    
    # Check if this is a tracking task by checking for motion command.
    # Tracking tasks require -- motion-file velocity tasks don't
    is_tracking_task = "motion" in cfg.env.commands and isinstance(
        cfg.env.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task:
        if not cfg.motion_file:
            raise ValueError("For tracking tasks, --motion-file must be set ...")
        
        motion_path = Path(cfg.motion_file).expanduser().resolve()
        
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        
        motion_cmd.motion_file = str(motion_path)
        print(f"[INFO] Using motion file: {motion_cmd.motion_file}")
        
        # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
        if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
            print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    resume_path: Path | None = None
    if cfg.agent.resume:
        # Load checkpoint from local filesystem.
        resume_path = get_checkpoint_path(
            log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
    print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner_kwargs = {}
    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    # runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    # Only write config files from rank 0 to avoid race conditions.
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):

    # Create log directory once before launching workers.
    log_root_path = (Path("logs") / "rsl_rl" / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name
    
    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)
    
    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        os.environ["MUJOCO_GL"] = "egl"
    
    if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
    # Multi-GPU: use torchrunx.
        import torchrunx
        
        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)
        
        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
            # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
        else:
            # Default: put logs in training directory.
            os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")
        
        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
          hostnames=["localhost"],
          workers_per_host=num_gpus,
          backend=None,  # Let rsl_rl handle process group initialization.
          copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)

from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
          
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    # Reading actor and critic width from the checkpoint rather than guessing
    sd = torch.load(str(resume_path), map_location="cpu")["actor_state_dict"]
    dims = [sd[k].shape[0] for k in sorted(sd) if k.endswith("weight")][:-1]
    agent_cfg.actor.hidden_dims = dims
    agent_cfg.critic.hidden_dims = dims
    print(f"[play] obs {sd['mlp.0.weight'].shape[1]}, dims {dims}")
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    _inner, _n = policy, [0]
    def policy(obs, *a, **k):
        if _n[0] == 0:
            import numpy as np
            arr = obs["actor"][0].detach().cpu().numpy()
            np.savetxt("/tmp/obs_mjlab.txt", arr, fmt="%.9f")
            mc = env.unwrapped.command_manager.get_term("motion")
            print(f"[obs dump] wrote {arr.size} values, "
                  f"start frame {int(mc.time_steps[0])}", flush=True)
        _n[0] += 1
        return _inner(obs, *a, **k)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  elif resolved_viewer == "none":
    obs, _ = env.get_observations()
    for _ in range(cfg.video_length + 10):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()
