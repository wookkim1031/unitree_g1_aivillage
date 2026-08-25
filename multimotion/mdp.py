from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
import numpy as np
import torch
from pathlib import Path
import json
from dataclasses import dataclass

class MultiMotionCommand(MotionCommand): 
    """
    MotionCommand over a merged multi-clip buffer
    """
    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg, env): 
        super().__init__(cfg,env)

        with np.load(cfg.motion_file) as d: 
            starts = np.asarray(d["clip_starts"], dtype=np.int64)
            lengths = np.asarray(d["clip_lengths"], dtype=np.int64)
            names = [str(n) for n in np.asarray(d["clip_names"])]
        keep = np.arange(len(lengths))
        if cfg.split_file:
            split = json.loads(Path(cfg.split_file).read_text())
            keep = np.asarray(split[cfg.split], dtype=np.int64)
            if keep.size == 0:
                raise ValueError(f"split {cfg.split!r} is empty")

        keep = keep[lengths[keep] >= max(2, cfg.min_clip_frames)]

        self.clip_starts = torch.as_tensor(starts[keep], device=self.device)
        self.clip_lengths = torch.as_tensor(lengths[keep], device=self.device)
        self.clip_names = [names[i] for i in keep]
        self.num_clips = int(keep.size) 
        
        self.clip_begin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end   = torch.zeros_like(self.clip_begin)

        # Length-proportional by default; uniform over clips over-weights short ones
        self._clip_weights = (
          torch.ones(self.num_clips, device=self.device)
          if cfg.sample_uniform_over_clips
          else self.clip_lengths.float())
    
        self.env_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._det_ptr = 0
        
        # Envs sampled during super().__init__ took the fallback path; redo them
        # now that the clip tables exist.
        self._uniform_sampling(torch.arange(self.num_envs, device=self.device))
    
        print(f"[MultiMotionCommand] {self.num_clips} clips, "
              f"{int(self.clip_lengths.sum()):,} frames, split={cfg.split}")
    
        
    def _uniform_sampling(self, env_ids):
        """
        Draw a (clip, phase) per env. Called by the base _resample_command
        before it reads time_steps back to teleport 
        """
        
        if not hasattr(self, "clip_starts"):
          return super()._uniform_sampling(env_ids)  # during super().__init__
        
        n = int(env_ids.numel())
        if n == 0:
          return
        
        # deterministic clips for exhaustive evaluation
        # iterates through clip in order
        if self.cfg.deterministic_clips:
          clip = (self._det_ptr + torch.arange(n, device=self.device)) % self.num_clips
          self._det_ptr = int((self._det_ptr + n) % self.num_clips)
        else:
          clip = torch.multinomial(self._clip_weights, n, replacement=True)
        self.env_clip[env_ids] = clip
        
        self.clip_begin[env_ids] = self.clip_starts[clip]
        self.clip_end[env_ids]   = self.clip_starts[clip] + self.clip_lengths[clip] - 1

        # sample uniformly from valid start positions
        if self.cfg.random_start_phase:
          span = (self.clip_lengths[clip] - self.cfg.min_remaining_frames).clamp(min=1)
          phase = (torch.rand(n, device=self.device) * span.float()).long().clamp(max=span - 1)
          self.time_steps[env_ids] = self.clip_starts[clip] + phase
        else:
        # Always start from beginning of a clip
          phase = torch.zeros(n, dtype=torch.long, device=self.device)

        # Sets absolute time steps for the frame,
        self.time_steps[env_ids] = self.clip_starts[clip] + phase
        
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.num_clips, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _update_command(self, env_ids: torch.Tensor | None = None):
        """Clamp to clip_end so the base class doesn't step past this clip
        into the next one's opening frames."""
        super()._update_command(env_ids)
        if env_ids is None:
            self.time_steps.clamp_(max=self.clip_end)
        else:
            self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], self.clip_end[env_ids])
        
    @property
    def clip_phase(self):
        span = (self.clip_end - self.clip_begin).clamp(min=1)
        return (self.time_steps - self.clip_begin).float() / span.float()

@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
  split_file: str | None = None      # .split.json from merge_motions; None = all
  split: str = "train"
  min_clip_frames: int = 50          # 1 s at 50 Hz
  min_remaining_frames: int = 25     # frames left after the sampled start phase
  sample_uniform_over_clips: bool = False
  random_start_phase: bool = True
  deterministic_clips: bool = False  # env e -> clip (ptr+e) % K; exhaustive eval
  # lookahead: tuple[int, ...] = (5, 10, 20)   # frames ahead: 0.1 / 0.2 / 0.4 s
  # lookahead_frame: str = "ref"       # "ref" is deployable, "root" is not
    
  def build(self, env):
    if self.sampling_mode != "uniform":
      raise ValueError(
        f"sampling_mode must be 'uniform', got {self.sampling_mode!r} — "
        "adaptive bins span clip boundaries")
    return MultiMotionCommand(self, env)

def clip_timeout(env) -> torch.Tensor:
    cmd = env.command_manager.get_term("motion")
    return cmd.time_steps >= cmd.clip_end