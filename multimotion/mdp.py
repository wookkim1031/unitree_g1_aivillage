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
        self._sampler = None
        self._force_ids = None
        # Base sets up buffers, then resets all envs 
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

        # Multi-Clip buffer built here 
        # Puts every environment into a valid starting state 
        self.clip_starts = torch.as_tensor(starts[keep], device=self.device)
        self.clip_lengths = torch.as_tensor(lengths[keep], device=self.device)
        self.clip_names = [names[i] for i in keep]
        self.num_clips = int(keep.size) 
        
        self.clip_begin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end   = torch.zeros_like(self.clip_begin)

        # Length-proportional by default; uniform over clips over-weights short ones
        base = (
            torch.ones(self.num_clips, device=self.device)
            if cfg.sample_uniform_over_clips
            else self.clip_lengths.float().to(self.device)
        )
        self._clip_base = base / base.mean()
        self._clip_scale = torch.ones_like(self._clip_base)
        
        self.env_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._det_ptr = 0
        
        # Envs sampled during super().__init__ took the fallback path; redo them
        # now that the clip tables exist.
        self._uniform_sampling(torch.arange(self.num_envs, device=self.device))
    
        print(f"[MultiMotionCommand] {self.num_clips} clips, "
              f"{int(self.clip_lengths.sum()):,} frames, split={cfg.split}")

    def set_clip_weights(self, scale):
        assert scale.numel() == self.num_clips
        self._clip_scale.copy_(scale.to(self._clip_scale))
        
    def _uniform_sampling(self, env_ids):
        """
        Draw a (clip, phase) per env. Called by the base _resample_command
        before it reads time_steps back to teleport 

        args: 
            env_ids: tensor of indices of the environment being resetted 
        """
        if getattr(self, "clip_starts", None) is None: 
            return super()._uniform_sampling(env_ids)   # during super().__init__
        
        n = int(env_ids.numel())
        
        # deterministic clips for exhaustive evaluation
        # iterates through clip in order
        if self.cfg.deterministic_clips:
            # det_ptr which clip to hand out next 
            clip = (self._det_ptr + torch.arange(n, device=self.device)) % self.num_clips
            self._det_ptr = int((self._det_ptr + n) % self.num_clips)
        else: 
            # multinomial samples with replacement, so 1024 env draw some clips twice
            clip = torch.multinomial(self._clip_weights, n, replacement=True)

        self.env_clip[env_ids] = clip
        self.clip_begin[env_ids] = self.clip_starts[clip]
        self.clip_end[env_ids] = self.clip_starts[clip] + self.clip_lengths[clip] - 1

        if self.cfg.random_start_phase:
            span = (self.clip_lengths[clip] - self.cfg.min_remaining_frames).clamp(min=1)
            phase = (torch.rand(n, device=self.device) * span.float()).long().clamp(max=span - 1)
        else:
            phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.time_steps[env_ids] = self.clip_starts[clip] + phase
    
        p = self._clip_weights / self._clip_weights.sum()
        self.metrics["sampling_entropy"][:] = -(p * (p + 1e-12).log()).sum()
        self.metrics["sampling_top1_prob"][:] = p.max()
        
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

    @property
    def _clip_weights(self):
        return self._clip_base * self._clip_scale

@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
  split_file: str | None = None      # .split.json from merge_motions; None = all
  split: str = "train"
  min_clip_frames: int = 50          # 1 s at 50 Hz
  min_remaining_frames: int = 25     # frames left after the sampled start phase
  sample_uniform_over_clips: bool = False
  random_start_phase: bool = False
  deterministic_clips: bool = True  # env e -> clip (ptr+e) % K; exhaustive eval
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