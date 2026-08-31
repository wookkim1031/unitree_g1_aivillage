# ----------------------------------------
# Config
# ----------------------------------------
from dataclasses import asdict, dataclass, field
import torch
from pathlib import Path
import math
import re
import collections


@dataclass
class SamplingWeightsCfg:
    enabled: bool = True
    sweep_every: int = 500
    warmup_iters: int = 500
    exclude_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    max_sweep_frames: int = 1600
    sweep_num_envs: int = 512
    """Cap on rollout length per clip. PHUMA clips are 199-200 frames at 50 Hz."""

    # weight udpate
    # alpha: weight multiplier for clips that passed <1
    success_decay: float = 0.9
    # beta: weight multiplier for clips that failed >1
    failure_boost: float = 1.4
    """Equilibrium is bang-bang: clips failing more often than
    q* = log(1/alpha) / (log(beta) + log(1/alpha)) drift to w_max, the rest to
    w_min. With the defaults above, q* ~ 0.24. Transit time between the two is
    log(w_max/w_min) / log(beta) ~ 14 sweeps, so budget for it."""
    w_min: float = 0.1
    w_max: float = 10.0
    rank: int = 0
    dev_threshold: float = 0.5 

    log_dir: str | None = None
    
# ----------------------------------------
# Weight buffer
# ----------------------------------------
class MotionSamplingWeights: 
    # Persistent per-clip sampling weights with a multiplicative update

    def __init__(
        self, 
        num_clips: int, 
        device: torch.device | str,
        cfg: SamplingWeightsCfg,
        clip_names: list[str] | None = None,
    ) -> None: 
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_clips = num_clips
        self.clip_names = clip_names

        self.weights = torch.ones(num_clips, device=self.device)
        self.num_sweeps = 0
        self.last_failed: torch.Tensor | None = None

        self._dir = Path(cfg.log_dir) if cfg.log_dir else None
        if self._dir is not None:
            (self._dir / "failed_motions").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def update(self, failed: torch.Tensor, epoch: int) -> None:
        """failed: (num_clips,) bool from the sweep."""
        cfg = self.cfg
        f = failed.to(self.device).float()

        mult = cfg.success_decay ** (1.0 - f) * cfg.failure_boost**f
        self.weights = (self.weights * mult).clamp(cfg.w_min, cfg.w_max)

        self.num_sweeps += 1
        self.last_failed = failed
        self._persist(failed, epoch)

    @torch.no_grad()
    def probs(self) -> torch.Tensor:
        return self.weights / self.weights.sum()

    @torch.no_grad()
    def sample(self, n:int) -> torch.Tensor: 
        if not self.cfg.enabled or self.num_sweeps == 0:
            return torch.randint(0, self.num_clips, (n,), device=self.device)
        return torch.multinomial(self.probs(), n, replacement=True)

    def _persist(self, failed: torch.Tensor, epoch: int) -> None: 
        if self._dir is None: 
            return
        ids = torch.nonzero(failed).flatten().tolist()
        path = self._dir / "failed_motions" / f"failed_motions_epoch_{epoch}_rank_{self.cfg.rank}.txt"
        if self.clip_names is not None:
            lines = [f"{i}\t{self.clip_names[i]}" for i in ids]
        else:
            lines = [str(i) for i in ids]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        torch.save(
            {"weights": self.weights.cpu(), "num_sweeps": self.num_sweeps},
            self._dir / "sampling_weights.pt",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_file():
            blob = torch.load(path, map_location=self.device)
            self.weights = blob["weights"].to(self.device) 
            self.num_sweeps = int(blob.get("num_sweeps", 1)) 
            print(f"[dyn-sample] restored weights from {path}")
            return

        fdir = path / "failed_motions"
        if not fdir.exists():
            print(f"[dyn-sample] nothing to restore from {path}") 
            return
        pattern = re.compile(r"failed_motions_epoch_(\d+)\.txt")
        best, best_epoch = None, -1
        for fp in fdir.iterdir():
            m = pattern.match(fp.name)
            if m and int(m.group(1)) > best_epoch:
                best, best_epoch = fp, int(m.group(1))
        if best is None:
            print(f"[dyn-sample] no epoch files in {fdir}")
            return
        ids = [
            int(line.split("\t")[0])
            for line in best.read_text().splitlines()
            if line.strip()
        ]
        failed = torch.zeros(self.num_clips, dtype=torch.bool, device=self.device)
        failed[torch.tensor(ids, device=self.device, dtype=torch.long)] = True
        self.update(failed, epoch=best_epoch)
        print(f"[dyn-sample] rebuilt weights from {best.name} ({len(ids)} failed)")

    
    @torch.no_grad()
    def log_dict(self) -> dict[str, float]:
        p = self.probs()
        uniform = 1.0 / self.num_clips
        ess = 1.0 / p.pow(2).sum()
        out = {
            "dyn_sample/top1_ratio": (p.max() / uniform).item(),
            "dyn_sample/ess_frac": (ess / self.num_clips).item(),
            "dyn_sample/at_wmax_frac": (self.weights >= self.cfg.w_max - 1e-6)
            .float()
            .mean()
            .item(),
            "dyn_sample/at_wmin_frac": (self.weights <= self.cfg.w_min + 1e-6)
            .float()
            .mean()
            .item(),
            "dyn_sample/num_sweeps": float(self.num_sweeps),
        }
        if self.last_failed is not None:
            out["dyn_sample/fail_frac"] = self.last_failed.float().mean().item()
        return out



class SweepEvaluator:
    # Deterministic pass over the whole clip library inside the training env
    def __init__(self, env, command_name: str = "motion") -> None: 
        self.env = env
        self.cmd = env.command_manager.get_term(command_name)
        self.device = env.device
        self.num_envs = env.num_envs

    def _tracked_body_slice(self, exclude: tuple[str, ...]) -> torch.Tensor: 
        """
        Indices INTO the command's tracked-body axis, minus excluded names.

        names: 0 ... 29 robot body index. All bodies in the MJCF 
        tracked: 0 ... 13 the 14 bodies your command actually tracks
        npz body index: 0 ... 29 the merged buffer's 30 bodies
        """
        robot = self.env.scene["robot"]
        names = list(self.env.scene["robot"].body_names)
        tracked = [names[i] for i in self.cmd.body_indexes.tolist()]
        # keep holds positions on the tracked axis 
        keep = [k for k, n in enumerate(tracked) if n not in exclude]
        dropped = [n for n in tracked if n in exclude]
        if dropped:
            print(f"[dyn-sample] excluded from failure criterion: {dropped}")
        keep_t = torch.tensor(keep, device=self.device, dtype=torch.long)
        return keep_t, [tracked[k] for k in keep]
    """
    def _disable_events(self):
        
        _mode_term_names: dict from mode
        
            - startup: once at env construction. body_ipos, body_subtreemaas, geom_friction, encoder bias
            - reset: every episode reset, per env
            - interval: every N seconds during the episode. push_robot with its 1-3s interval is here
        
        em = self.env.event_manager
        saved = {}
        for mode in list(em._mode_term_names.keys()):
            saved[mode] = list(em._mode_term_names[mode])
            em._mode_term_names[mode] = []
        return saved
    
    def _restore_events(self, saved):
        em = self.env.event_manager
        for mode, names in saved.items():
            em._mode_term_names[mode] = names
    """
    def run(self, policy, cfg: SamplingWeightsCfg) -> dict[str, torch.Tensor]:
        """
        Roll out every clip once, deterministically from frame 0
        """
        num_clips = int(self.cmd.clip_lengths.numel())
        keep, kept_names = self._tracked_body_slice(cfg.exclude_body_names)

        print("[sweep] has robot_body_pos_w:", hasattr(self.cmd, "robot_body_pos_w"))
        print("[sweep] body_pos_relative_w:", self.cmd.body_pos_relative_w.shape)
        print("[sweep] robot_body_pos_w   :", self.cmd.robot_body_pos_w.shape)
        print("[sweep] body_indexes       :", len(self.cmd.body_indexes),
              "keep:", len(keep))

        # max body deviation seen, per clip
        peak = torch.zeros(num_clips, device=self.device)
        # running sum of joint error, per clip 
        jerr = torch.zeros(num_clips, device=self.device)
        # frames counted, per clip
        jcnt = torch.zeros(num_clips, device=self.device)
        # saved_events = self._disable_events()
        was_training = getattr(policy, "training", False)
        over_run = torch.zeros(self.num_envs, device=self.device)      # current streak
        max_run  = torch.zeros(num_clips, device=self.device)          # longest per clip
        prev_ids = self.cmd.env_clip.clone()

        first_fail = torch.full((num_clips,), float("inf"), device=self.device)

        
        if hasattr(policy, "eval"):
            policy.eval()

        # saved_flags = (self.cmd.cfg.deterministic_clips, self.cmd.cfg.random_start_phase)
        # self.cmd.cfg.deterministic_clips = False
        # self.cmd.cfg.random_start_phase = False
        self.cmd._det_ptr = 0

        robot = self.env.scene["robot"]
        body_ids = self.cmd.body_indexes

        try: 
            with torch.inference_mode():
                obs, _ = self.env.reset()
            for frame in range(cfg.max_sweep_frames):
                with torch.inference_mode():
                    obs, _, _, _, _ = self.env.step(policy(obs))
                with torch.no_grad():
                    ids = self.cmd.env_clip
                    changed = ids != prev_ids
                    over_run = torch.where(changed, torch.zeros_like(over_run), over_run)
                    prev_ids = ids.clone()

                    ref = self.cmd.body_pos_relative_w[:, keep, :]
                    cur = self.cmd.robot_body_pos_w[:, keep, :]
                    per_body = torch.linalg.norm(ref - cur, dim=-1)
                    dev, worst = per_body.max(dim=-1)
                    dev = dev.clone()
                    jd = torch.linalg.norm(
                        self.cmd.joint_pos - robot.data.joint_pos, dim=-1
                    ).clone()

                    span = (self.cmd.clip_end - self.cmd.clip_begin).clamp(min=1).float()
                    frac = (self.cmd.time_steps - self.cmd.clip_begin).float() / span

                    exceeded = dev > cfg.dev_threshold
                    tmp = torch.where(exceeded, frac, torch.full_like(frac, float("inf")))
                    first_fail.index_reduce_(0, ids, tmp, "amin")

                    over_run = torch.where(exceeded, over_run + 1.0, torch.zeros_like(over_run))
                    max_run.index_reduce_(0, ids, over_run, "amax")

                    peak.index_reduce_(0, ids, dev, "amax")
                    jerr.index_add_(0, ids, jd)
                    jcnt.index_add_(0, ids, torch.ones_like(jd))

        finally: 
            # self._restore_events(saved_events)
            # self.cmd.cfg.deterministic_clips, self.cmd.cfg.random_start_phase = saved_flags
            if was_training and hasattr(policy, "train"):
                policy.train()
                

        return {
            "failed": torch.isfinite(first_fail),
            "first_fail": first_fail,
            "peak_dev": peak,
            "joint_err": jerr / jcnt.clamp(min=1.0),
            "max_run": max_run,
        }


        