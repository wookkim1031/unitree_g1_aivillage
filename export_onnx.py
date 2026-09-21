from pathlib import Path
import torch
from eval import load_config, build_eval_env, load_policy

def main():
    cfg = load_config()
    device = cfg.device
    env, agent_cfg = build_eval_env(cfg, 1, cfg.device)
    _, _, runner = load_policy(cfg, env, agent_cfg, cfg.device)

    out = Path(cfg.checkpoint_file).parent / "exported"
    runner.export_policy_to_onnx(str(out), "policy.onnx")
    env.close()

if __name__ == "__main__":
    main()