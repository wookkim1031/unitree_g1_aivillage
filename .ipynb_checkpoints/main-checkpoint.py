def main():
    import config_loader
    import utils
    from pathlib import Path
    import multimotion

    # Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation
    # TASK = "Unitree-G1-Tracking-MultiClip-No-State-Estimation"
    TASK = "G1-MultiClip"
    config = utils.TrainConfig.from_task(TASK)

    config = config_loader.load_and_overwrite_train_config(config, Path("configs/ppo_training.yaml"))

    config.agent.resume = False
    # config.agent.load_run = "2026-08-16_15-11-36"
    # config.agent.load_checkpoint = "model_2999.pt"
    config.agent.run_name =  "multiclip_smoke"
    config.motion_file = "/opt/nb/johan/data/motion_file/phuma_track_v2.npz"
    config.video = False
    # config.video_length = 200
    # config.video_interval = 2000
    config.enable_nan_guard = False
    config.torchrunx_log_dir = None
    config.gpu_ids = [0]

    utils.launch_training(task_id=TASK, args=config)

if __name__ == "__main__":
    main()
