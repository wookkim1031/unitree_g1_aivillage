def main():
    import config_loader
    import utils
    from pathlib import Path

    TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
    config = utils.TrainConfig.from_task(TASK)

    config = config_loader.load_and_overwrite_train_config(config, Path("configs/ppo_training.yaml"))

    config.agent.resume = False
    # config.agent.load_run = "2026-08-16_15-11-36"
    # config.agent.load_checkpoint = "model_2999.pt"
    config.agent.run_name = "ft_walk_small_from_2999"
    config.motion_file = "/opt/nb/johan/data/motion_file/phuma_g1_v2/LocoMuJoCo/walk_chunk_0000.npz"
    config.video = False
    # config.video_length = 200
    # config.video_interval = 2000
    config.enable_nan_guard = False
    config.torchrunx_log_dir = None
    config.gpu_ids = [0]

    utils.launch_training(task_id=TASK, args=config)


if __name__ == "__main__":
    main()
