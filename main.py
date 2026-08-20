def main():
    import config_loader
    import utils
    from pathlib import Path

    TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
    config = utils.TrainConfig.from_task(TASK)

    config = config_loader.load_and_overwrite_train_config(config, Path("configs/ppo_training.yaml"))

    utils.launch_training(task_id=TASK, args=config)


if __name__ == "__main__":
    main()
