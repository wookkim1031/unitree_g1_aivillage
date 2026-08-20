def main():
    import config_loader
    import utils

    config = config_loader.load_config_json("./configs/train_config.json")

    TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
    # train_config = TrainConfig.from_task(TASK)

    utils.launch_training(task_id=TASK, args=config)


if __name__ == "__main__":
    main()
