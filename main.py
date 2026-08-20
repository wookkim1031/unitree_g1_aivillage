def main():
    import config_loader

    config = config_loader.load_config_json("./configs/train_config.json")

    print(config)


if __name__ == "__main__":
    main()
