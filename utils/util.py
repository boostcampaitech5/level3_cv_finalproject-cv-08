from datetime import datetime, timezone, timedelta
from easydict import EasyDict
import yaml


def get_current_time():
    now = datetime.now()
    timezone_kst = timezone(timedelta(hours=9))
    now = now.astimezone(timezone_kst)
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    return now


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def validate_config(cfg):
    class ConfigError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if cfg.train_dataset.args.min_key != cfg.test_dataset.args.min_key:
        raise ConfigError(
            f"'min_key' in 'train_dataset' ({cfg.train_dataset.args.min_key}) is not equal to 'min_key' in 'test_dataset' ({cfg.test_dataset.args.min_key})."
        )

    if cfg.train_dataset.args.max_key != cfg.test_dataset.args.max_key:
        raise ConfigError(
            f"'min_key' in 'train_dataset' ({cfg.train_dataset.args.max_key}) is not equal to 'min_key' in 'test_dataset' ({cfg.test_dataset.args.max_key})."
        )

    if (
        cfg.train_dataset.args.max_key - cfg.test_dataset.args.min_key + 1
        != cfg.model.args.num_classes
    ):
        raise ConfigError(
            f"'num_classes' of model must be {cfg.train_dataset.args.max_key - cfg.test_dataset.args.min_key + 1}, not {cfg.model.args.num_classes}."
        )
