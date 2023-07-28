import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from torch import optim
# from Video2Roll_dataset import Video2RollDataset
from torch.utils.data import DataLoader

import dataset
import model
import wandb
from dataset import augmentation
from dataset.balance_data import MultilabelBalancedRandomSampler
from trainer import Video2Roll_Trainer
from utils.util import get_current_time, load_config, validate_config

# import Video2RollNet







class F1Loss:
    def __init__(self):
        pass

    def __call__(self, pred, target):
        eps = 1e-8
        label_np_T = target.to(torch.float32)
        roll_output_T = torch.sigmoid(pred)

        tp = torch.sum(label_np_T * roll_output_T, dim=0).to(torch.float32)
        tn = torch.sum((1 - label_np_T) * (1 - roll_output_T), dim=0).to(torch.float32)
        fp = torch.sum((1 - label_np_T) * roll_output_T, dim=0).to(torch.float32)
        fn = torch.sum(label_np_T * (1 - roll_output_T), dim=0).to(torch.float32)

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)

        f1 = 2 * p * r / (p + r + eps)
        f1 = torch.nansum(f1) / (torch.sum(~torch.isnan(f1)) + eps)
        return 1 - f1
        # f1 = 2* (p*r) / (p + r + eps)
        # f1 = f1.clamp(min=eps, max=1-eps)
        # return 1 - f1.mean()


class Hybrid:
    def __init__(self):
        self.f1 = F1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, pred, target):
        f1_loss = self.f1(pred, target)
        bce_loss = self.bce(pred, target)
        print(f1_loss)
        return bce_loss


def main(cfg):
    save_model_path = "./experiments"
    expr_count = len(os.listdir(save_model_path))
    now = get_current_time()
    save_model_path = os.path.join(
        save_model_path, f"{expr_count+1}_{now}_{cfg['wandb']['run_name']}"
    )
    os.makedirs(save_model_path, exist_ok=True)
    with open(os.path.join(save_model_path, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    cfg = EasyDict(cfg)

    if cfg.wandb.use:
        wandb.init(entity="lijm1358", project="vmt", name=cfg.wandb.run_name, config=cfg)

    if cfg.train_dataset.augmentation is not None:
        train_transform = getattr(augmentation, cfg.train_dataset.augmentation)()
    else:
        train_transform = None

    train_dataset = getattr(dataset, cfg.train_dataset.type)(
        **cfg.train_dataset.args, transform=train_transform, subset="train"
    )
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
    train_data_loader = DataLoader(
        train_dataset, **cfg.train_dataset.loader_args, sampler=train_sampler
    )

    test_data_loader = []
    for test_ds in cfg.test_dataset:
        if test_ds.augmentation is not None:
            test_transform = getattr(augmentation, test_ds.augmentation)()
        else:
            test_transform = None
        test_dataset = getattr(dataset, test_ds.type)(
            **test_ds.args, transform=test_transform, subset="test"
        )
        test_data_loader.append(DataLoader(test_dataset, **test_ds.loader_args))
    device = cfg.device

    net = getattr(model, cfg.model.type)(**cfg.model.args)
    if cfg.model.load_ckpt.path is not None:
        net.load_state_dict(torch.load(cfg.model.load_ckpt.path, map_location=device))
    if cfg.model.load_ckpt.fc_out is not None:
        net.fc = nn.Linear(128, cfg.model.load_ckpt.fc_out)

    net = net.to(device)
    optimizer = getattr(optim, cfg.optimizer.type)(net.parameters(), **cfg.optimizer.args)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = F1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    # data_loader, test_data_loader, model, criterion, optimizer, lr_scheduler, epochs, save_model_path
    trainer = Video2Roll_Trainer(
        data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        model=net,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epochs=cfg.epochs,
        save_model_path=save_model_path,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, default="./config.yaml")

    args = parser.parse_args()

    cfg = load_config(args.config)
    # validate_config(EasyDict(cfg))

    main(cfg)
