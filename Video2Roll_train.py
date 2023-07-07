import argparse
# from Video2Roll_dataset import Video2RollDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import yaml

import os

# import Video2RollNet

from trainer import Video2Roll_Trainer
import torch.nn as nn
from dataset.balance_data import MultilabelBalancedRandomSampler

import dataset
import model
from easydict import EasyDict


from utils.util import get_current_time, load_config, validate_config

def main(cfg):
    save_model_path = "./experiments"
    expr_count = len(os.listdir(save_model_path))
    now = get_current_time()
    save_model_path = os.path.join(save_model_path, f"{expr_count+1}_{now}")
    os.makedirs(save_model_path, exist_ok=True)
    with open(os.path.join(save_model_path, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
        
    cfg = EasyDict(cfg)
    
    train_dataset = getattr(dataset, cfg.train_dataset.type)(**cfg.test_dataset.args, subset="train")
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
    train_data_loader = DataLoader(train_dataset, **cfg.train_dataset.loader_args, sampler=train_sampler)
    test_dataset = getattr(dataset, cfg.test_dataset.type)(**cfg.test_dataset.args, subset="test")
    test_data_loader = DataLoader(test_dataset, **cfg.test_dataset.loader_args)
    device = cfg.device

    net = getattr(model, cfg.model.type)(**cfg.model.args)
    if cfg.model.load_ckpt.path is not None:
        net.load_state_dict(torch.load(cfg.model.load_ckpt.path, map_location=device))
    if cfg.model.load_ckpt.fc_out is not None:
        net.fc = nn.Linear(128, cfg.model.load_ckpt.fc_out)
    
    net = net.to(device)
    optimizer = getattr(optim, cfg.optimizer.type)(net.parameters(), **cfg.optimizer.args)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
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
        device=device)
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    # validate_config(EasyDict(cfg))
    
    main(cfg)
