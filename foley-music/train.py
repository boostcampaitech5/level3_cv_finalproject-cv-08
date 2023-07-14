"""
Code modified from:
    https://github.com/chuangg/Foley-Music/blob/main/train.py
"""
import os
from datetime import datetime, timezone, timedelta
import yaml
import json
from tqdm import tqdm
from easydict import EasyDict
import torch
from torch import nn, optim
from pprint import pprint

from core.engine import BaseEngine
from core.dataloaders import DataLoaderFactory
from core.models import ModelFactory
from core.dataloaders.youtube_dataset import YoutubeDataset
from core.criterion import SmoothCrossEntropyLoss
from core.optimizer import CustomSchedule
from core.metrics import compute_epiano_accuracy

from utils import (AverageMeter, MetricTrakcer, init_curves, plot_curves,
                   write_msg_header, write_msg, seed_everything)


class Engine(BaseEngine):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_builder = ModelFactory(cfg)
        self.dataset_builder = DataLoaderFactory(cfg)

        self.train_ds = self.dataset_builder.build(split='train')
        self.test_ds = self.dataset_builder.build(split='val')
        self.ds: YoutubeDataset = self.train_ds.dataset

        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.val_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.model: nn.Module = self.model_builder.build(device=torch.device('cuda'), wrapper=nn.DataParallel)
        # optimizer = optim.Adam(self.model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
        # self.optimizer = CustomSchedule(
        #     # self.cfg.get_int('model.emb_dim'),
        #     self.cfg.model.d_model,
        #     optimizer=optimizer,
        # )
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=float(cfg.lr), betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=4000,
                                                              eta_min=0)

        self.num_epochs = cfg.epochs

        self.learning_curves = init_curves(cfg.exp, 'Accuracy')

    def train(self, epoch=0):
        loss_meter = AverageMeter('Loss', higher_is_better=False)
        acc_meter = AverageMeter('Accuracy', higher_is_better=True)

        self.model.train()
        for i, data in enumerate(tqdm(self.train_ds, leave=False, ncols=80)):
            midi_x, midi_y = data['midi_x'], data['midi_y']

            if self.ds.use_pose:
                feat = data['pose']
            elif self.ds.use_rgb:
                feat = data['rgb']
            elif self.ds.use_flow:
                feat = data['flow']
            else:
                raise Exception('No feature!')

            feat, midi_x, midi_y = (
                feat.cuda(non_blocking=True),
                midi_x.cuda(non_blocking=True),
                midi_y.cuda(non_blocking=True)
            )

            if self.ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
            else:
                control = None

            output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

            loss = self.train_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
    
            acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.PAD_IDX)

            batch_size = len(midi_x)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)

            self.scheduler.step()

        return {'Loss': loss_meter, 'Accuracy': acc_meter}

    def test(self, epoch=0):
        loss_meter = AverageMeter('Loss', higher_is_better=False)
        acc_meter = AverageMeter('Accuracy', higher_is_better=True)
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_ds, leave=False, ncols=80)):
                midi_x, midi_y = data['midi_x'], data['midi_y']

                if self.ds.use_pose:
                    feat = data['pose']
                elif self.ds.use_rgb:
                    feat = data['rgb']
                elif self.ds.use_flow:
                    feat = data['flow']
                else:
                    raise Exception('No feature!')

                feat, midi_x, midi_y = (
                    feat.cuda(non_blocking=True),
                    midi_x.cuda(non_blocking=True),
                    midi_y.cuda(non_blocking=True)
                )

                if self.ds.use_control:
                    control = data['control']
                    control = control.cuda(non_blocking=True)
                else:
                    control = None

                output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

                """
                For CrossEntropy
                output: [B, T, D] -> [BT, D]
                target: [B, T] -> [BT]
                """
                loss = self.val_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

                acc = compute_epiano_accuracy(output, midi_y)

                batch_size = len(midi_x)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc.item(), batch_size)

        return {'Loss': loss_meter, 'Accuracy': acc_meter}

    @staticmethod
    def epoch_time(start_time: float, end_time: float):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_loss = float('inf')
        train_summary = MetricTrakcer(mode='train')
        valid_summary = MetricTrakcer(mode='valid')
        write_msg_header('Accuracy')
        for epoch in range(1, self.num_epochs+1):
            train_metrics = self.train(epoch)
            valid_metrics = self.test(epoch)
            loss = valid_metrics['Loss'].avg

            train_summary.update(train_metrics, epoch)
            valid_summary.update(valid_metrics, epoch)
            write_msg(epoch, train_summary.metrics, valid_summary.metrics, 'Accuracy')
            plot_curves(self.learning_curves, self.cfg.exp, 
                        train_summary.metrics, valid_summary.metrics, 'Accuracy')

            # is_best = loss < best_loss
            # best_loss = min(loss, best_loss)

            ckpt_dir = f'./results/{self.cfg.exp}/checkpoints'
            ckpt_name = f'epoch{epoch:03d}.pt'
            
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(
                {
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                ckpt_path
            )

    def close(self):
        pass


def main():
    cfg_path = './config.yaml'
    with open(cfg_path, 'rt') as f:
        cfg = EasyDict(yaml.safe_load(f))

    seed_everything(cfg.seed)

    cfg.exp = datetime.now(timezone(timedelta(hours=9))).strftime("%y%m%d-%H%M")

    results_dir = f'./results/{cfg.exp}'
    os.makedirs(results_dir, exist_ok=True)

    with open(f'./results/{cfg.exp}/config.yaml', 'w') as f:
        json_str = json.loads(json.dumps(cfg, indent=4))
        yaml.dump(json_str, f)

    engine = Engine(cfg)
    engine.run()
    engine.close()


if __name__ == '__main__':
    main()
