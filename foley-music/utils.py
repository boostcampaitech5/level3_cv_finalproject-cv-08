import os
import random
from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt


class AverageMeter:
    def __init__(self, name, higher_is_better=True):
        self.reset()
        self.name = name
        self.higher_is_better = higher_is_better

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def compute(self):
        return self.avg
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class MetricTrakcer:
    def __init__(self, start_epoch=1, mode='train'):
        self.metrics = defaultdict(int)
        self.best = defaultdict(list)
        self.start_epoch = start_epoch
        self.n_epochs = 0
        self.mode = mode

    def reset(self):
        for key in self.metrics.keys():
            self.metrics[key].reset()

        for key in self.best.keys():
            self.best[key].reset()

        self.n_epochs = 0
    
    def update(self, result, epoch):
        for fn, obj in result.items():
            res = obj.compute()
            self.metrics[f'{self.mode}_{fn.lower()}'] = res

            better = True
            if epoch > self.start_epoch:
                if ((fn == 'Loss' or not result[fn].higher_is_better) and
                    self.best[f'{self.mode}_{fn.lower()}'][1] <= res):
                    better = False
                elif (result[fn].higher_is_better and 
                      self.best[f'{self.mode}_{fn.lower()}'][1] >= res):
                    better = False

            if better:
                self.best[f'{self.mode}_{fn.lower()}'] = [epoch, res]
        
        self.n_epochs += 1


class CurvePlotter:
    def __init__(self, title, xlabel, ylabel, i=1):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fignum = i
        self.fig = plt.figure(num=self.fignum, figsize=(7,5))
        self.values = defaultdict(list)

    def update_values(self, label, val):
        self.values[label].append(val)

    def plot_learning_curve(self, label):
        plt.figure(self.fignum)
        plt.plot(np.arange(1, len(self.values[label])+1),
                 self.values[label],
                 label=label,
                 marker='o',
                 markersize=2,
                 )

    def save_fig(self, save_path):
        plt.figure(self.fignum)
        plt.title(self.title)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(alpha=0.2)
        plt.savefig(save_path)
        plt.close(self.fignum)


def init_curves(exp, metric):
    curves = {
        'loss_curve': CurvePlotter(title=f'{exp}', xlabel='Epoch', 
                                   ylabel='Loss', i=1),
        f'{metric.lower()}_curve': CurvePlotter(title=f'{exp}', xlabel='Epoch', 
                                                ylabel=metric, i=1),
    }
    return curves


def plot_curves(curves, exp, train_metrics, valid_metrics, metric):
    curves['loss_curve'].update_values('train_loss', train_metrics['train_loss'])
    curves['loss_curve'].update_values('valid_loss', valid_metrics['valid_loss'])
    curves['loss_curve'].plot_learning_curve(label='train_loss')
    curves['loss_curve'].plot_learning_curve(label='valid_loss')
    curves['loss_curve'].save_fig(f'./results/{exp}/loss_curve.png')

    metric = metric.lower()
    curves[f'{metric}_curve'].update_values(f'train_{metric}', train_metrics[f'train_{metric}'])
    curves[f'{metric}_curve'].update_values(f'valid_{metric}', valid_metrics[f'valid_{metric}'])
    curves[f'{metric}_curve'].plot_learning_curve(label=f'train_{metric}')
    curves[f'{metric}_curve'].plot_learning_curve(label=f'valid_{metric}')
    curves[f'{metric}_curve'].save_fig(f'./results/{exp}/{metric}_curve.png')


def write_msg_header(metric):
    epoch_msg_header = (
        f"{'Epoch':^8}"
        f"{'Train Loss':^16}"
        f"{'Valid Loss':^16}"
        f"{f'Train {metric}':^20}"
        f"{f'Valid {metric}':^20}"
    )

    # logging.info(epoch_msg_header)
    epoch_msg_header = '\n' + '=' * 80 + '\n' + epoch_msg_header + '\n' + '=' * 80
    print(epoch_msg_header)


def write_msg(epoch, train_metrics, valid_metrics, metric, end='\n'):
    epoch_msg = (
        f"""{f'{epoch:03d}':^8}"""
        f"""{f"{train_metrics['train_loss']:.6f}":^16}"""
        f"""{f"{valid_metrics['valid_loss']:.6f}":^16}"""
        f"""{f"{train_metrics[f'train_{metric.lower()}']:.4f}":^20}"""
        f"""{f"{valid_metrics[f'valid_{metric.lower()}']:.4f}":^20}"""
    )

    print(epoch_msg, end=end)
    # if end == '':
    #     epoch_msg += '[!n]'
    # logging.info(epoch_msg)


def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False