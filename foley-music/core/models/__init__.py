"""
Code modified from:
    https://github.com/chuangg/Foley-Music/blob/main/core/models/__init__.py
"""
import torch
from torch import nn, Tensor


class ModelFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, device=torch.device('cpu'), wrapper=lambda x: x):
        emb_dim = self.cfg.model.d_model
        duration = self.cfg.dataset.duration
        fps = self.cfg.dataset.fps
        layout = self.cfg.model.layout

        if self.cfg.model.name == 'music_transformer':
            from .music_transformer_dev.music_transformer import music_transformer_dev_baseline
            pose_seq2seq = music_transformer_dev_baseline(
                240 + 3,
                d_model=emb_dim,
                dim_feedforward=emb_dim * 2,
                encoder_max_seq=int(duration * fps),
                decoder_max_seq=self.cfg.model.decoder_max_seq,
                layout=layout,
                num_encoder_layers=self.cfg.model.num_encoder_layers,
                num_decoder_layers=self.cfg.model.num_decoder_layers,
                rpr=self.cfg.model.rpr,
                use_control=False,
                rnn=None,
                layers=self.cfg.model.n_layer
            )
            # if ckpt != 'ckpt':
            #     pass
            #     # TODO load weight for finetune

        else:
            raise Exception

        pose_seq2seq = pose_seq2seq.to(device)
        pose_seq2seq = wrapper(pose_seq2seq)

        return pose_seq2seq
