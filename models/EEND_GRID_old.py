# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.package.grid_decoder import frameRNN_dec

class EEND_GRID(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(EEND_GRID, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos



        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = frameRNN_dec(n_units, self.n_speakers, dropout=dropout)
        ###
#        from models.package.resegment import num_pred_seg
#        self.segmenter = num_pred_seg(n_units)
        ###
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, seq_lens, label=None,  change_points=None, has_mask=False,
                th=0.5, beam_size=1, chunk_size=2000):
        device = src.device
        max_len = src.shape[1]

        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src_padding_mask = torch.zeros(src.shape[:-1], device=src.device).bool()  # (B*T)
        for idx, l in enumerate(seq_lens):
            src_padding_mask[idx, l:] = 1

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        if label is None:
            src_chunked = torch.split(src, chunk_size, dim=0)
            mask_chunked = torch.split(src_padding_mask, chunk_size, dim=1)
            enc_output = [self.transformer_encoder(s, mask=self.src_mask, src_key_padding_mask=m) 
                            for s, m in zip(src_chunked, mask_chunked)]
            enc_output = torch.cat(enc_output, dim=0)
        else:
            enc_output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_padding_mask)
            
        # output: (B, T, E)
        enc_output = enc_output.transpose(0, 1)


        if label is not None:
            all_losses = self.decoder(enc_output, seq_lens, label)
            # all_losses = [spk_loss, active_loss]
            return all_losses
        else:
            pred = []
            for each, each_len in zip(enc_output, seq_lens):
                each_pred = self.decoder.dec_each_offline(each[:each_len], th=th)
                pred.append(each_pred)

            return self._align_for_parallel(pred, self.n_speakers, max_len), {}


    def _align_for_parallel(self, outputs, n_spk, max_len):
        aligned_outputs = []
        for i in outputs:
            aligned_i = torch.zeros((max_len, n_spk), device = i.device)
            aligned_i[: i.shape[0], :i.shape[1]] = i
            aligned_outputs.append(aligned_i)
        aligned_outputs = torch.stack(aligned_outputs, dim = 0)
        return aligned_outputs
