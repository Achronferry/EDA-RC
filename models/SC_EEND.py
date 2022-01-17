# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import os,sys

from torch._C import device
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_utils.loss import batch_pit_loss, pit_loss

class SC_EEND(nn.Module):
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
        super(SC_EEND, self).__init__()
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
        self.decoder = SpeakerChain_dec(n_units, self.n_speakers, dropout=dropout)
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
            pit_loss = self.decoder(enc_output, seq_lens, label)
            return [pit_loss]
        else:
            prob = self.decoder.dec_greedy(enc_output)
            pred = (prob > th)

            return pred, {}





class SpeakerChain_dec(nn.Module):
    def __init__(self, n_units, n_speakers, rnn_cell='LSTM', dropout=0.2):
        super(SpeakerChain_dec, self).__init__()
        self.n_speakers = n_speakers
        self.label_proj = nn.Linear(1, n_units)
        self.chain_rnn = nn.LSTM(2*n_units, n_units, batch_first=True)
        self.rnn_init_h = nn.Parameter(torch.zeros((1, n_units)))
        self.pred = nn.Sequential(nn.Linear(n_units, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, enc_output, seq_len, label):
        '''
        frame_emb: (B, T, D)

        '''

        device = enc_output.device
        
        # ===stage 1===
        pred = self.dec_greedy(enc_output)
        _, teacher_label = batch_pit_loss(pred, label)
        teacher_label = torch.stack(teacher_label, dim=0)
        teacher_label = F.pad(teacher_label,(0,1))

        # ===stage 2===
        batch_size, max_len, _ = enc_output.shape

        y_prev = torch.zeros((batch_size, max_len,1), device=device, dtype=torch.float)
        h = self.rnn_init_h.repeat(batch_size * max_len, 1).unsqueeze(0)
        c = torch.zeros_like(h)

        z = []
        for spk_turn in range(self.n_speakers + 1): # empty
            E_spk = torch.cat([enc_output, self.label_proj(y_prev)], dim=-1)
            E_in = E_spk.view(batch_size * max_len, 1, -1)
            h_out, (h,c) = self.chain_rnn(self.dropout(E_in), (h,c))
            h_out = h_out.view(batch_size, max_len, -1)
            z_s = self.pred(h_out)
            z.append(z_s)
            y_prev = teacher_label[:, :, spk_turn : spk_turn + 1]
        z = torch.cat(z, dim=-1)

        y_tf = [l[:ilen, :-1] for l,ilen in zip(teacher_label, seq_len)]
        z_tf = [o[:ilen, :-1] for o,ilen in zip(z, seq_len)]
        pit_loss, _ = batch_pit_loss(z_tf, y_tf)
        
        y_mpt = torch.cat([l[:ilen, -1] for l,ilen in zip(teacher_label, seq_len)], dim=0)
        z_mpt = torch.cat([o[:ilen, -1] for o,ilen in zip(z, seq_len)], dim=0)
        empty_loss = F.binary_cross_entropy(z_mpt, y_mpt)
        
        return pit_loss+empty_loss



    def dec_greedy(self, enc_output, th=0.5):
        device = enc_output.device
        batch_size, max_len, _ = enc_output.shape
        y_prev = torch.zeros((batch_size, max_len,1), device=device, dtype=torch.float)
        h = self.rnn_init_h.repeat(batch_size * max_len, 1).unsqueeze(0)
        c = torch.zeros_like(h)

        z = []
        for spk_turn in range(self.n_speakers):
            E_spk = torch.cat([enc_output, self.label_proj(y_prev)], dim=-1)
            E_in = E_spk.view(batch_size * max_len, 1, -1)
            h_out, (h,c) = self.chain_rnn(E_in, (h,c))
            h_out = h_out.view(batch_size, max_len, -1)
            z_s = self.pred(h_out)
            z.append(z_s)
            y_prev = (z_s > 0.5).float()
        
        return torch.cat(z, dim=-1)
