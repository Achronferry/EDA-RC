# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from models.package.relation_aware_attn import TransformerEncoderLayer_RP

class TransformerLinearModel_RP(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5,
                 has_pos=False, num_predict=False, max_relative_position=2, gap=100):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerLinearModel_RP, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        if num_predict:
            self.num_predictor = nn.Linear(n_units, n_speakers + 1)
        else:
            self.num_predictor = None
        # if self.has_pos:
        #     self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer_RP(n_units, n_heads, max_relative_position, gap, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.decoder = nn.Linear(n_units, n_speakers)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # for name, param in self.decoder.named_parameters():
        #     if "weight_hh" in name:
        #         nn.init.orthogonal_(param.data)
        #     elif "weight_ih" in name:
        #         nn.init.xavier_uniform_(param.data)
        #     elif "bias" in name:
        #         nn.init.zeros_(param.data)
        #         param.data[self.n_units:2 * self.n_units] = 1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.dec_init_tag.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_lens, label=None, has_mask=False):
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
        #src = src.transpose(0, 1)

        # output: (B, T, E)
        enc_output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_padding_mask)
        # output: (B, T, E)
        # enc_output = enc_output.transpose(0, 1)
        if self.num_predictor is not None:
            num_probs = self.num_predictor(enc_output)
        else:
            num_probs = None

        # Decoder
        output = self.decoder(enc_output)

        # output: (B, T, C)
        output = torch.sigmoid(output)

        return output, num_probs


    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []

        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])

        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)



if __name__ == "__main__":
    import torch

    model = TransformerLSTMattnModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
