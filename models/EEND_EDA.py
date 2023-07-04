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
from model_utils.loss import batch_pit_loss
from models.package.spk_extractor import eda_spk_extractor

class EEND_EDA(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=1024, dropout=0.2, has_pos=False, num_predict=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(EEND_EDA, self).__init__()
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

        self.decoder = eda_spk_extractor(n_units, n_speakers, dropout=dropout)
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
            attractors, active_prob = self.decoder(enc_output, seq_lens) #(B,C,E)
            output = torch.sigmoid(torch.bmm(enc_output, attractors.transpose(-1, -2))) # (B, T, C)
            
            all_losses = []
            spks_num = (label.sum(dim=1, keepdim=False) > 0).sum(dim=-1, keepdim=False) # (B, )
            act_label = torch.zeros_like(active_prob)
            for l in range(act_label.shape[0]):
                act_label[l, :spks_num[l]] = 1

            prob_loss = torch.stack([F.binary_cross_entropy(p, l) 
                        for p,l in zip(active_prob, act_label)]).mean(dim=0)

            all_losses.append(prob_loss)

            # output = nn.utils.rnn.pad_sequence(
            #                 [o[:, :n].transpose(-1,-2) for o,n in zip(output, spks_num)]
            #                 , batch_first=True).transpose(-2,-1)
            valid_output = F.pad(output, pad=(0, label.shape[-1]-output.shape[-1]))
            truth = [l[:ilen] for l,ilen in zip(label, seq_lens)]
            pred = [o[:ilen] for o,ilen in zip(valid_output, seq_lens)]
            pit_loss, _ = batch_pit_loss(pred, truth)
            all_losses.append(pit_loss)

            return all_losses
        else:
            enc_chunked = torch.split(enc_output, chunk_size, dim=1)
            seq_len_chunked = [m.sum(dim=1) for m in torch.split(~src_padding_mask, chunk_size, dim=1)]
            output, stat_outputs = [], {}
            for enc, l in zip(enc_chunked, seq_len_chunked):
                attractors, active_prob = self.decoder(enc, l)
                output.append(torch.sigmoid(torch.bmm(enc, attractors.transpose(-1, -2))))
                
                spks_num = [np.where(p_ < th)[0] for p_ in active_prob.cpu().detach().numpy()]
                spks_num = [i[0] if i.size else active_prob.shape[-1] for i in spks_num]

                for idx, n in enumerate(spks_num):
                    output[-1][idx, :, n:] = 0

            output = torch.cat(output, dim=1)

            return output, stat_outputs



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


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    import torch
    model = TransformerLinearModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
    
