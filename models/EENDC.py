# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import os, sys

from numpy.lib import utils
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools, copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.package.spk_extractor import spk_extractor
from models.package.rnn_cluster import RNN_Clustering
from models.package.resegment import vec_sim_seg, num_pred_seg, lstm_seg, lstm_seg_v2, lstm_seg_v2_rd, pool_seg

class EENDC(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, 
                    dim_feedforward=2048, dropout=0.5, has_pos=False, num_predict=False
                    , mode='all'):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Max number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(EENDC, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        assert mode in ['all', 'seg', 'clst']
        self.mode = mode

        if num_predict:
            self.num_predictor = nn.Linear(n_units, n_speakers + 1)
        else:
            self.num_predictor = None

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)


        self.segmenter = pool_seg(n_units)
        # self.segmenter = num_pred_seg(n_units)

        self.decoder = spk_extractor(n_units, n_speakers, dropout=dropout)
        self.rnn_cluster = RNN_Clustering(n_units, n_speakers, dropout=dropout)


        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_lens, label=None, change_points=None, has_mask=False,
                th=0.5, beam_size=1, chunk_size=2000):
        device = src.device
        max_len = src.shape[1]

        if has_mask:
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
        # output: (B, T, C)
        # if self.num_predictor is not None:
        #     num_probs = self.num_predictor(enc_output)
        # else:
        #     num_probs = None



        if label is not None:
            all_losses = []

            # Train Re-Segmenter
            if self.mode != 'clst':
               reseg_loss = self.segmenter(enc_output, seq_lens, label=label)
               all_losses.append(reseg_loss)
            
            if self.mode != 'seg':
                change_points = F.pad((torch.abs(label[:, 1:] - label[:, :-1]).sum(dim=-1) != 0), pad=(1, 0))
                change_points = [torch.cat(
                                        [torch.tensor([0], device=device),
                                        torch.nonzero(i[:n]).squeeze(-1),
                                        torch.tensor([n], device=device)], dim=0)
                                     for i,n in zip(change_points, seq_lens)]

                local_lens = [(i[1:] - i[:-1]).cpu().tolist() for i in change_points]
 
                zip_enc_output = [e[: len_] for e, len_ in zip(enc_output, seq_lens)]
                zip_label = [l[: len_] for l, len_ in zip(label, seq_lens)]

                zip_enc_output = [torch.split(e, ck_sz) for e, ck_sz in zip(zip_enc_output, local_lens)]
                zip_enc_output = [torch.stack([j.mean(dim=0) for j in i], dim=0) for i in zip_enc_output]
                
                zip_label = [torch.split(l, ck_sz) for l, ck_sz in zip(zip_label, local_lens)]
                zip_label = [torch.stack([j[0] for j in i], dim=0) for i in zip_label]
                
                zip_seq_lens = [len(i) for i in zip_enc_output]
                # print(f"len before:{seq_lens}\nlen after{zip_seq_lens}")
                zip_enc_output = nn.utils.rnn.pad_sequence(zip_enc_output, batch_first=True)
                zip_label = nn.utils.rnn.pad_sequence(zip_label, batch_first=True)


                ctmpr_spks = torch.sum(zip_label.long(), dim=-1, keepdim=False)

                spk_emb, ext_prob = self.decoder(zip_enc_output)

                ## Extractor loss
                prob_loss = []
                for seq_prob, truth_num, seq_len in zip(ext_prob, ctmpr_spks, zip_seq_lens):
                    seq_prob, truth_num = seq_prob[:seq_len], truth_num[:seq_len]

                    p_label = torch.zeros_like(seq_prob)
                    for l in range(p_label.shape[0]):
                        p_label[l, :truth_num[l]] = 1
                    # print(torch.cat([p_label,n.unsqueeze(1)], dim=1))
                    prob_loss.append(F.binary_cross_entropy(
                                        seq_prob, p_label))
                prob_loss = torch.stack(prob_loss)
                all_losses.append(prob_loss)
                
                ## cluster loss
                cluster_loss = self.rnn_cluster(spk_emb, zip_label)
                all_losses.append(cluster_loss)


            return all_losses
        else:
            stat_outputs = {}

            # Re-Segmenter
            if change_points is None:
                change_points_ = self.segmenter(enc_output, seq_lens, th=th)
                stat_outputs["change_points"] = change_points_ # B, T
                # stat_outputs["num_pred"] = num_pred_
                change_points = change_points_

            change_points = [torch.cat(
                                [torch.tensor([0], device=device),
                                torch.nonzero(i[:n]).squeeze(-1),
                                torch.tensor([n], device=device)], dim=0)
                            for i,n in zip(change_points, seq_lens)]
            local_lens = [(i[1:] - i[:-1]) for i in change_points]
 
            zip_enc_output = [e[: len_] for e, len_ in zip(enc_output, seq_lens)]

            zip_enc_output = [torch.split(e, ck_sz.cpu().tolist()) for e, ck_sz in zip(zip_enc_output, local_lens)]
            zip_enc_output = [torch.stack([j.mean(dim=0) for j in i], dim=0) for i in zip_enc_output]
                
                
            zip_seq_lens = [len(i) for i in zip_enc_output]
            zip_enc_output = nn.utils.rnn.pad_sequence(zip_enc_output, batch_first=True)

            spk_emb, ext_prob = self.decoder(zip_enc_output)
            # Inference
            res = []
            for e, p, l in zip(spk_emb, ext_prob, zip_seq_lens): 
                e, p = e[:l], p[:l]
                n = [np.where(p_ < th)[0] for p_ in p.cpu().detach().numpy()]
                # print(n)
                n = [i[0] if i.size else None for i in n]
                e = [e_[: n_] for e_, n_ in zip(e,n)]
                best_beam = self.rnn_cluster.decode_beam_search(e, beam_size)[0]
                res.append(nn.utils.rnn.pad_sequence(best_beam.pred, batch_first=True))
            
            res = [torch.repeat_interleave(z, l, dim=0) for z, l in zip(res, local_lens)]
            res = self._align_for_parallel(res, self.n_speakers, max_len)
            return res, stat_outputs



        # spk_emb, ext_prob = self.decoder(enc_output)
        # # return (spk_emb, ext_prob), num_probs
        # if label is not None:
        #     # Training
            
        #     label_delay = 0
        #     prob_loss = []
        #     assert self.n_speakers == label.shape[-1]

        #     ctmpr_spks = torch.sum(label.long(), dim=-1, keepdim=False)
        #     ## speaker prob loss
        #     for seq_emb, seq_prob, truth_num, seq_len in zip(spk_emb, ext_prob, ctmpr_spks, seq_lens):
        #         seq_emb, seq_prob, truth_num = seq_emb[:seq_len], seq_prob[:seq_len], truth_num[:seq_len]

        #         p_label = torch.zeros_like(seq_prob)
        #         for l in range(p_label.shape[0]):
        #             p_label[l, :truth_num[l]] = 1
        #         # print(torch.cat([p_label,n.unsqueeze(1)], dim=1))
        #         prob_loss.append(F.binary_cross_entropy(
        #                             seq_prob[label_delay:, ...],
        #                             p_label[:len(p_label) - label_delay, ...]))
        #     prob_loss = torch.stack(prob_loss)

        #     ## cluster loss
        #     cluster_loss = self.rnn_cluster(spk_emb, label)

        #     return [prob_loss, cluster_loss] # (bsize, 1)
        # else:
        #     # Inference
        #     res = []
        #     for e, p, l in zip(spk_emb, ext_prob, seq_lens): 
        #         e, p = e[:l], p[:l]
        #         n = [np.where(p_ < th)[0] for p_ in p.cpu().detach().numpy()]
        #         # print(n)
        #         n = [i[0] if i.size else None for i in n]
        #         e = [e_[: n_] for e_, n_ in zip(e,n)]
        #         best_beam = self.rnn_cluster.decode_beam_search(e, beam_size)[0]
        #         res.append(nn.utils.rnn.pad_sequence(best_beam.pred, batch_first=True))
        #     res = self._align_for_parallel(res, self.n_speakers, max_len)
        #     return res, stat_outputs
    
    def _align_for_parallel(self, outputs, n_spk, max_len):
        aligned_outputs = []
        for i in outputs:
            aligned_i = torch.zeros((max_len, n_spk), device = i.device)
            aligned_i[: i.shape[0], :i.shape[1]] = i
            aligned_outputs.append(aligned_i)
        aligned_outputs = torch.stack(aligned_outputs, dim = 0)
        return aligned_outputs





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
    model = EENDC(3, 40, 4, 512, 2)
    inp = torch.randn(2, 50, 40)
    seq_len = torch.Tensor([30,45]).long()
    label = None #(torch.rand((4,500,3)) > 0.5).long()
    res = model(inp, seq_len, label=label, beam_size=3)
    print(res.shape)

    # print("Model attention:", model.get_attention_weight(input).size())
    # print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
