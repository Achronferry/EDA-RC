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
from models.package.rnn_cluster import RNN_Clustering

class EDA_RC(nn.Module):
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
        super(EDA_RC, self).__init__()
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

        self.decoder = eda_spk_extractor(n_units, n_speakers)
        self.rnn_cluster = RNN_Clustering(n_units, n_speakers)
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
                th=0.5, beam_size=1, chunk_size=500):
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

        src_chunked = torch.split(src, chunk_size, dim=0)
        mask_chunked = torch.split(src_padding_mask, chunk_size, dim=1)
        # output: (B, T, E)
        enc_chunked = [self.transformer_encoder(s, mask=self.src_mask, src_key_padding_mask=m).transpose(0, 1)
                        for s, m in zip(src_chunked, mask_chunked)]
        seq_len_chunked = [(~m).sum(dim=-1) for m in mask_chunked]


        att_chunked, act_prob_chunked = zip(*[self.decoder(enc_c, seqlen_c) 
                                            for enc_c, seqlen_c in zip(enc_chunked, seq_len_chunked)]) 
        output_chunked = [torch.sigmoid(torch.bmm(e, a.transpose(-1, -2))) 
                            for e,a in zip(enc_chunked, att_chunked)]



        if label is not None:
            all_losses = []

            active_prob = torch.stack(act_prob_chunked, dim=1)
            spks_num = label.long().sum(dim=-1, keepdim=False).max(dim=-1, keepdim=False).values # (B, )
            act_label = torch.zeros_like(active_prob)
            for l in range(act_label.shape[0]):
                act_label[l, :spks_num[l]] = 1
            prob_loss = torch.stack([F.binary_cross_entropy(p, l) 
                        for p,l in zip(active_prob, act_label)])
            all_losses.append(prob_loss)

            label_chunked = torch.split(label, chunk_size, dim=1)
            pit_loss = torch.zeros_like(prob_loss)
            for sub_output, sub_label, sub_len in zip(output_chunked, label_chunked, seq_len_chunked):
                sub_spks_num = sub_label.long().sum(dim=-1, keepdim=False).max(dim=-1, keepdim=False).values
                valid_output = nn.utils.rnn.pad_sequence(
                                [o[:, :n].transpose(-1,-2) for o,n in zip(sub_output, sub_spks_num)]
                                , batch_first=True).transpose(-2,-1)
                valid_output = F.pad(valid_output, pad=(0, sub_label.shape[-1]-valid_output.shape[-1]))
                truth = [l[:ilen] for l,ilen in zip(sub_label, sub_len)]
                pred = [o[:ilen] for o,ilen in zip(valid_output, sub_len)]
                pit_loss += batch_pit_loss(pred, truth)[0]
            all_losses.append(pit_loss)


            spk_emb = torch.stack(att_chunked, dim=1) # N, #chunk, max_spk, D
            zip_label = torch.stack([(l.sum(dim=1) != 0) for l in label_chunked])
            cluster_loss = self.rnn_cluster(spk_emb, zip_label)
            all_losses.append(cluster_loss)
            return all_losses
        else:
            output, stat_outputs = [], {}

            spk_emb = torch.stack(att_chunked, dim=1) # N, #chunk, max_spk, D
            ext_prob = torch.stack(act_prob_chunked, dim=1)
            output_shuffled = list(zip(output_chunked)) # N,#chunk, (len(chunk),max_spk)
            print(output_shuffled)



            # for i in output_chunked, att_chunked, act_prob_chunked, seq_len_chunked:
            #     # Inference

            # split by batch
            for e, p, o in zip(spk_emb, ext_prob, output_shuffled): 
                n = [np.where(p_ < th)[0] for p_ in p.cpu().detach().numpy()]
                # print(n)
                n = [i[0] if i.size else None for i in n]
                e = [e_[: n_] for e_, n_ in zip(e,n)]
                best_beam = self.rnn_cluster.decode_beam_search(e, beam_size)[0]
                each_output = []
                # split by chunk
                for each_chunk, each_order in zip(o, best_beam.pred_id):
                    ordered_chunk = torch.zeros_like(each_chunk)
                    ordered_chunk[:,each_order]=each_chunk[:, :len(each_order)]
                    each_output.append(ordered_chunk)
                each_output = torch.cat(each_output, dim=0)
                output.append(each_output)
                 
            output = torch.stack(output, dim=0)
            print(output.shape)
            input()

            return (output > th), stat_outputs



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


