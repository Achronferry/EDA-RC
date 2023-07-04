# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import random
import os,sys
from sklearn import cluster
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_utils.loss import batch_pit_loss
from models.package.spk_extractor import eda_spk_extractor
from models.package.rnn_cluster import RNN_Clusterer

class EDA_RC(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, shuffle_rate=0., dim_feedforward=2048, dropout=0.5, has_pos=False):
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
        self.shuffle_rate = shuffle_rate

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.decoder = eda_spk_extractor(n_units, n_speakers)
        transformer_dec_layers = nn.TransformerDecoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.spk_emb_extractor = nn.TransformerDecoder(transformer_dec_layers, 1)
        self.rnn_cluster = RNN_Clusterer(n_units, n_speakers)
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

    def forward(self, src, seq_lens, label=None, has_mask=False,
                th=0.5, beam_size=1, chunk_size=500, **kargs):
        device = src.device
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

        
        enc_output_t = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_padding_mask)
                # output: (B, T, E)
        enc_output = enc_output_t.transpose(0, 1)

        bsize, max_len, _ = enc_output.shape
        padded_len = chunk_size - (max_len % chunk_size) if max_len % chunk_size != 0 else 0
        
        enc_chunked = torch.split(F.pad(enc_output, (0,0,0,padded_len)), chunk_size, dim=1)
        seq_len_chunked = [(~m).sum(dim=-1) for m in  torch.split(src_padding_mask, chunk_size, dim=1)]
        
        enc_stacked = torch.cat(enc_chunked, dim=0) # n_chunk*B, T, D
        att_stacked, act_prob_stacked = self.decoder(enc_stacked, torch.cat(seq_len_chunked, dim=0))

        act_prob_chunked = torch.split(act_prob_stacked[:, : max_len], bsize, dim=0)
        att_chunked = torch.split(att_stacked, bsize, dim=0)

        output_stacked = torch.sigmoid(torch.bmm(enc_stacked, att_stacked.transpose(-1,-2)))
        output_chunked = torch.split(output_stacked[:, : max_len, :], bsize, dim=0)

        # spk_emb = [self.spk_emb_extractor(i.transpose(0,1), enc_output_t, memory_key_padding_mask=src_padding_mask).transpose(0,1) for i in att_chunked]

        spk_emb = torch.stack(att_chunked, dim=1) # N, #chunk, max_spk, D
        bsize, n_chunk, n_spk, _ = spk_emb.shape
        tgt_mask = torch.ones((n_chunk * n_spk, n_chunk * n_spk), device=spk_emb.device).bool() #seperate chunks
        for i in range(0, n_chunk):
            tgt_mask[i*n_spk : i*n_spk + n_spk, i*n_spk : i*n_spk + n_spk] = False
        spk_emb = self.spk_emb_extractor(spk_emb.reshape(bsize, n_chunk*n_spk, -1).transpose(0,1), enc_output_t, 
                                        memory_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask)
        spk_emb = spk_emb.transpose(0,1).reshape(bsize, n_chunk, n_spk, -1)
          
        if label is not None:
            all_losses = []
            # # global loss
            # global_attractors, global_active_prob = self.decoder(enc_output, seq_lens)
            # global_output = torch.sigmoid(torch.bmm(enc_output, global_attractors.transpose(-1, -2))) # (B, T, C)
            # all_losses += self.calculate_eda_loss(global_output, global_active_prob, seq_lens, label)[0]

            label_chunked = torch.split(label, chunk_size, dim=1)
            pit_order, spks_num = [], []
            pit_loss, prob_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
            for sub_output, sub_prob, sub_label, sub_len in zip(output_chunked, act_prob_chunked, label_chunked, seq_len_chunked):
                [prob_loss_, pit_loss_], spks_num_, pit_order_ = self.calculate_eda_loss(sub_output, sub_prob, sub_len, sub_label)
                spks_num.append(spks_num_)
                pit_order.append(pit_order_)
                prob_loss += prob_loss_
                pit_loss += pit_loss_

            chunk_spk_nums = torch.stack(spks_num, dim=-1)
            ordered_label = torch.tensor(pit_order, device=spk_emb.device, dtype=torch.float).transpose(0,1) # (N, #chunk, max_spk) 

            spk_emb, chunk_spk_nums, ordered_label = self.shuffle_for_clustering(spk_emb, chunk_spk_nums, ordered_label, self.shuffle_rate)
            cluster_loss, clusters = self.rnn_cluster(spk_emb, chunk_spk_nums, ordered_label)

            # if curriculum and spk_id is not None:
            #     spk_emb, chunk_spk_nums, ordered_label, spk_id = spk_emb[:(bsize // 2) * 2], chunk_spk_nums[:(bsize // 2) * 2], ordered_label[:(bsize // 2) * 2], spk_id[:(bsize // 2) * 2] 
            #     cat_spk_emb = spk_emb.reshape(bsize // 2, n_chunk * 2, n_spk, -1)
            #     cat_chunk_spk_nums = chunk_spk_nums.reshape(bsize // 2, -1)
            #     cat_ordered_label = ordered_label.reshape(bsize // 2, n_chunk * 2, -1)
            #     cat_ordered_label[:, n_chunk:] += n_spk
            #     cat_spk_id = spk_id.reshape(bsize // 2, -1)
            #     for b in range(len(cat_ordered_label)):
            #         for i in range(n_spk):
            #             for j in range(n_spk, 2*n_spk):
            #                 if cat_spk_id[b][i] == cat_spk_id[b][j]:
            #                     cat_ordered_label[b] = cat_ordered_label[b].masked_fill(cat_ordered_label[b] == cat_spk_id[b][j], cat_spk_id[b][i])
                
                
            #     shuffle_id = [0 for _ in range(n_chunk)] + [1 for _ in range(n_chunk)]
            #     random.shuffle(shuffle_id)
            #     d = [0,n_chunk]
            #     for i in range(len(shuffle_id)):
            #         d_ = shuffle_id[i]
            #         shuffle_id[i] = d[d_]
            #         d[d_] += 1
            #     cat_spk_emb, cat_chunk_spk_nums, cat_ordered_label = cat_spk_emb[: , shuffle_id], cat_chunk_spk_nums[: , shuffle_id], cat_ordered_label[: , shuffle_id]
            #     curriculum_loss, _ = self.rnn_cluster(cat_spk_emb, cat_chunk_spk_nums, cat_ordered_label)
            #     cluster_loss = 0.9 * cluster_loss + 0.1 * curriculum_loss

            # print(all_losses)

            all_losses.append(prob_loss / n_chunk)
            all_losses.append(pit_loss / n_chunk)
            all_losses.append(cluster_loss)
            return all_losses
        else:
            
            output, stat_outputs = [], {}
            spks_num = []
            unzip_spk_emb = [[j for j in i] for i in spk_emb]
            for chunk_id, (sub_prob, sub_len) in enumerate(zip(act_prob_chunked, seq_len_chunked)):
                spk_num_ = [np.where(p_ < th)[0] for p_ in sub_prob.cpu().detach().numpy()]
                spk_num_ = [i[0] if i.size else sub_prob.shape[-1] for i in spk_num_]
                spks_num.append(spk_num_)

                for idx, n in enumerate(spk_num_):
                    # output_chunked[chunk_id][idx, :, n:] = 0
                    unzip_spk_emb[idx][chunk_id] = unzip_spk_emb[idx][chunk_id][:n ,:]

            oracle = kargs.pop('oracle')
            oracle = torch.split(F.pad(oracle, (0,0,0,padded_len)), chunk_size, dim=1)
            best_orders = []
            for sub_output, sub_prob, sub_label, sub_len in zip(output_chunked, act_prob_chunked, oracle, seq_len_chunked):
                [prob_loss_, pit_loss_], spks_num_, pit_order_ = self.calculate_eda_loss(sub_output, sub_prob, sub_len, sub_label)
                best_orders.append(pit_order_)
            for nb, e in enumerate(unzip_spk_emb):
                
                best_order = [k[nb][:len(l)] for (k,l) in zip(best_orders, e)]


                # # beams = self.rnn_cluster.decode_beam_search(e, beam_size)
                # beams = self.rnn_cluster.decode_refine(e, beam_size)
                # best_order = beams[0].pred_order

                # assert len(e) == len(best_order)
                # spk_num = max(list(map(lambda x: (x.max().cpu().item()+1 if len(x) != 0 else 0), best_order)))
                # print(spk_num)
                
                batch_output = []
                for nc, o in enumerate(best_order):
                    current_chunk = output_chunked[nc][nb, :, :len(o)]
                    ordered_per_chunk = torch.zeros((current_chunk.shape[0], self.n_speakers), device=current_chunk.device)
                    if len(o) > 0:
                        ordered_per_chunk[:, o] = current_chunk
                    batch_output.append(ordered_per_chunk)
                batch_output = torch.cat(batch_output, dim=0)
                
                output.append(batch_output)

            # output = nn.utils.rnn.pad_sequence([i.transpose(-1,-2) for i in output], batch_first=True).transpose(-1,-2)
            output = torch.stack(output, dim=0)
            return (output > th), stat_outputs

    def calculate_eda_loss(self, output, active_prob, seq_lens, label):
            losses = []
            spk_num = (label.sum(dim=1, keepdim=False) > 0).sum(dim=-1, keepdim=False) # (B, )
            act_label = torch.zeros_like(active_prob)
            for l in range(act_label.shape[0]):
                act_label[l, :spk_num[l]] = 1

            prob_loss = torch.stack([F.binary_cross_entropy(p, l) 
                        for p,l in zip(active_prob, act_label)]).mean(dim=0)

            losses.append(prob_loss)

            valid_output = nn.utils.rnn.pad_sequence(
                            [o[:, :n].transpose(-1,-2) for o,n in zip(output, spk_num)]
                            , batch_first=True).transpose(-2,-1)
            valid_output = F.pad(valid_output, pad=(0, label.shape[-1]-valid_output.shape[-1]))
            truth = [l[:ilen] for l,ilen in zip(label, seq_lens)]
            pred = [o[:ilen] for o,ilen in zip(valid_output, seq_lens)]
            pit_loss, _,  pit_order = batch_pit_loss(pred, truth, output_order=True)
            losses.append(pit_loss)

            return losses, spk_num, pit_order

    def shuffle_for_clustering(self, spk_emb, chunk_spk_nums, ordered_label, shuffle_rate=0.25):
        bsize, n_chunk, n_spk, _ = spk_emb.shape
        device = spk_emb.device
        p = torch.rand(bsize) < shuffle_rate
        shuffled_ind = []
        for p_ in p:
            shuffled_ind.append(torch.randperm(n_chunk, device=device) if p_ else torch.arange(n_chunk, device=device))
        shuffled_ind = torch.stack(shuffled_ind, dim=0)

        shuffled_spk_emb = torch.gather(spk_emb, 1, shuffled_ind.unsqueeze(-1).unsqueeze(-1).expand_as(spk_emb))
        shuffled_spk_nums = torch.gather(chunk_spk_nums, 1, shuffled_ind)
        shuffled_labels = torch.gather(ordered_label, 1, shuffled_ind.unsqueeze(-1).expand_as(ordered_label))

        return shuffled_spk_emb, shuffled_spk_nums, shuffled_labels

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


