# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

from itertools import combinations
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
from models.package.rnn_cluster import RNN_Clusterer
from models.package.cop_kmeans import cop_kmeans

class EDA_UC(nn.Module):
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
        super(EDA_UC, self).__init__()
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
        transformer_dec_layers = nn.TransformerDecoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.spk_emb_extractor = nn.TransformerDecoder(transformer_dec_layers, 1)


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
            # global loss
            global_attractors, global_active_prob = self.decoder(enc_output, seq_lens)
            global_output = torch.sigmoid(torch.bmm(enc_output, global_attractors.transpose(-1, -2))) # (B, T, C)
            all_losses += self.calculate_eda_loss(global_output, global_active_prob, seq_lens, label)[0]

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

            pair_loss = torch.tensor(0., device=device)
            for e,n,l in zip(spk_emb, chunk_spk_nums, ordered_label):
                stack_e = torch.cat([e_[:n_] for e_,n_ in zip(e,n)], dim=0)
                stack_l = torch.cat([l_[:n_] for l_,n_ in zip(l,n)], dim=0)
                if len(stack_l) == 0: 
                    continue
                count_attr = torch.stack([(stack_l == i).sum() for i in stack_l], dim=0).float().unsqueeze(-1)
                count_attr = torch.mm(count_attr, count_attr.transpose(0,1))

                sim_matr = calculate_sim_matr(stack_e, stack_e)
                label_matr = (stack_l.unsqueeze(0).expand_as(sim_matr) == stack_l.unsqueeze(-1).expand_as(sim_matr)).float()

                loss_matr = label_matr * (1 - sim_matr) + (1 - label_matr) * sim_matr.masked_fill(sim_matr < 0, 0.)
                loss_matr = loss_matr / count_attr
                pair_loss += loss_matr.sum() / (n_spk ** 2)  

            all_losses.append(prob_loss / n_chunk)
            all_losses.append(pit_loss / n_chunk)
            all_losses.append(pair_loss / bsize)
            # print(all_losses)
            return all_losses
        else:
            # Switch
            #global_out, global_stat = self.global_inference(enc_output, seq_lens)
            #if global_stat["spk_num"] < 3:
            #    return global_out, global_stat

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


            for nb, e in enumerate(unzip_spk_emb):

                stack_e = torch.cat(e, dim=0)
                sim_matr = calculate_sim_matr(stack_e, stack_e)
                R = sim_matr.masked_fill(sim_matr < 0, 0)
                S_local = [e_.shape[0] for e_ in e]
                st, constraints = 0, []
                for n_ in S_local:
                    R[st : st+n_, st : st+n_] = torch.diag(torch.ones((n_), device=device,dtype=torch.float))
                    if n_ >= 2:
                        constraints += [i for i in combinations(range(st, st+n_), 2)]
                    st += n_
                eigen_e = torch.linalg.eigvals(R).real.cpu().tolist()
                sorted(eigen_e, reverse=True)
                min_gap, n_cluster = float('inf'), 0
                for i in range(len(eigen_e) - 1):
                    if eigen_e[i] < 1:
                        break
                    elif eigen_e[i+1] / eigen_e[i] < min_gap:
                        min_gap =  eigen_e[i+1] / eigen_e[i]
                        n_cluster = i + 1
                
                n_cluster = min(max(n_cluster, max(S_local)), self.n_speakers)
                clusters, _ = cop_kmeans(stack_e.cpu().numpy(), k=n_cluster, cl=constraints)
                if clusters is None:
                    clusters, _ = cop_kmeans(stack_e.cpu().numpy(), k=n_cluster) # TODO use CLC-kmeans instead
                best_order = torch.tensor(clusters, device=device).split(S_local, dim=0)

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
            return output, stat_outputs

    def calculate_eda_loss(self, output, active_prob, seq_lens, label):
            global_losses = []
            spks_num = (label.sum(dim=1, keepdim=False) > 0).sum(dim=-1, keepdim=False) # (B, )
            act_label = torch.zeros_like(active_prob)
            for l in range(act_label.shape[0]):
                act_label[l, :spks_num[l]] = 1

            prob_loss = torch.stack([F.binary_cross_entropy(p, l) 
                        for p,l in zip(active_prob, act_label)]).mean(dim=0)

            global_losses.append(prob_loss)

            valid_output = nn.utils.rnn.pad_sequence(
                            [o[:, :n].transpose(-1,-2) for o,n in zip(output, spks_num)]
                            , batch_first=True).transpose(-2,-1)
            valid_output = F.pad(valid_output, pad=(0, label.shape[-1]-valid_output.shape[-1]))
            truth = [l[:ilen] for l,ilen in zip(label, seq_lens)]
            pred = [o[:ilen] for o,ilen in zip(valid_output, seq_lens)]
            pit_loss, _,  pit_order = batch_pit_loss(pred, truth, output_order=True)
            global_losses.append(pit_loss)

            return global_losses, spks_num, pit_order


    def global_inference(self, enc_output, seq_len, th=0.5):
        enc_chunked = [enc_output]
        seq_len_chunked = [seq_len]
        output, stat_outputs = [], {}
        for i, l in zip(enc_chunked, seq_len_chunked):
            attractors, active_prob = self.decoder(i, l)
            output.append(torch.sigmoid(torch.bmm(i, attractors.transpose(-1, -2))))
                
            spks_num = [np.where(p_ < th)[0] for p_ in active_prob.cpu().detach().numpy()]
            spks_num = [i[0] if i.size else active_prob.shape[-1] for i in spks_num]
            # Here has a bug, only suit for bsize=1 !!!
            stat_outputs["spk_num"] = max(max(spks_num), stat_outputs.get("spk_num", 0))

            for idx, n in enumerate(spks_num):
                output[-1][idx, :, n:] = 0

        output = torch.cat(output, dim=1)

        return (output > th), stat_outputs


def calculate_sim_matr(att1, att2):
    '''
    att1: #local1, D
    att2: #local2, D
    out: #local1, #local2
    '''
    sim_matr = torch.mm(att1, att2.transpose(-1,-2))
    n1 = torch.norm(att1, p=2, dim=-1, keepdim=True)
    n2 = torch.norm(att2, p=2, dim=-1, keepdim=True)
    n = torch.mm(n1, n2.transpose(-1,-2)) + 1e-9
    return sim_matr / n

class PsitionalEncoding(nn.Module):
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


