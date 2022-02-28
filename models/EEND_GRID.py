# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import imp
import numpy as np
import math
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from model_utils.loss import batch_pit_loss
from itertools import permutations

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
        # if self.has_pos:
        #     self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        self.decoder = GridTransformer_dec(n_units, n_heads, dim_feedforward, dropout, n_speakers)
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
        self.decoder.init_weights()

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
            # pred = self.decoder.decode(enc_output, src_padding_mask, th=th)
            # _, teacher_label = batch_pit_loss(pred, label)
            # teacher_label = torch.stack(teacher_label, dim=0)
            # teacher_label = F.pad(teacher_label,(0,1))

            # z = self.decoder(enc_output, src_padding_mask, teacher_label)
            # y_tf = torch.cat([l[:ilen] for l,ilen in zip(teacher_label, seq_lens)], dim=0)
            # z_tf = torch.cat([o[:ilen] for o,ilen in zip(z, seq_lens)], dim=0)
            # pit_loss = F.binary_cross_entropy(z_tf, y_tf)

            # return [pit_loss]

            losses = []
            for label_perm in permutations(label.permute(2,0,1)):
                teacher_label = torch.stack(label_perm, dim=-1)
                teacher_label = F.pad(teacher_label,(0,1))

                z = self.decoder(enc_output, src_padding_mask, teacher_label)
                y_tf = torch.cat([l[:ilen] for l,ilen in zip(teacher_label, seq_lens)], dim=0)
                z_tf = torch.cat([o[:ilen] for o,ilen in zip(z, seq_lens)], dim=0)
                tmp_loss = F.binary_cross_entropy(z_tf, y_tf)
                losses.append(tmp_loss)     
            return [min(losses)]
        else:
            stat_outputs = {}
            output = self.decoder.decode(enc_output, src_padding_mask, th=th)
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


class GridTransformer_dec(nn.Module):
    def __init__(self, n_units, n_heads, dim_feedforward=2048, dropout=0.2, n_speakers=2):
        super(GridTransformer_dec, self).__init__()
        self.n_speakers = n_speakers
        self.n_heads = n_heads
        self.label_proj = nn.Linear(1, n_units)

        self.time_seq_dec = MultiheadAttention(embed_dim=n_units, kdim=2*n_units, num_heads=n_heads, dropout=dropout)
        self.spk_chain_dec = TransformerEncoderLayer(2*n_units, n_heads, dim_feedforward, dropout)
        self.frame_proj = nn.Linear(n_units + n_units, n_units)
        self.pred = nn.Bilinear(n_units, n_units, 1)
        self.dropout = nn.Dropout(dropout)
        self.sos = nn.Parameter(torch.zeros((1,n_units), dtype=torch.float).uniform_(-0.1, 0.1))

    def init_weights(self):
        pass
    
    def forward(self, enc_output, enc_padding_mask, label):
        '''
        enc_output: (B, T, D)
        label: (B, T, C)
        enc_padding_mask: (B, T) 0-reserve 1-mask

        the order of label is pre-defined.
        '''
        b_size, max_len, spk_num = label.shape
        active_probs = []
        enc_out_iter = enc_output
        attn_mask = torch.ones((max_len, max_len), device=enc_output.device).bool()
        attn_mask = ~ torch.logical_xor(torch.triu(attn_mask, diagonal=1), torch.triu(attn_mask, diagonal=-3)) #B,L,S

        # For bugs in MultiheadAttention
        attn_mask = torch.logical_or(enc_padding_mask.unsqueeze(1), attn_mask)
        attn_mask = torch.masked_fill(attn_mask, enc_padding_mask.unsqueeze(-1), False).repeat_interleave(self.n_heads, dim=0)

        # attn_mask = torch.triu(attn_mask, diagonal=1)

        for spk_turn in range(spk_num):
            y_prev = self.label_proj(F.pad(self.dropout(label[:, : -1, spk_turn : spk_turn + 1]), (0,0,1,0)))
            history_enc_out = torch.cat([self.sos.unsqueeze(dim=0).expand(b_size,1,-1), enc_out_iter[:, :-1,:]], dim=1)
            history_states = torch.cat([history_enc_out, y_prev], dim=-1)
            seq_dec_out, _ = self.time_seq_dec(enc_out_iter.transpose(0,1), history_states.transpose(0,1), history_enc_out.transpose(0,1), attn_mask=attn_mask)
            seq_dec_out = seq_dec_out.transpose(0,1).contiguous()
            spk_prob = torch.sigmoid(self.pred(enc_out_iter, seq_dec_out))
            active_probs.append(spk_prob)

            #======
            last_state = torch.cat([enc_out_iter, self.label_proj(label[ :, : , spk_turn : spk_turn + 1])], dim=-1)
            enc_out_iter = self.spk_chain_dec(last_state.transpose(0,1), src_key_padding_mask=enc_padding_mask).transpose(0,1)
            enc_out_iter = self.frame_proj(enc_out_iter)
        return torch.cat(active_probs, dim=-1)

    @torch.no_grad()
    def decode(self, enc_output, enc_padding_mask, th=0.5):
        device = enc_output.device
        b_size, max_len, _ = enc_output.shape
        active_probs = torch.zeros((b_size,max_len,self.n_speakers), device=device)
        enc_out_iter = enc_output
        for spk_turn in range(self.n_speakers):
            current_active_prob = []
            y_prev = self.label_proj(torch.zeros((b_size,1,1), device=device))
            for t in range(max_len):
                history_enc_out = torch.cat([self.sos.unsqueeze(dim=0).expand(b_size,1,-1), enc_out_iter[:, :t,:]], dim=1)
                history_states = torch.cat([history_enc_out, y_prev], dim=-1)
                if history_states.shape[1] > 4:
                    history_states = history_states[:, -4: ,:]
                    history_enc_out = history_enc_out[:, -4: ,:]
                seq_dec_out, att_weight = self.time_seq_dec(enc_out_iter[:,t:t+1,:].transpose(0,1), history_states.transpose(0,1), history_enc_out.transpose(0,1))
                seq_dec_out = seq_dec_out.transpose(0,1)
                spk_prob = torch.sigmoid(self.pred(enc_out_iter[:,t:t+1,:], seq_dec_out))
                current_active_prob.append(spk_prob)

                y_next = self.label_proj((spk_prob > th).float())
                y_prev = torch.cat([y_prev, y_next], dim=1)

            active_probs[:,:,spk_turn:spk_turn+1] = torch.cat(current_active_prob, dim=1)
            #TODO only suit for a fixed number of speaker!!!
            #======
            last_state = torch.cat([enc_out_iter, y_prev[:,1:,:]], dim=-1)
            enc_out_iter = self.spk_chain_dec(last_state.transpose(0,1), src_key_padding_mask=enc_padding_mask).transpose(0,1)
            enc_out_iter = self.frame_proj(enc_out_iter)
            #TODO
        return active_probs



if __name__ == "__main__":
    import torch
    model = EEND_GRID(n_speakers=3, in_size=40, n_heads=4, n_units=256, n_layers=2)
    input = torch.randn(8, 100, 40)
    seq_lens = torch.Tensor([100,25,10,50,1,8,6,6]).long()
    label = (torch.randn((8,100, 3)) > 0.5).float()
    print("Model output:", model(input, seq_lens,label))

