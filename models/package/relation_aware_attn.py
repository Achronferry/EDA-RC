# coding=utf8
from typing import Optional, Any

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position, gap=1):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.gap=gap
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q, device=self.embeddings_table.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        if self.gap == 1:
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        else:
            distance_mat_clipped = torch.zeros_like(distance_mat)
            for i in range(1, self.max_relative_position + 1):
                distance_mat_clipped_tmp = torch.clamp(distance_mat, -i * self.gap, i * self.gap)
                distance_mat_clipped += distance_mat_clipped_tmp.masked_fill(torch.abs(distance_mat_clipped_tmp) != i * self.gap, 0) // (i * self.gap)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        # print(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

    def extra_repr(self):
        return f"num_units={self.num_units},relative_position={list(range(-self.max_relative_position,self.max_relative_position+1))},relation_gap={self.gap}"

class MultiheadAttention_RP(nn.Module):
    def __init__(self, hidden_size, n_heads, max_relative_position, gap, dropout):
        super(MultiheadAttention_RP, self).__init__()

        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.max_relative_position = max_relative_position

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.relative_position_k = RelativePosition(self.hidden_size, self.max_relative_position, gap)
        self.relative_position_v = RelativePosition(self.hidden_size, self.max_relative_position, gap)
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size]))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        nn.init.constant_(self.query.bias, 0.)
        nn.init.constant_(self.key.bias, 0.)
        nn.init.constant_(self.value.bias, 0.)



    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        '''
        :param query: batch * query_len * embed_size
        :param key: batch * key_len * embed_size
        :param value: batch * key_len * embed_size
        :param attn_mask: NotImplemented yet!
        :param key_padding_mask: batch * key_len  1 masked, 0 reserved
        :return: context_layer: batch * query_len * hidden_size
        :return: attention_probs: batch * query_len * key_len
        '''
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)


        r_q1 = query.view(batch_size, -1, self.n_heads, self.hidden_size).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.hidden_size).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.hidden_size)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn_score = (attn1 + attn2) / self.scale.to(device=attn1.device)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand_as(
                attn_score).contiguous()  # batch * num_heads * query * key
            attn_score = attn_score.masked_fill(attn_mask, -float('inf'))

        # Normalize the attention scores to probabilities.
        attn_probs = F.softmax(attn_score, dim=-1)  # batch * num_heads * query * key
        attn_probs = self.dropout(attn_probs)

        #attn_probs = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.hidden_size).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn_probs, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn_probs.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.hidden_size)

        context = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_heads * self.hidden_size)

        # batch * query * num_heads X hidden
        context = self.dense(context)
        return context, attn_probs


class TransformerEncoderLayer_RP(nn.TransformerEncoderLayer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, max_relative_position, gap=100, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super(TransformerEncoderLayer_RP, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = MultiheadAttention_RP(d_model, nhead, max_relative_position, gap=gap, dropout=dropout)
        self.batch_first = batch_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if not self.batch_first:
            src = src.transpose(0, 1) #(T, B, E) -> (B, T, E)
            
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if not self.batch_first:
            src = src.transpose(0, 1) #(B, T, E) -> (T, B, E)
        return src

if __name__=='__main__':
    # x = TransformerEncoderLayer_RP(20,4,2)
    # inp = torch.ones((2,3,20),dtype=torch.float)
    # print(x(inp).shape)
    m=RelativePosition(256, 3)
    print(m(10,10))