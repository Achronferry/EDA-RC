import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

class num_pred_seg(nn.Module):
    def __init__(self, n_units, n_speakers=3):
        super(num_pred_seg, self).__init__()
        self.num_predictor = nn.Sequential(nn.Linear(n_units, n_speakers + 1), 
                                            nn.Sigmoid())


    def forward(self, seg_emb, seq_len=None, label=None, th=0.5):
        num_prob = self.num_predictor(seg_emb)

        if label is not None:  
            truth_num = torch.sum(label.long(), dim=-1, keepdim=True)
            l = torch.zeros_like(num_prob)
            l = l.scatter(dim=2, index=truth_num, src=torch.ones_like(num_prob))
            num_pred_loss = []
            for y, t, sample_len in zip(num_prob, l, seq_len):
                y, t = y[:sample_len], t[:sample_len]
                num_pred_loss.append(F.binary_cross_entropy(y, t))
            num_pred_loss = torch.stack(num_pred_loss, dim=0)
            return num_pred_loss
        else:
            pred = torch.argmax(num_prob, dim=-1)
            change_points = F.pad((torch.abs(pred[:, 1:] - pred[:, :-1]) != 0).float(), pad=(1, 0))
            return change_points, pred






class vec_sim_seg(nn.Module):
    def __init__(self, n_units):
        super(vec_sim_seg, self).__init__()
        self.input_trans = nn.Linear(n_units, n_units)


    def DPCL_loss(self, emb, label):
        '''
        emb: (T, D); label: (T, C)
        '''

        # TODO partially different ? 
        n_frames = emb.shape[0] 
        def bin2dec(b):
            bits = b.shape[-1]
            mask = 2 ** torch.arange(0, bits, 1).to(b.device, b.dtype).unsqueeze(0)
            return torch.sum(mask * b, -1)

        l_t = bin2dec(label)
        l_trans = torch.zeros((label.shape[0], 2**label.shape[1]), device=label.device)
        l_trans = l_trans.scatter(-1, l_t.unsqueeze(-1).long(), 1)

        dpcl_loss = torch.mm(emb.transpose(-1, -2), emb).square().sum() \
                    + torch.mm(l_trans.transpose(-1, -2), l_trans).square().sum() \
                    - 2 * torch.mm(emb.transpose(-1, -2), l_trans).square().sum()
        dpcl_loss = dpcl_loss / (n_frames ** 2)

        return dpcl_loss



    def forward(self, seg_emb, seq_len, label=None, th=0.5):

        def normalize(batch_v):
            return batch_v / (torch.norm(batch_v, p=2, dim=-1) + 1e-9).unsqueeze(-1)

        trans_seg = self.input_trans(seg_emb)
        trans_seg = normalize(trans_seg)

        if label is not None:
            # Deep cluster
            dpcl_losses = [self.DPCL_loss(e[:n], l[:n]) for e,l,n in zip(trans_seg, label, seq_len)]
            dpcl_losses = torch.stack(dpcl_losses, dim=0)
            return dpcl_losses
        else:
            prev_seg = trans_seg[:, 1: , :]
            next_seg = trans_seg[:, :-1, :]
            change_point_prob = next_seg.mul(prev_seg).sum(dim=-1)
            change_point_prob = - F.pad(change_point_prob, pad=(1, 0), value=1) + 1
            return change_point_prob > th



from models.package.focal_loss import focal_loss
class lstm_seg(nn.Module):
    def __init__(self, n_units):
        super(lstm_seg, self).__init__()
        self.segmenter = nn.LSTM(n_units, n_units, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(n_units, 1), nn.Sigmoid())





    def forward(self, seg_emb, seq_len, label=None, th=0.5):
        lstm_out, _ = self.segmenter(seg_emb)
        # lstm_out = seg_emb
        change_prob = self.classifier(lstm_out).squeeze(-1)

        

        if label is not None:
            change_points =  F.pad((torch.abs(label[:, 1:] - label[:, :-1]).sum(dim=-1) != 0).float(), pad=(1, 0))
            change_prob = [i[:l] for i,l in zip(change_prob, seq_len)]
            change_label = [i[:l] for i,l in zip(change_points, seq_len)]


            seg_loss = [focal_loss(i, j, gamma=2, alpha=0.8) for i,j in zip(change_prob, change_label)]
            seg_loss = torch.stack(seg_loss, dim=0)
            return seg_loss
        else:
            return change_prob > th


import random
class lstm_seg_v2_rd(nn.Module):
    def __init__(self, n_units):
        super(lstm_seg_v2_rd, self).__init__()
        self.segmenter = nn.LSTM(n_units, n_units, batch_first=True)






    def forward(self, seg_emb, seq_len, label=None, th=0.5):

        if label is not None:
            rand_idx = [[j for j in range(i)] for i in seq_len]
            for i in range(len(rand_idx)):
                random.shuffle(rand_idx[i])

            rand_idx = [torch.tensor(i, device=seg_emb.device) for i in rand_idx]
            rand_emb = [torch.index_select(e,0,i) for i,e in zip(rand_idx, seg_emb)]
            rand_label = [torch.index_select(e,0,i) for i,e in zip(rand_idx, label)]
            seg_emb = nn.utils.rnn.pad_sequence(rand_emb, batch_first=True)
            label = nn.utils.rnn.pad_sequence(rand_label, batch_first=True)

            lstm_out, _ = self.segmenter(seg_emb)
            # lstm_out = seg_emb
            prev_lstm = lstm_out[:, :-1, :]
            current_emb = seg_emb[:, 1:, :]
            change_prob = torch.sigmoid(current_emb.mul(prev_lstm).sum(dim=-1))
            change_prob = F.pad(change_prob, pad=(1, 0), value=0)
            change_points =  F.pad((torch.abs(label[:, 1:] - label[:, :-1]).sum(dim=-1) != 0).float(), pad=(1, 0))
            change_prob = [i[:l] for i,l in zip(change_prob, seq_len)]
            change_label = [i[:l] for i,l in zip(change_points, seq_len)]


            seg_loss = [focal_loss(i, j, gamma=2) for i,j in zip(change_prob, change_label)]
            seg_loss = torch.stack(seg_loss, dim=0)
            return seg_loss
        else:
            lstm_out, _ = self.segmenter(seg_emb)
            # lstm_out = seg_emb
            prev_lstm = lstm_out[:, :-1, :]
            current_emb = seg_emb[:, 1:, :]
            change_prob = torch.sigmoid(current_emb.mul(prev_lstm).sum(dim=-1))
            change_prob = F.pad(change_prob, pad=(1, 0), value=0)
            return change_prob > th


class lstm_seg_v2(nn.Module):
    def __init__(self, n_units):
        super(lstm_seg_v2, self).__init__()
        self.segmenter = nn.LSTM(n_units, n_units, batch_first=True)

    def forward(self, seg_emb, seq_len, label=None, th=0.5):
        lstm_out, _ = self.segmenter(seg_emb)
        # lstm_out = seg_emb
        prev_lstm = lstm_out[:, :-1, :]
        current_emb = seg_emb[:, 1:, :]
        change_prob = torch.sigmoid(current_emb.mul(prev_lstm).sum(dim=-1))
        change_prob = F.pad(change_prob, pad=(1, 0), value=0)

        

        if label is not None:
            change_points =  F.pad((torch.abs(label[:, 1:] - label[:, :-1]).sum(dim=-1) != 0).float(), pad=(1, 0))
            change_prob = [i[:l] for i,l in zip(change_prob, seq_len)]
            change_label = [i[:l] for i,l in zip(change_points, seq_len)]


            seg_loss = [focal_loss(i, j, gamma=2, alpha=0.8) for i,j in zip(change_prob, change_label)]
            seg_loss = torch.stack(seg_loss, dim=0)
            return seg_loss
        else:
            return change_prob > th











if __name__=='__main__':
    mdl = vec_sim_seg(256)
    input_emb = torch.rand((4,200,256))
    seq_len = torch.tensor([199,58,52,200])
    label = (torch.rand((4,200,3)) > 0.5).long()
    output = mdl(input_emb, seq_len, label)
    print(output)
