
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools, time
import copy
from models.package.focal_loss import focal_loss




class frameRNN_dec(nn.Module): # offline 
    def __init__(self, n_units, n_speakers, rnn_cell='GRU', dropout=0.2):
        super(frameRNN_dec, self).__init__()
        self.n_speakers = n_speakers
        self.enc_dec_proj = nn.Linear(n_units, n_units)
        # if rnn_cell == 'reluRNN':
        #     self.mixer = nn.RNNCell(n_units, n_units, nonlinearity='relu')
        # else:
        #     self.mixer = getattr(nn, f"{rnn_cell}Cell")(n_units, n_units)
        self.adder = nn.GRU(n_units, n_units, batch_first=True)
        self.miner = nn.GRU(n_units, n_units, batch_first=True)

        self.projection_spk = lambda x: x # nn.Linear(n_units, n_units)
        self.projection_emb = lambda x: x # nn.Linear(n_units, n_units)

        # TODO replace self.vad by self.rnn_init_hidden ?
        # self.vad = nn.Sequential(nn.Linear(n_units, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.rnn_init_hidden = nn.Parameter(torch.zeros(1, n_units))

    def vad(self, emb):
        vad_vec = self.rnn_init_hidden.expand_as(emb)
        vad_prob = torch.sigmoid(torch.mul(self.projection_spk(vad_vec), self.projection_emb(emb)).sum(dim=-1))
        return vad_prob



    def forward(self, enc_output, seq_len, label):
        '''
        frame_emb: (B, T, D)
        seq_len: (B, )
        label: (B, T, C)
        '''
        device = self.rnn_init_hidden.device

        batch_size, max_len, spk_num = label.shape
        
        # active_spk = torch.nonzero(label.transpose(1,2))
        # spk_active_frames = [ [[] for _ in range(spk_num)] 
        #                         for _ in range(batch_size)]
        # for b, n, i in active_spk.cpu().numpy():
        #     spk_active_frames[b][n].append(i) # spk_active_frames[batch_id][spk_id] = orderedlist(frame_ids)
        # for b in range(len(spk_active_frames)):
        #     spk_active_frames[b].sort(key=lambda x: x[0] if len(x)!=0 else max_len+1) # sort by starting frame
        
        spk_active_frames = []
        for b in label.transpose(1,2):
            spk_in_batch = [torch.nonzero(n).squeeze(-1) for n in b]
            spk_in_batch.sort(key=lambda x: x[0] if len(x)!=0 else max_len+1) # sort by starting frame
            spk_active_frames.append(spk_in_batch)

        frame_emb = self.dropout(self.enc_dec_proj(enc_output))
        spk_in_frame = torch.sum(label, dim=-1)
        valid_frames = seq_len + torch.sum(spk_in_frame, dim=-1)
        
        batch_spk_loss, frame_active_loss = [torch.zeros((), device=device, requires_grad=True) for _ in range(batch_size)], \
                                            [torch.zeros((), device=device, requires_grad=True) for _ in range(batch_size)]
        for i in range(batch_size):
            vad_label = (spk_in_frame[i, :seq_len[i]] > 0).float()

            vad_weight = vad_label.masked_fill(vad_label == 1, vad_label.shape[0] / (vad_label.sum(dim=0) + 1e-9)).masked_fill(vad_label == 0, vad_label.shape[0] / (vad_label.shape[0] - vad_label.sum(dim=0) + 1e-9))
            frame_active_loss[i] = frame_active_loss[i] + F.binary_cross_entropy(
                                       self.vad(frame_emb[i,:seq_len[i]]) , vad_label, 
                                       weight=vad_weight, reduction='sum')

        spk_init_state = self.rnn_init_hidden.repeat(batch_size, 1).unsqueeze(0)

        def extract_frames(b, spk):
            select_ids = spk_active_frames[b][spk]
            return select_ids, frame_emb[b,select_ids]
            
        for spk_id in range(spk_num):
            chosen_frame_ids, chosen_frames = zip(*[extract_frames(b, spk_id) for b in range(batch_size)])
            # print(chosen_frame_ids)

            nonempty_chosen_frames = [f for i,f in zip(chosen_frame_ids, chosen_frames) if len(i)!=0]
            if len(nonempty_chosen_frames) == 0:
                continue

            packed_adder_inp = nn.utils.rnn.pack_sequence(nonempty_chosen_frames, enforce_sorted=False)
            packed_adder_out, adder_h = self.adder(packed_adder_inp, spk_init_state)
            adder_out, add_out_len = nn.utils.rnn.pad_packed_sequence(packed_adder_out, batch_first=True)

            # Spk active loss(BCE) ========= ADDER
            adder_out = [o[:l] for o,l in zip(adder_out, add_out_len)]
            for batch_id in range(batch_size):
                frame_ids = chosen_frame_ids[batch_id]
                if len(frame_ids) == 0:
                    continue
                elif frame_ids[0] == (seq_len[batch_id] - 1):
                    adder_out.pop(0)
                    continue

                st_frame = frame_ids[0]
                batch_frames = frame_emb[batch_id, st_frame + 1 : seq_len[batch_id]].unsqueeze(1)
                repeat_times = F.pad(frame_ids[1:], (0,1), value=seq_len[batch_id]-1) - frame_ids
                batch_hidden = torch.repeat_interleave(adder_out.pop(0), repeat_times, dim=0).unsqueeze(1)


                active_score = torch.sigmoid(torch.bmm(self.projection_emb(batch_frames), #T,1,D X T,D,1
                                    self.projection_spk(batch_hidden).transpose(1,2)).squeeze(-1).squeeze(-1))
                spk_label = torch.zeros_like(active_score)
                spk_label[frame_ids[1:] - st_frame - 1] = 1
                # spk_label = label[batch_id, st_frame + 1 : seq_len[batch_id], spk_id]

                spk_weight = spk_label.masked_fill(spk_label == 1, spk_label.shape[0] / (spk_label.sum(dim=0) + 1e-9)).masked_fill(spk_label == 0, spk_label.shape[0] / (spk_label.shape[0] - spk_label.sum(dim=0) + 1e-9))
                batch_spk_loss[batch_id]  = batch_spk_loss[batch_id] + F.binary_cross_entropy(
                                                active_score, spk_label, weight=spk_weight)

                # prev_hidden = torch.repeat_interleave(adder_out[batch_id])

            miner_inp = torch.cat(chosen_frames, dim=0).unsqueeze(dim=1)
            
            # TODO Offline training, consider online?
            miner_in_h = torch.repeat_interleave(adder_h, add_out_len.to(device=adder_h.device), dim=1)

            miner_out, miner_h = self.miner(self.dropout(miner_inp), miner_in_h)
            updated_chosen_frames = miner_out.squeeze(1)
  
            update_index = (torch.cat([torch.ones_like(i) * bid 
                                for bid, i in enumerate(chosen_frame_ids)], dim=0), 
                            torch.cat(chosen_frame_ids, dim=0))
            frame_emb = torch.index_put(frame_emb, update_index, updated_chosen_frames) # update frame_emb

            # frame active loss
            st = 0

            for i in range(batch_size):
                updated_chosen_vad = self.vad(updated_chosen_frames[st :st+len(chosen_frame_ids[i])])
                st += len(chosen_frame_ids[i])
                spk_in_frame[i, chosen_frame_ids[i]] -= 1
                updated_chosen_label = (spk_in_frame[i, chosen_frame_ids[i]] > 0).float()
                # assert torch.any(updated_chosen_label >= 0) or updated_chosen_label.shape[0] == 0,f"{updated_chosen_label}"
                pos = updated_chosen_label.shape[0] / (updated_chosen_label.sum(dim=0) + 1e-9) 
                vad_weight = updated_chosen_label.masked_fill(updated_chosen_label == 0, updated_chosen_label.shape[0] / (updated_chosen_label.sum(dim=0) + 1e-9)).masked_fill(updated_chosen_label == 0, updated_chosen_label.shape[0] / (updated_chosen_label.shape[0] - updated_chosen_label.sum(dim=0) + 1e-9))

                frame_active_loss[i] = frame_active_loss[i] + F.binary_cross_entropy(
                                        updated_chosen_vad, updated_chosen_label,
                                        weight=vad_weight, reduction='sum')



        batch_spk_loss = torch.stack(batch_spk_loss, dim=0) / spk_num
        frame_active_loss = torch.stack(frame_active_loss, dim=0) / (valid_frames + 1e-5)
        # print(batch_spk_loss)
        # print(frame_active_loss)
        # print(valid_frames)

        return batch_spk_loss, frame_active_loss

    def dec_each_offline(self, enc_output, th=0.5):
        '''
        enc_output: (T,D)
        '''
        #TODO many bug here?
        device, time_len = enc_output.device, enc_output.shape[0]
        frame_emb = self.enc_dec_proj(enc_output) # (T,D)
        pred = torch.zeros((time_len, self.n_speakers), device=device)
        for spk_turn in range(self.n_speakers):
            # frame_vad_prob = self.vad(frame_emb).squeeze(-1) # (T,)
            spk_states = self.rnn_init_hidden
            spk_active_frames = []
            for time_step, time_emb in enumerate(frame_emb):
                # if frame_vad_prob[time_step] < th: # No speaker
                #     continue

                if spk_states is None: # The first frame
                    spk_active_frames.append(time_step)
                    _, spk_states  = self.adder(time_emb.view(1,1,-1), self.rnn_init_hidden.view(1,1,-1))
                    spk_states = spk_states.squeeze()
                    continue

                # calculate spk active probability
                spk_prob = torch.sigmoid(torch.mm(self.projection_spk(spk_states),self.projection_emb(time_emb).unsqueeze(1)).sum())

                # if active, update spk_states(+) 
                if spk_prob > th:
                    spk_active_frames.append(time_step)
                    _, spk_states  = self.adder(time_emb.view(1,1,-1), spk_states.view(1,1,-1))
                    spk_states = spk_states.squeeze(0)
            
            # update frame_emb (for offline)
            if spk_active_frames != []:

                # offline_prob, st = [], 0
                # while st < frame_emb.shape[0]:
                #     offline_prob.append(torch.sigmoid(torch.mm(self.projection_spk(spk_states), 
                #             self.projection_emb(frame_emb[st: st+256]).transpose(0,1))).squeeze(dim=0))
                #     st += 256
                # offline_prob = torch.cat(offline_prob, dim=0)
                # torch.sigmoid(torch.mm(self.projection_spk(spk_states), self.projection_emb(frame_emb).transpose(0,1))).squeeze(dim=0)
                # spk_active_frames = torch.nonzero(offline_prob > th).squeeze(dim=-1)

                pred[spk_active_frames, spk_turn] = 1
                chosen_frames = frame_emb[spk_active_frames]
            
                miner_in_h = spk_states.expand_as(chosen_frames)

                miner_out, _ = self.miner(chosen_frames.unsqueeze(1).contiguous(), miner_in_h.unsqueeze(dim=0).contiguous())
                miner_out = miner_out.squeeze(1)
                frame_emb[spk_active_frames] = miner_out
            else:
                break

        return pred






if __name__=='__main__':
    frame_emb = torch.randn((4,25, 256))
    seq_len = torch.tensor([10, 0, 25,1], dtype=torch.int)
    label = (torch.randn((4,25, 3)) > 0.6).long()
    for i,j in enumerate(seq_len):
        label[i,j:] = 0
    # truth_num = torch.sum(label.long(), dim=-1)
    # l = torch.zeros_like(label)
    # l = l.scatter(dim=2, index=truth_num, src=torch.ones_like(label))
    # print(label)
    # print(l)
    # input()

    dec = frameRNN_dec(256, n_speakers=3)
    # dec(frame_emb, seq_len, label)
    print('===================================')
    xx = dec.dec_each_offline(frame_emb[3], th=0.5)
    print(xx.shape)
    pass 