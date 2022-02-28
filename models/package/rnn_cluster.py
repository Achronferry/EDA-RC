from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import copy

class RNN_Clusterer(nn.Module):
    def __init__(self, n_units, n_speakers, rnn_cell='GRU', dropout=0.2):
        super(RNN_Clusterer, self).__init__()
        self.n_speakers = n_speakers
        if rnn_cell == 'reluRNN':
            self.mixer = nn.RNNCell(n_units, n_units, nonlinearity='relu')
        else:
            self.mixer = getattr(nn, f"{rnn_cell}Cell")(n_units, n_units)
        self.rnn_init_hidden = nn.Parameter(torch.zeros(1, n_units))
        self.pred = nn.Bilinear(n_units, n_units, 1)

    def forward(self, spk_emb, spk_nums, label):
        '''
        spk_emb: Tensor (N, #chunk, max_spk, D) invalid embs should be all-zeros
        spk_nums: Tensor (N, #chunk) [4,4,4,3]
        label: (N, #chunk, max_spk) be like [[0,1,2], [2,0,1], [1,0,2]]
        
        The order of spk in each chunk is fixed.
        '''
        device = spk_emb.device
        bsize, max_chunk_num, max_spk, _ = spk_emb.shape
        #TODO mask invalid embeddings?
        clusters = self.rnn_init_hidden.unsqueeze(0).expand(bsize, max_spk, -1) #(N, max_spk, D)

        chunk_losses = torch.tensor(0., device=device)
        for n_step in range(max_chunk_num):
            n_spk = spk_nums[:,n_step] #(N,)
            step_label = label[:, n_step, :].long() #(N, max_spk)
            step_emb = spk_emb[ :, n_step, :, :] #(N, max_spk, D)
            # calculate probs
            step_log_prob = torch.log_softmax(torch.bmm(step_emb, clusters.transpose(-1,-2)), dim=-1)
            truth = [l[:ilen] for l,ilen in zip(step_label, n_spk)]
            pred = [o[:ilen] for o,ilen in zip(step_log_prob, n_spk)]
            step_loss = []
            for p,t in zip(pred, truth):
                step_loss.append(F.nll_loss(p, t, reduction='sum') if len(p)!=0 else torch.tensor(0., device=device))
            step_loss = torch.stack(step_loss)
            chunk_losses += step_loss.sum()
            # chunk_log_probs.append(chunk_log_probs)
            # Update states
            ordered_emb = torch.gather(step_emb, dim=1, index=step_label.unsqueeze(-1).expand_as(step_emb))
            stack_outp = self.mixer(ordered_emb.reshape(bsize*max_spk, -1), clusters.reshape(bsize*max_spk, -1)).reshape(bsize, max_spk, -1)
            for b in range(bsize):
                ind2 = step_label[b, :n_spk[b]]
                ind1 = torch.ones_like(ind2) * b
                # clusters[b, step_label[b, :n_spk[b]],:] = stack_outp[b, step_label[b, :n_spk[b]],:]
                clusters = clusters.index_put((ind1, ind2), stack_outp[b, ind2])
        chunk_losses = chunk_losses / spk_nums.sum().float()
        return chunk_losses

    # TODO for a certain number of speakers
    def decode_fix_spk(self, spk_emb, spk_num):
        exist_clusters = self.rnn_init_hidden.expand(spk_num, -1)
        for chunk in spk_emb:
            sim_score = torch.mm(chunk, exist_clusters.transpose(0, 1))
        pass

    def decode_beam_search(self, spk_emb, beam_size=3):
        '''
        spk_emb: List of [Tensor(#chunk_spk, D), Tensor(#chunk_spk, D), Tensor(#chunk_spk, D), ...]
        '''
        #TODO beam search?
        # beams = [BeamState(device=self.rnn_init_hidden.device)]
        exist_clusters = self.rnn_init_hidden
        for chunk in spk_emb:
            sim_score = F.log_softmax(torch.mm(chunk, exist_clusters.transpose(0, 1)))
            expand_sim_score = F.pad(sim_score, (0,chunk.shape[0] - 1,0,0), mode='replicate')




            
            pass


            



            
            pass

class RNN_Clusterer_p(nn.Module):
    def __init__(self, n_units, n_speakers, rnn_cell='GRU', dropout=0.2):
        super(RNN_Clusterer_p, self).__init__()
        self.n_speakers = n_speakers
        if rnn_cell == 'reluRNN':
            self.mixer = nn.RNNCell(n_units, n_units, nonlinearity='relu')
        else:
            self.mixer = getattr(nn, f"{rnn_cell}Cell")(n_units, n_units)
        self.rnn_init_hidden = nn.Parameter(torch.zeros(1, n_units))

    def forward(self, spk_emb, label=None):
        '''
        spk_emb: (N, #chunk, #spk, D)
        label: (N, #chunk, #spk) 0/1
        '''
        device = self.rnn_init_hidden.device
        ctmpr_spks = torch.sum(label.long(), dim=-1, keepdim=False)

        if label is not None:
            batch_dec_set = [[j for j in zip(*i)] for i in zip(spk_emb, ctmpr_spks, label)]
            batch_hidden_set = [self.rnn_init_hidden.repeat(self.n_speakers, 1)
                                 for _ in batch_dec_set] #B, (C,D)
            cluster_loss = []
            cnt = 0
            while True:
                # find next non-empty frame for each sample
                current_one_step = []
                for i in batch_dec_set:
                    search_next = [None, 0, None]
                    while search_next[1] == 0:
                        if i == []:
                            search_next = [None, 0, None]
                            break
                        search_next = i.pop(0)
                    current_one_step.append(search_next)
                # print(current_one_step)


                if sum(map(lambda x: x[1], current_one_step)) == 0:
                    break
                cnt += 1
                step_len, step_label, step_prob, step_loss = [], [], [], []
                prev_hiddens, step_inp = [], []
                for i, bh in zip(current_one_step, batch_hidden_set):
                    step_len.append(i[1])
                    if i[1] == 0:
                        step_loss.append(torch.tensor(0., device=device))
                        continue
                    i_valid = i[0][:i[1]]
                    i_prob = torch.softmax(torch.mm(i_valid, bh.transpose(0,1)), dim=-1)
                    # i_perms = [torch.stack(x, dim=0).float() for x in itertools.permutations(
                    #                         torch.masked_select(torch.diag(i[2]), i[2].unsqueeze(-1).bool()).view((-1,i[2].shape[-1])))]
                    # i_losses = torch.stack([F.binary_cross_entropy(i_prob, x)
                    #                         for x in i_perms], dim=0)
                    # i_loss = torch.min(i_losses)
                    # i_label = i_perms[i_losses.argmin()].argmax(dim=-1)
                    i_perms = [torch.stack(x, dim=0) for x in itertools.permutations(torch.nonzero(i[2]).squeeze(-1))]
                    i_losses = torch.stack([F.nll_loss(i_prob.log(), x)
                                            for x in i_perms], dim=0)
                    i_loss = torch.min(i_losses)
                    i_label = i_perms[i_losses.argmin()]
                    
                    step_inp.append(i_valid)
                    prev_hiddens.append(torch.index_select(bh, 0, i_label))

                    step_prob.append(i_prob)
                    step_label.append(i_label)
                    step_loss.append(i_loss)
                step_loss = torch.stack(step_loss, dim=0) # B, 1
                cluster_loss.append(step_loss)
                # print(step_len) # B, 1 
                # print(step_loss) 
                # print(step_label) 
                # print(step_prob)
                
                # Update hidden states
                step_inp = torch.cat(step_inp, dim=0)
                prev_hiddens = torch.cat(prev_hiddens, dim=0)
                assert step_inp.shape == prev_hiddens.shape
                rnn_hiddens = self.mixer(step_inp, prev_hiddens)
                
                st = 0
                for idx, n in enumerate(step_len):
                    if n == 0:
                        continue
                    
                    update_spks = step_label.pop(0)
                    #batch_hidden_set[idx][update_spks] = rnn_hiddens[st: st+n]
                    batch_hidden_set[idx] = batch_hidden_set[idx].index_put((update_spks, ), rnn_hiddens[st: st+n])
                    st += n
                assert st == rnn_hiddens.shape[0]

            if cluster_loss != []:
                cluster_loss = torch.stack(cluster_loss, dim=0).sum(dim=0)
                non_zero_nums = (ctmpr_spks != 0).sum(dim=-1)
                # assert non_zero_nums == (cluster_loss!=0).sum(dim=0)
                cluster_loss = cluster_loss / (non_zero_nums + 1e-6)
            else:
                cluster_loss = torch.tensor([0.], device=device)

            return cluster_loss
        
        if label is None:
            raise NotImplementedError



    def decode_beam_search(self, e, beam_size=3):
        beams = [BeamState(device=self.rnn_init_hidden.device)]
        # cnt = 0
        for step_embs in e:
            # print(cnt)
            # cnt += 1
            if step_embs.shape[0] == 0:
                beams = [b.pad() for b in beams]
                continue
            # print(f"step_embs:  {step_embs.shape}")
            # print(len(beams))
            new_beams = []
            while beams != []:
                b = beams.pop(0)
                exist_spk_num = 0 if b.hidden_states is None else b.hidden_states.shape[0]
                prev_h = self.rnn_init_hidden.expand(step_embs.shape[0], -1) if exist_spk_num == 0 \
                            else torch.cat([b.hidden_states, 
                                self.rnn_init_hidden.expand(step_embs.shape[0], -1)], dim=0)
                
                # TODO limit max speaker
                prev_h = prev_h[:self.n_speakers]
                
                scores = torch.mm(step_embs, prev_h.transpose(0, 1))
                scores = F.log_softmax(scores, dim=-1)
                # TODO allow same?
                # preds = torch.argmax(probs, dim = -1)
                perms = torch.stack([torch.stack(x, dim=0).bool() for x in itertools.permutations(
                            torch.diag(torch.ones_like(scores[0])), scores.shape[0])], dim=0)
                hyp_scores = torch.stack([torch.masked_select(scores, i).sum() for i in perms], dim=0)
                hyp_scores = torch.topk(hyp_scores, min(beam_size, hyp_scores.shape[0]))
                # print(hyp_scores)

                new_hyp_scores = hyp_scores.values
                new_hyp_perms = perms[hyp_scores.indices]
                # new_hyp_indices = new_hyp_perm.long().argmax(dim=-1)

                for p,s in zip(new_hyp_perms, new_hyp_scores):
                    if len(new_beams) >= beam_size and s < new_beams[-1].score:
                        continue

                    i = p.long().argmax(dim=-1) #[[0,1,0], [0,0,1]] -> [2,3]
                    before_hiddens = prev_h[i]
                    after_hiddens = self.mixer(step_embs, before_hiddens)
                    next_h = prev_h.clone()
                    next_h[i] = after_hiddens
                    
                    new_label = torch.sum(p, dim=0) #[[0,1,0], [0,0,1]] -> [0,1,1]
                    avail_mask = new_label.clone().bool()
                    avail_mask[:exist_spk_num] = True
                    # print(avail_mask)
                    new_label = torch.masked_select(new_label, avail_mask)
                    next_h = next_h[torch.nonzero(avail_mask, as_tuple=True)]
                    # next_h = torch.masked_select(next_h, avail_h)
                    new_beams.append(b.clone_and_apply(s, next_h, new_label,i))

                new_beams.sort(key=lambda x: x.score, reverse=True)
            beams = new_beams[:beam_size]
                # rnn_hiddens = torch.index_select(prev_h, new_hyp_indices)
                # new_hiddens = self.mixer(, rnn_hiddens
                

        return beams


class BeamState:
    '''States for beam search decoding.'''

    def __init__(self, device=None):
        self.device = device
        self.hidden_states = None
        self.cluster_embs = []
        self.score = 0.
        self.pred = []
        self.T = 0
        self.pred_order = []
    
    def copy(self):
        new_beam = BeamState(self.device)
        new_beam.hidden_states = self.hidden_states
        new_beam.cluster_embs = copy.deepcopy(self.cluster_embs)
        new_beam.score = self.score
        new_beam.pred = copy.copy(self.pred)
        new_beam.pred_order = copy.copy(self.pred_order)
        new_beam.T = self.T   
        return new_beam     



    def clone_and_apply(self, score, hidden_states, pred_label, pred_order):
        new_state = self.copy()
        new_state.T += 1
        new_state.score += score
        new_state.hidden_states = hidden_states
        new_state.pred.append(pred_label)
        new_state.pred_order.append(pred_order)
        
        return new_state

    def pad(self):
        self.T += 1
        self.pred.append(torch.zeros((1), device=self.device))
        return self




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
    dec(frame_emb, seq_len, label)
    pass 
