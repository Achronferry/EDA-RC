import torch
import torch.nn as nn

class spk_extractor(nn.Module):
    def __init__(self, hidden_size, max_speaker_num, dropout=0.2):
        super(spk_extractor, self).__init__()
        assert max_speaker_num > 0
        self.speaker_limit = max_speaker_num + 1

        self.decoder_cell_init = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                            nn.Tanh())
        self.attractor = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 1),
                                            nn.Sigmoid())
        self.project = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states): 
        '''
        hiddtn_states: Tensor B,T,D
        extract spk embs for each frame
        '''
        self.attractor.flatten_parameters()
        batch_size, max_len, hidden_dim = hidden_states.shape
        hidden_states = self.dropout(hidden_states)
        h_0 = self.decoder_cell_init(hidden_states).view(-1, hidden_dim).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)

        inp_vector = torch.zeros((batch_size*max_len, self.speaker_limit, hidden_dim), device=h_0.device, dtype=torch.float)
        output, _ = self.attractor(inp_vector, (h_0, c_0)) # B * T , max_spk + 1, D

        active_prob = self.discriminator(output).squeeze(-1).view(batch_size, max_len, -1).contiguous() # B , T , max_spk + 1

        spk_hidden = self.project(output[: , :-1 , : ]).view(batch_size, max_len, self.speaker_limit-1, -1) # (B , T , max_spk , D)
        # similarity_matr = torch.sigmoid(torch.matmul(spk_hidden, spk_hidden.transpose(1,2))) # (B , T * max_spk , T * max_spk)
        # similarity_matr = similarity_matr.view(batch_size, max_len, self.speaker_limit-1, max_len, self.speaker_limit-1).permute(0,1,3,2,4).contiguous() # B , T ,T, max_spk , max_spk
        return spk_hidden, active_prob


class eda_spk_extractor(nn.Module):
    def __init__(self, hidden_size, max_speaker_num, dropout=0.1):
        super(eda_spk_extractor, self).__init__()
        assert max_speaker_num > 0
        self.speaker_limit = max_speaker_num + 1

        self.dec_hidden_size = hidden_size
        self.rnn_encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.attractor = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 1),
                                            nn.Sigmoid())
        self.project = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, lengths): 
        '''
        hiddtn_states: Tensor B,T,D
        extract spk embs for the total sequence
        '''
        self.rnn_encoder.flatten_parameters()
        self.attractor.flatten_parameters()
        batch_size, max_len, hidden_dim = hidden_states.shape
        nonempty_seqs = torch.nonzero(lengths).squeeze(-1)

        shuffled_inp, nonempty_lengths = [], []
        for h, l in zip(hidden_states, lengths):
            if l == 0:
                continue
            shuffled_inp.append(h.index_select(0, torch.randperm(l, device=hidden_states.device)))
            nonempty_lengths.append(l)

        output = torch.zeros((batch_size, self.speaker_limit, self.dec_hidden_size), device=hidden_states.device)
        if nonempty_lengths == []:
            return output[: , :-1 , : ], torch.zeros((batch_size, self.speaker_limit), device=hidden_states.device)

        shuffled_inp = nn.utils.rnn.pad_sequence(shuffled_inp, batch_first=True)
        shuffled_inp = self.dropout(shuffled_inp)

        padded_inp = nn.utils.rnn.pack_padded_sequence(shuffled_inp, nonempty_lengths, batch_first=True, enforce_sorted=False)

        _, dec_init_states = self.rnn_encoder(padded_inp)

        inp_vector = torch.zeros((shuffled_inp.shape[0], self.speaker_limit, hidden_dim), device=hidden_states.device, dtype=torch.float)
        nonempty_output, _ = self.attractor(inp_vector, dec_init_states) # B , max_spk + 1, D

        output = output.index_put((nonempty_seqs,), nonempty_output)

        active_prob = self.discriminator(output.detach()).squeeze(-1) # B , max_spk + 1
        active_prob = active_prob.masked_fill((lengths == 0).unsqueeze(-1), 0.)

        spk_hidden = self.project(output[: , :-1 , : ]) # (B , max_spk , D)
        return spk_hidden, active_prob




if __name__=='__main__':
    x = torch.randn((5,10,64), dtype=torch.float)
    module = spk_extractor(64,3)
    s,p = module(x)
    print(s.shape)
    print(p.shape)
    print(s[0,1,2,:,:])
    print(s[0,2,1,:,:])
    # print(p)
        
