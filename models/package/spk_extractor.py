import torch
import torch.nn as nn

class spk_extractor(nn.Module):
    def __init__(self, hidden_size, max_speaker_num):
        super(spk_extractor, self).__init__()
        assert max_speaker_num > 0
        self.speaker_limit = max_speaker_num + 1

        self.decoder_cell_init = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                            nn.Tanh())
        self.attractor = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 1),
                                            nn.Sigmoid())
        self.project = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states): 
        '''
        hiddtn_states: Tensor B,T,D
        extract spk embs for each frame
        '''
        self.attractor.flatten_parameters()
        batch_size, max_len, hidden_dim = hidden_states.shape
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
    def __init__(self, hidden_size, max_speaker_num):
        super(eda_spk_extractor, self).__init__()
        assert max_speaker_num > 0
        self.speaker_limit = max_speaker_num + 1

        self.rnn_encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.attractor = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 1),
                                            nn.Sigmoid())
        self.project = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, lengths): 
        '''
        hiddtn_states: Tensor B,T,D
        extract spk embs for the total sequence
        '''
        self.encoder.flatten_parameters()
        self.attractor.flatten_parameters()
        batch_size, max_len, hidden_dim = hidden_states.shape
        shuffled_inp = hidden_states.index_select(1, torch.randperm(max_len))
        padded_inp = nn.utils.rnn.pack_padded_sequence(shuffled_inp, lengths, batch_first=True, enforce_sorted=False)
        _, dec_init_states = self.rnn_encoder(padded_inp)

        inp_vector = torch.zeros((batch_size, self.speaker_limit, hidden_dim), device=hidden_states.device, dtype=torch.float)
        output, _ = self.attractor(inp_vector, dec_init_states) # B , max_spk + 1, D

        active_prob = self.discriminator(output).squeeze(-1) # B , max_spk + 1

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
        
