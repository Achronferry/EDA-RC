# import kaldiio

# data_dir = 'data/LibriSpeech/data/dev_clean_ns2_beta2_500'
# chunk_size=2000
# context_size=0
# subsampling=1
# use_last_samples=False
# label_delay=0

# from kaldiio import ReadHelper, load_ark, load_mat
# # with ReadHelper(f'scp:{data_dir}/feats.scp') as reader:
# #     for key, numpy_array in reader:
# #         print(key)
# #         print(numpy_array)
# #         print(numpy_array.shape)
# #         input()
# print(load_mat('/GPFS/data/chenyuyang/exp/RPE_EEND/local/diarization-data/LibriSpeech/fbank/train_clean_360_ns2_beta2_100000/raw_fbank_train_clean_360_ns2_beta2_100000.1.ark:70').shape)
# # with ReadHelper(f'scp:{data_dir}/wav.scp') as reader:
# #     for key, (rate, numpy_array) in reader:
# #         print(key)
# #         print(numpy_array)
# #         print(numpy_array.shape)
# #         input()

import torch
# a = torch.zeros((3,5,5))
# b = torch.randn((3,5,5)) > 0.5
# print(b)
# c = b.sum(dim = -1).sum(dim=-1).sum(dim=-1)
# d = torch.range(1,c).float()
# print(d)
# f = torch.masked_scatter(a,b,d)
# print(f)
# def bin2dec(b, bits):
#     mask = 2 ** torch.arange(0, bits, 1).to(b.device, b.dtype).unsqueeze(0).unsqueeze((0))
#     print(mask)
#     return torch.sum(mask * b, -1)

# b = torch.tensor([[0,0,1,0,1], [0,0,0,0,0]]).unsqueeze(0).expand(3,2,5)
# print(b)
# print(b.sum(axis=-1))

a = 'fix_len_as'
b = 'fix_len_2014'
c = 'fix_len_'
d = 'qwq'
e = 'fix_len_2aqrdqa'
import re
assert re.match('fix_len_\d+$',d)
print(re.match('fix_len_\d+$',b))
# b = torch.tensor([3,2,4])
# a[range(a.shape[0]), b] = 1
# print(a)

# a = torch.ones(5)
# b = torch.zeros(1)
# c = torch.tensor([6])
# print(torch.cat([b,a,c], dim=0))