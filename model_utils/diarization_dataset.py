#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from model_utils import kaldi_data
from model_utils import feature


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    # data, target = list(zip(*batch))
    return list(zip(*batch))


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            use_spk_id=False
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay
        self.use_spk_id = use_spk_id

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        if self.data.feats is None:
            Y, T, spk_id = feature.get_labeledSTFT(
                self.data,
                rec,
                st,
                ed,
                self.frame_size,
                self.frame_shift,
                self.n_speakers, 
                use_speaker_id=True)
            # Y: (frame, num_ceps)
            Y = feature.transform(Y, self.input_transform)
        else:
            Y, T, spk_id = feature.get_labeledfeat(
                self.data,
                rec,
                st,
                ed,
                self.frame_size,
                self.frame_shift,
                self.n_speakers, 
                use_speaker_id=True)
            # Y: (frame, num_ceps)
            if self.input_transform == 'logmel23_mn':
                Y = Y - np.mean(Y, axis=0)
            else:
                raise NotImplementedError

        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = feature.splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)

        Y_ss = torch.from_numpy(Y_ss).float()
        T_ss = torch.from_numpy(T_ss).float()
        if self.use_spk_id:
            if "librispeech" in self.data_dir.lower():
                spk_id = [i.split('-')[0] for i in spk_id]
            return Y_ss, T_ss, spk_id
        else:
            return Y_ss, T_ss
