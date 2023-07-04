#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os, re
import h5py
import numpy as np
from scipy.ndimage import shift
from tqdm import tqdm

import yamlargparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.gpu import set_torch_device
from models.utils import constract_models
from model_utils import feature
from model_utils import kaldi_data


def prepare_inputs(kaldi_obj, recid, context_size, input_transform, subsampling=10, frame_shift=80, rate=8000, cpd_mode='none', cp_file=None):
    Y = kaldi_obj.load_feat(recid)


    filtered_segments = kaldi_obj.segments[recid]
    speakers = np.unique(
        [kaldi_obj.utt2spk[seg['utt']] for seg
        in filtered_segments]).tolist()
    n_speakers = len(speakers)
    T = torch.zeros((Y.shape[0], n_speakers), dtype=torch.long)
    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
            seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
            seg['et'] * rate / frame_shift).astype(int)
        # rel_start = rel_end = None
        # if start <= start_frame and start_frame < end:
        #     rel_start = start_frame - start
        # if start < end_frame and end_frame <= end:
        #     rel_end = end_frame - start
        # if rel_start is not None or rel_end is not None:
        #     T[rel_start:rel_end, speaker_index] = 1
        T[start_frame:end_frame, speaker_index] = 1
    T = T[::subsampling]

    Y = feature.splice(Y, context_size=context_size)
    if input_transform == 'logmel23_mn':
        Y = Y - np.mean(Y, axis=0)
    Y = Y[::subsampling]

    return Y, T

def infer(args):
    # Prepare model

    model = constract_models(args, args.in_size)

    if args.gpu == 0: # single gpu
        device = set_torch_device(args.gpu)
    elif args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    cp_file = None
    if args.change_point_file is not None:
        cp_file = h5py.File(args.change_point_file, 'r')
    
    for recid in tqdm(kaldi_obj.feats):
        # data, rate = kaldi_obj.load_wav(recid)
        # Y = feature.stft(data, args.frame_size, args.frame_shift)
        # Y = feature.transform(Y, transform_type=args.input_transform)

        # Y = kaldi_obj.load_feat(recid)
        # Y = feature.splice(Y, context_size=args.context_size)
        # if args.input_transform == 'logmel23_mn':
        #     Y = Y - np.mean(Y, axis=0)
        # Y = Y[::args.subsampling]
        Y,T = prepare_inputs(kaldi_obj, recid, args.context_size, args.input_transform, args.subsampling, 
                            args.frame_shift, args.sampling_rate, cpd_mode=args.change_mode,cp_file=cp_file)

        with torch.no_grad():
            Y = torch.from_numpy(Y).to(device)

            outdata = T.cpu().detach().numpy()
            if args.save_attention_weight == 1:
                raise NotImplementedError()
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)

        if args.label_delay != 0:
            outdata = shift(outdata, (-args.label_delay, 0))

        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)



if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser(description='decoding')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('-c2', '--config2', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('-f', '--feature_config', help='feature config file path',
                        action=yamlargparse.ActionConfigFile)
                        
    parser.add_argument('data_dir',
                        help='kaldi-style data dir')
    parser.add_argument('model_file',
                        help='best.nnet')
    parser.add_argument('out_dir',
                        help='output directory.')
    parser.add_argument('--model_type', default='Transformer', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-speakers', type=int, default=2)
    parser.add_argument('--input-transform', default='',
                        choices=['', 'log', 'logmel',
                                 'logmel23', 'logmel23_swn', 'logmel23_mn'],
                        help='input transform')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                             ' for uni-directional rnn to see in the future')
    parser.add_argument('--hidden-size', default=256, type=int)
    parser.add_argument('--in-size', default=345, type=int)
    parser.add_argument('--inherit-from', default=None, type=str,
                        help='train from EEND (FOR EDA)')
    parser.add_argument('--rnn-cell', default='LSTM', type=str,
                        help='cell type, only for LSTM')
    parser.add_argument('--max-relative-position', default=2, type=int,
                        help='nember of relative positions, only for RAT')
    parser.add_argument('--gap', default=100, type=int,
                        help='gap of different relations, only for RP')
    parser.add_argument('--beam-size', default=1, type=int,
                        help='beam search for decoding')
    parser.add_argument('--chunk-size', default=2000, type=int,
                        help='input is chunked with this size')
    parser.add_argument('--context-size', default=0, type=int,
                        help='frame splicing')
    parser.add_argument('--subsampling', default=1, type=int)
    parser.add_argument('--sampling-rate', default=16000, type=int,
                        help='sampling rate')
    parser.add_argument('--frame-size', default=1024, type=int,
                        help='frame size')
    parser.add_argument('--frame-shift', default=256, type=int,
                        help='frame shift')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
    parser.add_argument('--save-attention-weight', default=0, type=int)
    parser.add_argument('--change-mode', default='none', type=str)
    parser.add_argument('--change-point-file', default=None, type=str)
    args = parser.parse_args()

    assert args.change_mode in ['none', 'oracle', 'full', 'from_file'] or re.match('fix_len_\d+$',args.change_mode)
    print(str(args))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    infer(args)


