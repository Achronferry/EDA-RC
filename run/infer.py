#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
from scipy.ndimage import shift
from tqdm import tqdm

import yamlargparse
import torch
import torch.nn as nn

from model_utils.gpu import set_torch_device
from models.utils import constract_models
from model_utils import feature
from model_utils import kaldi_data


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


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
    for recid in tqdm(kaldi_obj.feats):
        # data, rate = kaldi_obj.load_wav(recid)
        # Y = feature.stft(data, args.frame_size, args.frame_shift)
        # Y = feature.transform(Y, transform_type=args.input_transform)
        Y = kaldi_obj.load_feat(recid)
        Y = feature.splice(Y, context_size=args.context_size)
        if args.input_transform == 'logmel23_mn':
            Y = Y - np.mean(Y, axis=0)
        Y = Y[::args.subsampling]
        out_chunks = []
        probs_chunks = []
        with torch.no_grad():
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size):
                Y_chunked = torch.from_numpy(Y[start:end]).to(device)
                ys, spk_probs = model(Y_chunked.unsqueeze(0), seq_lens=torch.tensor([Y_chunked.shape[0]]).long().to(device))
                out_chunks.append(ys[0].cpu().detach().numpy())
                if 'np' in args.model_type.lower():
                    probs_chunks.append(spk_probs[0].cpu().detach().numpy())
                if args.save_attention_weight == 1:
                    raise NotImplementedError()
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if args.label_delay != 0:
            outdata = shift(np.vstack(out_chunks), (-args.label_delay, 0))
            if 'np' in args.model_type.lower():
                probdata = shift(np.vstack(probs_chunks), (-args.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks)
            if 'np' in args.model_type.lower():
                probdata = np.vstack(probs_chunks)

        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
            if 'np' in args.model_type.lower():
                wf.create_dataset('N_hat', data=outdata)


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
    parser.add_argument('--rnn-cell', default='LSTM', type=str,
                        help='cell type, only for LSTM')
    parser.add_argument('--max-relative-position', default=2, type=int,
                        help='nember of relative positions, only for RAT')
    parser.add_argument('--gap', default=100, type=int,
                        help='gap of different relations, only for RP')
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
    parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
    parser.add_argument('--save-attention-weight', default=0, type=int)
    args = parser.parse_args()

    print(str(args))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    infer(args)


