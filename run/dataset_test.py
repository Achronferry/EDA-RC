# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import numpy as np
from tqdm import tqdm
import yamlargparse
import yaml
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from models.utils import constract_models
from model_utils.gpu import set_torch_device
from model_utils.scheduler import NoamScheduler
from model_utils.diarization_dataset import KaldiDiarizationDataset, my_collate
import model_utils.loss as loss_func


def train(args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    print(args.num_speakers)

    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set = KaldiDiarizationDataset(
        data_dir=args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )
    dev_set = KaldiDiarizationDataset(
        data_dir=args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    Y, T = next(iter(train_set))


    if args.gpu == 0: # single gpu
        device = set_torch_device(args.gpu)
    elif args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_iter = DataLoader(
            train_set,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=16,
            collate_fn=my_collate
            )

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=16,
            collate_fn=my_collate
            )

    # Training
    # y: feats, t: label
    # grad accumulation is according to: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    for epoch in range(1):
        err_cnt = 0
        for step, (y, t) in tqdm(enumerate(train_iter), ncols=100, total=len(train_iter)):
            ilens = torch.tensor([yi.shape[0] for yi in y]).long().to(device)
            y = nn.utils.rnn.pad_sequence(y, padding_value=0, batch_first=True).to(device)
            tt = nn.utils.rnn.pad_sequence(t, padding_value=0, batch_first=True).to(device).long()
            for t in tt:
                for i in range(1, len(t)):
                    if (~(t[i] == t[i-1])).sum() != 0 and t[i].sum() == t[i-1].sum():
                        print(t[i])
                        print(t[i-1])
                        print('===============')
                        err_cnt += 1
        print(f"train_err_num:{err_cnt}")

        err_cnt = 0
        with torch.no_grad():
            for y, t in tqdm(dev_iter):
                ilens = torch.tensor([yi.shape[0] for yi in y]).long().to(device)
                y = nn.utils.rnn.pad_sequence(y, padding_value=0, batch_first=True).to(device)
                tt = nn.utils.rnn.pad_sequence(t, padding_value=0, batch_first=True).to(device).long()
            for t in tt:
                for i in range(1, len(t)):
                    if (~(t[i] == t[i-1])).sum() != 0 and t[i].sum() == t[i-1].sum():
                        print(t[i])
                        print(t[i-1])
                        print('===============')
                        err_cnt += 1
        print(f"test_err_num:{err_cnt}")       




if __name__=='__main__':
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('train_data_dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('valid_data_dir',
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num-speakers', default=2, type=int)

    parser.add_argument('--num-frames', default=2000, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='number of utterances in one batch')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                             ' for uni-directional rnn to see in the future')
    parser.add_argument('--context-size', default=7, type=int)
    parser.add_argument('--subsampling', default=1, type=int)
    parser.add_argument('--frame-size', default=1024, type=int)
    parser.add_argument('--frame-shift', default=256, type=int)
    parser.add_argument('--sampling-rate', default=16000, type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--input-transform', default='logmel23_mn',
                        choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                                 'logmel23_mvn', 'logmel23_swn'],
                        help='input transform')
    args,_ = parser.parse_known_args()

    train(args)