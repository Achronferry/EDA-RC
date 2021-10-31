# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import numpy as np
from tqdm import tqdm
import logging

import yamlargparse
import yaml
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.utils import constract_models
from model_utils.gpu import set_torch_device
from model_utils.scheduler import NoamScheduler
from model_utils.diarization_dataset import KaldiDiarizationDataset, my_collate
import model_utils.loss as loss_func


def process_stat_output(stat_dict, label):
    proc_state_dict = {}
    proc_state_dict['frames'] = sum([len(i) for i in label])
    if "change_points" in stat_dict:
        for pred, truth in zip(stat_dict["change_points"], label):
            cp_truth = (torch.abs(truth[1:] - truth[:-1]).sum(axis=-1) != 0)
            cp_pred = pred[1: len(truth)].bool()
            proc_state_dict["change_TP"] = proc_state_dict.get("change_TP", 0.) + float(((cp_pred == 1) & (cp_truth == 1)).sum())
            proc_state_dict["change_FP"] = proc_state_dict.get("change_FP", 0.) + float(((cp_pred == 1) & (cp_truth == 0)).sum())
            proc_state_dict["change_TN"] = proc_state_dict.get("change_TN", 0.) + float(((cp_pred == 0) & (cp_truth == 0)).sum())
            proc_state_dict["change_FN"] = proc_state_dict.get("change_FN", 0.) + float(((cp_pred == 0) & (cp_truth == 1)).sum())
    if "num_pred" in stat_dict:
        for pred, truth in zip(stat_dict["num_pred"], label):
            np_truth = (truth.sum(dim=-1) != 0)
            pred = pred[:len(np_truth)]
            proc_state_dict["num_pred_acc"] = proc_state_dict.get("num_pred_acc", 0.) + float((np_truth == pred).sum())


    proc_state_avg = {k: v/len(label) for k,v in proc_state_dict.items()}
    return proc_state_avg
  



def train(args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    # Logger settings====================================================
    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pytorch")
    fh = logging.FileHandler(args.model_save_dir + "/train.log", mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # ===================================================================
    logger.info(str(args))

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
    model = constract_models(args, Y.shape[1], args.model_save_dir + "/param.yaml")


    if args.gpu == 0: # single gpu
        device = set_torch_device(args.gpu)
    elif args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model, list(range(args.gpu)))

    # device = set_torch_device(args.gpu)
    model = model.to(device)
    logger.info('Prepared model')
    logger.info(model)

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    else:
        raise ValueError(args.optimizer)

    # For noam, we use noam scheduler
    if args.optimizer == 'noam':
        scheduler = NoamScheduler(optimizer,
                                  args.hidden_size,
                                  warmup_steps=args.noam_warmup_steps)

    # Init/Resume
    if args.initmodel:
        logger.info(f"Load model from {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))
    elif args.resume != 0:
        last_epoch_model = os.path.join(args.model_save_dir, f"transformer{args.resume}.th")
        model.load_state_dict(torch.load(last_epoch_model))
        optimizer.load_state_dict(torch.load(os.path.join(args.model_save_dir, "last.optim")))
        if args.optimizer == 'noam':
            scheduler.load_state_dict(torch.load(os.path.join(args.model_save_dir, "last.sche")))
        logger.info(f"Load model from {last_epoch_model}")

    assert args.batchsize % args.gradient_accumulation_steps == 0
    train_iter = DataLoader(
            train_set,
            batch_size=args.batchsize // args.gradient_accumulation_steps,
            shuffle=True,
            num_workers=16,
            collate_fn=my_collate
            )

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize // args.gradient_accumulation_steps,
            shuffle=False,
            num_workers=16,
            collate_fn=my_collate
            )

    # Training
    # y: feats, t: label
    # grad accumulation is according to: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    for epoch in range(args.resume + 1, args.max_epochs + 1):
        model.train()
        # zero grad here to accumualte gradient
        optimizer.zero_grad()
        loss_epoch = [0, 0, 0]
        num_total = 0
        for step, (y, t) in tqdm(enumerate(train_iter), ncols=100, total=len(train_iter)):
            ilens = torch.tensor([yi.shape[0] for yi in y]).long().to(device)
            y = nn.utils.rnn.pad_sequence(y, padding_value=0, batch_first=True).to(device)
            t = nn.utils.rnn.pad_sequence(t, padding_value=0, batch_first=True).to(device)
            all_losses = model(y, seq_lens=ilens, label=t)
            losses = [i.mean() for i in all_losses]
            loss = sum(losses)
            loss_epoch = [loss_epoch[i] + losses[i].item() for i in range(len(losses))]
            # clear graph here
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # noam should be updated on step-level
                if args.optimizer == 'noam':
                    scheduler.step()
                if args.gradclip > 0:
                    nn.utils.clip_grad_value_(model.parameters(), args.gradclip)
            num_total += 1
        loss_epoch = [i / num_total for i in loss_epoch]

        model.eval()
        with torch.no_grad():
            stats_avg = {}
            cnt = 0
            for y, t in dev_iter:
                ilens = torch.tensor([yi.shape[0] for yi in y]).long().to(device)
                y = nn.utils.rnn.pad_sequence(y, padding_value=0, batch_first=True).to(device)
                t = nn.utils.rnn.pad_sequence(t, padding_value=0, batch_first=True).to(device)
                cp = F.pad((torch.abs(t[:, 1:] - t[:, :-1]).sum(dim=-1) != 0), pad=(1, 0))
                output, stat_dict = model(y, seq_lens=ilens, change_points=cp)
                output = [out[:ilen] for out, ilen in zip(output.float(), ilens)]
                truth = [ti[:ilen] for ti, ilen in zip(t, ilens)]

                _, label = loss_func.batch_pit_loss(output, truth)
                stats = loss_func.report_diarization_error(output, label)
                # stats = {}
                stats.update(process_stat_output(stat_dict, truth))
                for k, v in stats.items():
                    stats_avg[k] = stats_avg.get(k, 0) + v
                cnt += 1
            stats_avg = {k:v/cnt for k,v in stats_avg.items()}
            stats_avg['DER'] = stats_avg.get('diarization_error', 0) / stats_avg.get('speaker_scored', 1e-6) * 100
            stats_avg['change_recall'] = stats_avg.get("change_TP", 0) / (stats_avg.get("change_TP", 0) + stats_avg.get("change_FN", 0) + 1e-6) * 100
            stats_avg['change_precision'] = stats_avg.get("change_TP", 0) / (stats_avg.get("change_TP", 0) + stats_avg.get("change_FP", 0) + 1e-6) * 100
            stats_avg['num_pred_acc'] = stats_avg.get("num_pred_acc", 0) / stats_avg.get("frames", 1e-6) * 100
            for k in stats_avg.keys():
                stats_avg[k] = round(stats_avg[k], 2)

        model_filename = os.path.join(args.model_save_dir, f"transformer{epoch}.th")
        torch.save(model.state_dict(), model_filename)
        torch.save(optimizer.state_dict(), os.path.join(args.model_save_dir, "last.optim"))
        if args.optimizer == 'noam':
            torch.save(scheduler.state_dict(), os.path.join(args.model_save_dir, "last.sche"))

        loss_info = '/'.join([f"{i:.5f}" for i in loss_epoch])
        logger.info(f"Epoch: {epoch:3d}, LR: {optimizer.param_groups[0]['lr']:.7f},\
            Training Loss: {loss_info}, Dev Stats: {stats_avg}")

    logger.info('Finished!')


if __name__=='__main__':
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('-c2', '--config2', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('-f', '--feature_config', help='feature config file path',
                        action=yamlargparse.ActionConfigFile)

    parser.add_argument('train_data_dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('valid_data_dir',
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('model_save_dir',
                        help='output model_save_dirdirectory which model file will be saved in.')
    parser.add_argument('--model-type', default='Transformer',
                        help='Type of model (Transformer)')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default=0, type=int,
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max-epochs', default=20, type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--input-transform', default='',
                        choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                                 'logmel23_mvn', 'logmel23_swn'],
                        help='input transform')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--num-speakers', default=2, type=int)
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--num-frames', default=2000, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='number of utterances in one batch')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                             ' for uni-directional rnn to see in the future')
    parser.add_argument('--hidden-size', default=256, type=int,
                        help='number of lstm output nodes')
    parser.add_argument('--in-size', default=None, type=int)
    parser.add_argument('--max-relative-position', default=2, type=int,
                        help='number of relative positions, only for RP')
    parser.add_argument('--gap', default=100, type=int,
                        help='gap of different relations, only for RP')
    parser.add_argument('--rnn-cell', default='LSTM', type=str,
                        help='cell type, only for LSTM')
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--subsampling', default=1, type=int)
    parser.add_argument('--frame-size', default=1024, type=int)
    parser.add_argument('--frame-shift', default=256, type=int)
    parser.add_argument('--sampling-rate', default=16000, type=int)
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--seed', default=777, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    train(args)
