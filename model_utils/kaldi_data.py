# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This library provides utilities for kaldi-style data directory.


from __future__ import print_function
import os
import sys
import numpy as np
import subprocess
import soundfile as sf
import io
from functools import lru_cache
import kaldiio


def load_segments(segments_file):
    """ load segments file as array """
    if not os.path.exists(segments_file):
        return None
    return np.loadtxt(
            segments_file,
            dtype=[('utt', 'object'),
                   ('rec', 'object'),
                   ('st', 'f'),
                   ('et', 'f')],
            ndmin=1)


def load_segments_hash(segments_file):
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        ret[utt] = (rec, float(st), float(et))
    return ret


def load_segments_rechash(segments_file):
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        if rec not in ret:
            ret[rec] = []
        ret[rec].append({'utt':utt, 'st':float(st), 'et':float(et)})
    return ret


def load_wav_scp(wav_scp_file):
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}


@lru_cache(maxsize=5)
def load_wav(wav_rxfilename, start=0, end=None):
    """ This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
    """
    if wav_rxfilename.endswith('|'):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1], shell=True,
                             stdout=subprocess.PIPE)
        data, samplerate = sf.read(io.BytesIO(p.stdout.read()),
                                   dtype='float32')
        # cannot seek
        data = data[start:end]
    elif wav_rxfilename == '-':
        # stdin
        data, samplerate = sf.read(sys.stdin, dtype='float32')
        # cannot seek
        data = data[start:end]
    else:
        # normal wav file
        data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
    return data, samplerate


def load_utt2spk(utt2spk_file):
    """ returns dictionary { uttid: spkid } """
    lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
    return {x[0]: x[1] for x in lines}


def load_spk2utt(spk2utt_file):
    """ returns dictionary { spkid: list of uttids } """
    if not os.path.exists(spk2utt_file):
        return None
    lines = [line.strip().split() for line in open(spk2utt_file)]
    return {x[0]: x[1:] for x in lines}


def load_reco2dur(reco2dur_file):
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(reco2dur_file):
        return None
    lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
    return {x[0]: float(x[1]) for x in lines}


def process_wav(wav_rxfilename, process):
    """ This function returns preprocessed wav_rxfilename
    Args:
        wav_rxfilename: input
        process: command which can be connected via pipe,
                use stdin and stdout
    Returns:
        wav_rxfilename: output piped command
    """
    if wav_rxfilename.endswith('|'):
        # input piped command
        return wav_rxfilename + process + "|"
    else:
        # stdin "-" or normal file
        return "cat {} | {} |".format(wav_rxfilename, process)


def extract_segments(wavs, segments=None):
    """ This function returns generator of segmented audio as
        (utterance id, numpy.float32 array)
        TODO?: sampling rate is not converted.
    """
    if segments is not None:
        # segments should be sorted by rec-id
        for seg in segments:
            wav = wavs[seg['rec']]
            data, samplerate = load_wav(wav)
            st_sample = np.rint(seg['st'] * samplerate).astype(int)
            et_sample = np.rint(seg['et'] * samplerate).astype(int)
            yield seg['utt'], data[st_sample:et_sample]
    else:
        # segments file not found,
        # wav.scp is used as segmented audio list
        for rec in wavs:
            data, samplerate = load_wav(wavs[rec])
            yield rec, data

def load_feats_scp(feats_file):
    """ returns dictionary { recid: feats }  """
    if not os.path.exists(feats_file):
        return None
    print(f"load fbank from {feats_file}")
    # with ReadHelper(f'scp:{feats_file}') as reader:
    #     data = { key:numpy_array for key, numpy_array in reader }
    lines = [line.strip().split(None, 1) for line in open(feats_file)]
    
    return {x[0]: x[1] for x in lines}

@lru_cache(maxsize=10)
def load_feats(ark_trace):
    return kaldiio.load_mat(ark_trace)

class KaldiData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.segments = load_segments_rechash(
                os.path.join(self.data_dir, 'segments'))
        self.utt2spk = load_utt2spk(
                os.path.join(self.data_dir, 'utt2spk'))
        self.wavs = load_wav_scp(
                os.path.join(self.data_dir, 'wav.scp'))
        self.reco2dur = load_reco2dur(
                os.path.join(self.data_dir, 'reco2dur'))
        self.spk2utt = load_spk2utt(
                os.path.join(self.data_dir, 'spk2utt'))

        self.feats = load_feats_scp(
                os.path.join(self.data_dir, 'feats.scp'))
        
        # self.rate = load_wav(list(self.wavs.values())[0])[1]
        self.rate = 8000

            
        # print(self.segments)

    def load_wav(self, recid, start=0, end=None):
        data, rate = load_wav(
            self.wavs[recid], start, end)
        return data, rate

    def load_feat(self, recid, start=0, end=None):
        feat = load_feats(self.feats[recid])
        return feat[start:end]