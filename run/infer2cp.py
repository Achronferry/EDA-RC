import h5py
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from model_utils.kaldi_data import KaldiData
from Overlapping_speaker_segmentation_using_multiple_hypothesis_tracking_of_fundamental_frequency.track_pitch import get_pitch_segments



# infer_dir = 'exp/LibriSpeech_3/EEND/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_100000_baseline/adapt_callhome_3/gradclip_5_batchsize_32_num_frames_500_adam_lr_1e-4/infer/simu'
# outpath = 'exp/LibriSpeech_3/EENDC/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_100000_all/adapt_callhome_3/change_points.h5'
# with h5py.File(outpath, 'w') as wf:
#     for fname in os.listdir(infer_dir):
#         if fname[-3:] != '.h5':
#             continue
#         filepath = os.path.join(infer_dir, fname)
#         recid = fname[:-3]
#         print(recid)
#         data = h5py.File(filepath, 'r')['T_hat']
#         a = np.where(data[:] > 0.5, 1, 0)
#         o = np.pad(np.abs(a[1:] - a[:-1]).sum(axis=-1) != 0, (1,0)) 
      
#         wf.create_dataset(f'{recid}', data=o)

dataset_dir = 'data/LibriSpeech/data/dev_clean_ns3_beta2_500'
outpath = 'data/LibriSpeech/data/dev_clean_ns3_beta2_500/pitch_tracks.h5'

def gen_pitch_track(dataset_dir, outpath, subsampling):
    kaldi_obj = KaldiData(dataset_dir)
    
    
    for recid in tqdm(kaldi_obj.wavs):
        with h5py.File(outpath, 'a') as wf:
            if recid not in wf:
                wav, fs = kaldi_obj.load_wav(recid)
                tracks = get_pitch_segments(wav, fs)
                tracks = tracks[::subsampling]
                change_points = np.pad((np.abs(tracks[1:] - tracks[:-1]).sum(axis=-1) != 0), pad_width=(1, 0))
                wf.create_dataset(f'{recid}', data=change_points)
            # print(list(wf[recid]))
            # input()
        


if __name__=='__main__':
    gen_pitch_track(dataset_dir, outpath, subsampling=10)
    # a = np.array([1,2,3,4])
    # print(np.pad(a, pad_width=(1, 0)))




