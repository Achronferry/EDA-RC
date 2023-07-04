#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options

stage=0

# Modify corpus directories
#  - callhome_dir
#    CALLHOME (LDC2001S97)

ami_dir=/GPFS/public/AMI/amicorpus 
save_dir=data/ami

simu_actual_dirs=(
/GPFS/data/chenyuyang/exp/RPE_EEND/local/diarization-data/ami
)

# simulation options

. path.sh
. cmd.sh
. parse_options.sh || exit

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
    # Prepare CALLHOME dataset. This will be used to evaluation.
    if ! validate_data_dir.sh --no-text --no-feats $save_dir/ami1 \
        || ! validate_data_dir.sh --no-text --no-feats $save_dir/ami2; then

        preprocess/make_ami.sh --AMI_DIR $ami_dir --data_dir $save_dir
        copy_data_dir.sh $save_dir/ami/train $save_dir/ami1
        combine_data.sh --extra-files rttm.annotation $save_dir/ami2 $save_dir/ami/dev $save_dir/ami/test

        cp $save_dir/ami/train/rttm.annotation $save_dir/ami1/rttm
        mv $save_dir/ami2/rttm.annotation $save_dir/ami2/rttm

    fi
fi



if [ $stage -le 1 ]; then
    # compose eval/ami2
    eval_set=$save_dir/eval/ami2
    if ! validate_data_dir.sh --no-text --no-feats $eval_set; then
        utils/copy_data_dir.sh $save_dir/ami2 $eval_set
        cp $save_dir/ami2/rttm $eval_set/rttm
        awk -v dstdir=$save_dir/wav/eval/ami2 '{print $1, dstdir"/"$1".wav"}' $save_dir/ami2/wav.scp > $eval_set/wav.scp
        mkdir -p $save_dir/wav/eval/ami2
        wav-copy scp:$save_dir/ami2/wav.scp scp:$eval_set/wav.scp
        utils/data/get_reco2dur.sh $eval_set
        # rm -rf $save_dir/ami2
    fi

    # compose eval/ami1
    adapt_set=$save_dir/eval/ami1
    if ! validate_data_dir.sh --no-text --no-feats $adapt_set; then
        utils/copy_data_dir.sh $save_dir/ami1 $adapt_set
        cp $save_dir/ami1/rttm $adapt_set/rttm
        awk -v dstdir=$save_dir/wav/eval/ami1 '{print $1, dstdir"/"$1".wav"}' $save_dir/ami1/wav.scp > $adapt_set/wav.scp
        mkdir -p $save_dir/wav/eval/ami1
        wav-copy scp:$save_dir/ami1/wav.scp scp:$adapt_set/wav.scp
        utils/data/get_reco2dur.sh $adapt_set
        # rm -rf $save_dir/ami1
    fi
fi



# if [ ${stage} -le 2 ]; then
#     ### Task dependent. You have to design training and dev sets by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 2: Feature Generation"
#     fbankdir=fbank
#     # Generate the fbank features; 
#     for dset in callhome1_spk${num_spk} callhome2_spk${num_spk}; do
#         # utils/fix_data_dir.sh  $save_dir/eval/$dset
#         preprocess/make_fbank.sh --cmd "$train_cmd" --nj 12 --write_utt2num_frames true --fbank_config conf/fbank.conf \
#             $save_dir/eval/$dset exp/make_fbank/callhome/$dset $simu_actual_dirs/fbank/$dset
#         # utils/fix_data_dir.sh  $save_dir/eval/$dset
#     done
#     # # subset of dev_set
#     # utils/subset_data_dir.sh data/${train_dev} 1000 data/${train_dev}_u1k

#     # # compute global CMVN
#     # compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
# fi