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

callhome_dir=~/data_store/LDC/LDC2001S97
save_dir=data/callhome
num_spk=4


# simulation options

. path.sh
. cmd.sh
. parse_options.sh || exit

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
    # Prepare CALLHOME dataset. This will be used to evaluation.
    if ! validate_data_dir.sh --no-text --no-feats $save_dir/callhome1_spk${num_spk} \
        || ! validate_data_dir.sh --no-text --no-feats $save_dir/callhome2_spk${num_spk}; then
        # imported from https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v1
        scripts/make_callhome.sh $callhome_dir $save_dir
        # Generate two-speaker subsets
        for dset in callhome1 callhome2; do
            # Extract two-speaker recordings in wav.scp
            copy_data_dir.sh $save_dir/${dset} $save_dir/${dset}_spk${num_spk}
            utils/filter_scp.pl <(awk '{if($2<='${num_spk}') print;}'  $save_dir/${dset}/reco2num_spk) \
                $save_dir/${dset}/wav.scp > $save_dir/${dset}_spk${num_spk}/wav.scp
            # Regenerate segments file from fullref.rttm
            #  $2: recid, $4: start_time, $5: duration, $8: speakerid
            awk '{printf "%s_%s_%07d_%07d %s %.2f %.2f\n", \
                 $2, $8, $4*100, ($4+$5)*100, $2, $4, $4+$5}' \
                $save_dir/callhome/fullref.rttm | sort > $save_dir/${dset}_spk${num_spk}/segments
            utils/fix_data_dir.sh $save_dir/${dset}_spk${num_spk}
            # Speaker ID is '[recid]_[speakerid]
            awk '{split($1,A,"_"); printf "%s %s_%s\n", $1, A[1], A[2]}' \
                $save_dir/${dset}_spk${num_spk}/segments > $save_dir/${dset}_spk${num_spk}/utt2spk
            utils/fix_data_dir.sh $save_dir/${dset}_spk${num_spk}
            # Generate rttm files for scoring
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                $save_dir/${dset}_spk${num_spk}/utt2spk $save_dir/${dset}_spk${num_spk}/segments \
                $save_dir/${dset}_spk${num_spk}/rttm
            utils/data/get_reco2dur.sh $save_dir/${dset}_spk${num_spk}
        done
    fi
fi



if [ $stage -le 1 ]; then
    # compose eval/callhome2_spk${num_spk}
    eval_set=$save_dir/eval/callhome2_spk${num_spk}
    if ! validate_data_dir.sh --no-text --no-feats $eval_set; then
        utils/copy_data_dir.sh $save_dir/callhome2_spk${num_spk} $eval_set
        cp $save_dir/callhome2_spk${num_spk}/rttm $eval_set/rttm
        awk -v dstdir=$save_dir/wav/eval/callhome2_spk${num_spk} '{print $1, dstdir"/"$1".wav"}' $save_dir/callhome2_spk${num_spk}/wav.scp > $eval_set/wav.scp
        mkdir -p $save_dir/wav/eval/callhome2_spk${num_spk}
        wav-copy scp:$save_dir/callhome2_spk${num_spk}/wav.scp scp:$eval_set/wav.scp
        utils/data/get_reco2dur.sh $eval_set
        rm -rf $save_dir/callhome2_spk${num_spk}
    fi

    # compose eval/callhome1_spk${num_spk}
    adapt_set=$save_dir/eval/callhome1_spk${num_spk}
    if ! validate_data_dir.sh --no-text --no-feats $adapt_set; then
        utils/copy_data_dir.sh $save_dir/callhome1_spk${num_spk} $adapt_set
        cp $save_dir/callhome1_spk${num_spk}/rttm $adapt_set/rttm
        awk -v dstdir=$save_dir/wav/eval/callhome1_spk${num_spk} '{print $1, dstdir"/"$1".wav"}' $save_dir/callhome1_spk${num_spk}/wav.scp > $adapt_set/wav.scp
        mkdir -p $save_dir/wav/eval/callhome1_spk${num_spk}
        wav-copy scp:$save_dir/callhome1_spk${num_spk}/wav.scp scp:$adapt_set/wav.scp
        utils/data/get_reco2dur.sh $adapt_set
        rm -rf $save_dir/callhome1_spk${num_spk}
    fi
fi
