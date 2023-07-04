#!/usr/bin/env bash
# Copyright   2020   Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.
#
# This recipe performs diarization for the mix-headset data in the
# AMI dataset. The x-vector extractor we use is trained on VoxCeleb v2 
# corpus with simulated RIRs. We use oracle SAD in this recipe.
# This recipe demonstrates the following:
# 1. Diarization using x-vector and clustering (AHC, VBx, spectral)
# 2. Training an overlap detector (using annotations) and corresponding
# inference on full recordings.

# We do not provide training script for an x-vector extractor. You
# can download a pretrained extractor from:
# http://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz
# and extract it.

stage=0
nj=50
decode_nj=15
data_dir=data/ami

train_set=train
test_sets="dev test"
AMI_DIR=/GPFS/public/AMI/amicorpus # Default,


. utils/parse_options.sh

tmp_dir=$data_dir/ami
# Path where AMI gets downloaded (or where locally available):


# # Download AMI corpus, You need around 130GB of free space to get whole data
# if [ $stage -le 1 ]; then
#   if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
#     echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
#     echo " ... Assuming the data does not need to be downloaded.  Please use --stage 2 or more."
#     exit 1
#   fi
#   if [ -e data/local/downloads/wget_$mic.sh ]; then
#     echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
#     exit 1
#   fi
#   preprocess/ami_download.sh $mic $AMI_DIR
# fi

# Prepare data directories. 
if [ $stage -le 1 ]; then
  # Download the data split and references from BUT's AMI setup
  if ! [ -d AMI-diarization-setup ]; then
    git clone https://github.com/BUTSpeechFIT/AMI-diarization-setup
  fi

  for dataset in train $test_sets; do
    echo "$0: preparing $dataset set.."
    mkdir -p $tmp_dir/$dataset
    # Prepare wav.scp and segments file from meeting lists and oracle SAD
    # labels, and concatenate all reference RTTMs into one file.
    python preprocess/prepare_ami_data.py --sad-labels-dir AMI-diarization-setup/only_words/labs/${dataset} \
      AMI-diarization-setup/lists/${dataset}.meetings.txt \
      $AMI_DIR $tmp_dir/$dataset
    cat AMI-diarization-setup/only_words/rttms/${dataset}/*.rttm \
      > $tmp_dir/${dataset}/rttm.annotation

    awk '{print $1,$2}' $tmp_dir/$dataset/segments > $tmp_dir/$dataset/utt2spk
    utils/utt2spk_to_spk2utt.pl $tmp_dir/$dataset/utt2spk > $tmp_dir/$dataset/spk2utt
    utils/fix_data_dir.sh $tmp_dir/$dataset
  done
fi