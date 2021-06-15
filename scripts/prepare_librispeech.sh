#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options
# This script does NOT include the composition of train/valid/test sets.
# The composition will be done at stage 1 of ./run.sh

stage=0
dataset_name=$1 #(LibriSpeech, mini_LibriSpeech)
dataset_dir=~/data_store/$dataset_name

# This script distributes simulated data under these directories
simu_actual_dirs=(
$PWD/data/local/diarization-data/$dataset_name
)

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker=4
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=100000
simu_opts_min_utts=10
simu_opts_max_utts=20

. cmd.sh
. path.sh
. parse_options.sh || exit

dev_nm=
trn_nm=
if [ $dataset_name = "mini_LibriSpeech" ]; then
    dev_nm=dev_clean_2
    trn_nm=train_clean_5
elif [ $dataset_name = "LibriSpeech" ]; then
    dev_nm=dev_clean
    trn_nm=train_clean_100
else
    echo "Illegal dataset name:${dataset_name}!" && exit
fi


if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
#    mini_librispeech_url=http://www.openslr.org/resources/31
    mkdir -p data/local
#    local/download_and_untar.sh data/local $mini_librispeech_url  dev-clean-2
#    local/download_and_untar.sh data/local $mini_librispeech_url train-clean-5
    if [ ! -f data/local/prepared_data/$dataset_name/$dev_nm/.done ]; then
        scripts/data_prep.sh $dataset_dir/${dev_nm//_/-} data/local/prepared_data/$dataset_name/$dev_nm || exit
        touch data/local/prepared_data/$dataset_name/$dev_nm/.done
    fi
    if [ ! -f data/local/prepared_data/$dataset_name/$trn_nm/.done ]; then
        scripts/data_prep.sh $dataset_dir/${trn_nm//_/-} data/local/prepared_data/$dataset_name/$trn_nm
        touch data/local/prepared_data/$dataset_name/$trn_nm/.done
    fi
    if [ ! -d data/musan_bgnoise ]; then
        echo "- unzip mini-musan-bgnoise"
        tar xzf musan_bgnoise.tar.gz data/
    fi
    if [ ! -f data/simu_rirs_8k/.done ]; then
        echo "- prepare simu_rirs_8k"
        mkdir -p data/simu_rirs_8k
        if [ ! -e sim_rir_8k.zip ]; then
            echo "-- downloading..."
            wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_8k.zip
        fi
        unzip sim_rir_8k.zip -d data/local/sim_rir_8k
        find $PWD/data/local/sim_rir_8k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/simu_rirs_8k/wav.scp
        awk '{print $1, $1}' data/simu_rirs_8k/wav.scp > data/simu_rirs_8k/utt2spk
        utils/fix_data_dir.sh data/simu_rirs_8k
        touch data/simu_rirs_8k/.done
    fi
fi

simudir=data/$dataset_name
if [ $stage -le 1 ]; then
    echo "simulation of mixture"
    mkdir -p $simudir/.work
    random_mixture_cmd=preprocess/random_mixture_nooverlap.py
    make_mixture_cmd=preprocess/make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=preprocess/random_mixture.py
        make_mixture_cmd=preprocess/make_mixture.py
    fi

    for simu_opts_sil_scale in 2; do
        for dset in $trn_nm $dev_nm; do
            if [ "$dset" == "train_clean_100" ]; then
                n_mixtures=${simu_opts_num_train}
            else
                n_mixtures=500
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $simu_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    data/local/prepared_data/$dataset_name/$dset data/musan_bgnoise data/simu_rirs_8k \
                    \> $simudir/.work/mixture_$simuid.scp
                nj=100
                mkdir -p $simudir/wav/$simuid
                # distribute simulated data to $simu_actual_dir
                split_scps=
                for n in $(seq $nj); do
                    split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
                    mkdir -p $simudir/.work/data_$simuid.$n
                    actual=${simu_actual_dirs[($n-1)%${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
                    mkdir -p $actual
                    ln -nfs $actual $simudir/wav/$simuid/$n
                done
                utils/split_scp.pl $simudir/.work/mixture_$simuid.scp $split_scps || exit 1

                $simu_cmd --max-jobs-run 32 JOB=1:$nj $simudir/.work/make_mixture_$simuid.JOB.log \
                    $make_mixture_cmd --rate=8000 \
                    $simudir/.work/mixture_$simuid.JOB.scp \
                    $simudir/.work/data_$simuid.JOB $simudir/wav/$simuid/JOB
                utils/combine_data.sh $simudir/data/$simuid $simudir/.work/data_$simuid.*
                steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    $simudir/data/$simuid/utt2spk $simudir/data/$simuid/segments \
                    $simudir/data/$simuid/rttm
                utils/data/get_reco2dur.sh $simudir/data/$simuid
            fi
        done
    done
fi
