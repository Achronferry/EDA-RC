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
dataset_name=LibriSpeech #(LibriSpeech, mini_LibriSpeech)
musan_root=~/data_store/musan

# This script distributes simulated data under these directories
simu_actual_dirs=(
/GPFS/data/chenyuyang/exp/RPE_EEND/local/diarization-data/$dataset_name
)

# simulation options
simu_opts_overlap=yes
simu_opts_num_speaker=3
simu_opts_sil_scale=8
simu_opts_rvb_prob=0.5
simu_opts_num_train=30000
simu_opts_min_utts=10 # 3: 10~20 4: 8~15 5: 6~12 
simu_opts_max_utts=20

. cmd.sh
. path.sh
. parse_options.sh || exit

dataset_dir=~/data_store/$dataset_name

dev_nm=
trn_nm=
if [ $dataset_name = "mini_LibriSpeech" ]; then
    dev_nm=dev_clean_2
    trn_nm=train_clean_5
elif [ $dataset_name = "LibriSpeech" ]; then
    dev_nm=dev_clean
    trn_nm=train_clean_360
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

    if ! validate_data_dir.sh --no-text --no-feats data/musan_noise_bg; then
        preprocess/make_musan.sh $musan_root data
        utils/copy_data_dir.sh data/musan_noise data/musan_noise_bg
        awk '{if(NR>1) print $1,$1}'  $musan_root/noise/free-sound/ANNOTATIONS > data/musan_noise_bg/utt2spk
        utils/fix_data_dir.sh data/musan_noise_bg
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

    for simu_opts_sil_scale in $simu_opts_sil_scale; do
        for dset in $trn_nm $dev_nm; do
        # for dset in $trn_nm; do
            if [ "$dset" == "dev_clean" ] || [ "$dset" == "dev_clean_2" ] ; then
                n_mixtures=500
            else
                n_mixtures=${simu_opts_num_train}
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $simu_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    --min_utts $simu_opts_min_utts --max_utts $simu_opts_max_utts \
                    data/local/prepared_data/$dataset_name/$dset data/musan_noise_bg data/simu_rirs_8k \
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

if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; 
    for dset in $trn_nm $dev_nm; do
    # for dset in $trn_nm; do
        if [ "$dset" == "dev_clean" ]; then
                n_mixtures=500
            else
                n_mixtures=${simu_opts_num_train}
        fi
        simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
        preprocess/make_fbank.sh --cmd "$train_cmd" --nj 12 --write_utt2num_frames true --fbank_config conf/fbank.conf \
            data/${dataset_name}/data/${simuid} exp/make_fbank/${dataset_name}/${simuid} $simu_actual_dirs/fbank/${simuid}
        # utils/fix_data_dir.sh  data/${dataset_name}/data/${simuid}
    done

    # # subset of dev_set
    # utils/subset_data_dir.sh data/${train_dev} 1000 data/${train_dev}_u1k

    # # compute global CMVN
    # compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
fi
