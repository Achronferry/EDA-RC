#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

dataset_name=callhome
num_speaker=2
model_name=EEND
stage=0
max_epoch=50
exp_dir=
# exp_dir=exp/LibriSpeech_2/Transformer_RP/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_50000_20210703185052


. utils/parse_options.sh || exit 1;

conf_dir=conf/$model_name
feature_conf=$conf_dir/feature.yaml
train_conf=$conf_dir/train.yaml
infer_conf=$conf_dir/infer.yaml


if [ "$exp_dir" = "" ];then  
    conf_mark=`awk '/batchsize|lr|num_frames|gradclip|noam_warmup_steps/ {sub(/: /,"_"); print $1} /optimizer/ {print $2}' $train_conf |
                paste -s -d '_'`
    # date_suf=`date +%Y%m%d%H%M%S`
    exp_dir=exp/${dataset_name}_${num_speaker}/$model_name/${conf_mark} #_${date_suf}
fi

train_dir=
dev_dir=
if [ $dataset_name = "mini_LibriSpeech" ]; then
    train_dir=data/$dataset_name/data/train_clean_5_ns${num_speaker}_beta2_2000
    dev_dir=data/$dataset_name/data/dev_clean_2_ns${num_speaker}_beta2_500
elif [ $dataset_name = "LibriSpeech" ]; then
    train_dir=data/$dataset_name/data/train_clean_360_ns${num_speaker}_beta2_100000
    dev_dir=data/$dataset_name/data/dev_clean_ns${num_speaker}_beta2_500
elif [ $dataset_name = "callhome" ]; then
    train_dir=data/$dataset_name/eval/callhome1_spk${num_speaker}
    dev_dir=data/$dataset_name/eval/callhome2_spk${num_speaker}
else
    echo "Illegal dataset name:${dataset_name}!" && exit
fi

model_dir=$exp_dir/models


#train_adapt_dir=data/eval/callhome1_spk2
# dev_adapt_dir=data/eval/callhome2_spk2
# model_adapt_dir=$exp_dir/models_adapt
# adapt_conf=$conf_dir/adapt.yaml
avg_model=$model_dir/avg.th

#test_dir=data/eval/callhome2_spk2
# test_model=$model_adapt_dir/avg.th
# infer_out_dir=$exp_dir/infer/callhome
# test_dir=data/simu/data/swb_sre_cv_ns2_beta2_500
# test_model=$model_dir/avg.th
infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score

# Training

if [ $stage -le 1 ]; then
    checkpoint=0
    python run/dataset_test.py $train_dir $dev_dir --num-speakers $num_speaker
fi

exit 1
echo "Finished !"
