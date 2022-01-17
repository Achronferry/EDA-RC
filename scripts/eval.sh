#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

dataset=LibriSpeech
num_speaker=3
model_name=EEND_EDA
stage=0
# max_epoch=50
exp_dir=
suffix=
# exp_dir=exp/LibriSpeech_2/Transformer_RP/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_50000_20210703185052


. utils/parse_options.sh || exit 1;

conf_dir=conf/$model_name
feature_conf=$conf_dir/feature.yaml
train_conf=$conf_dir/train.yaml
infer_conf=$conf_dir/infer.yaml


if [ "$exp_dir" = "" ];then  
    conf_mark=`awk '/^batchsize|^lr|^num_frames|^gradclip|^noam_warmup_steps/ {sub(/: /,"_"); print $1} /^optimizer/ {print $2}' $train_conf |
                paste -s -d '_'`
    # date_suf=`date +%Y%m%d%H%M%S`
    exp_dir=exp/${dataset}_${num_speaker}/$model_name/${conf_mark}${suffix}
fi
max_epoch=`awk '/max_epochs/ {print $2}' $train_conf`
cpd_mode=`awk '/change_mode/ {print $2}' $infer_conf`

train_dir=
dev_dir=
if [ $dataset = "mini_LibriSpeech" ]; then
    train_dir=data/$dataset/data/train_clean_5_ns${num_speaker}_beta2_2000
    dev_dir=data/$dataset/data/dev_clean_2_ns${num_speaker}_beta2_500
elif [ $dataset = "LibriSpeech" ]; then
    train_dir=data/$dataset/data/train_clean_360_ns${num_speaker}_beta2_100000
    dev_dir=data/$dataset/data/dev_clean_ns${num_speaker}_beta2_500
elif [ $dataset = "callhome" ]; then
    train_dir=data/$dataset/eval/callhome1_spk${num_speaker}
    dev_dir=data/$dataset/eval/callhome2_spk${num_speaker}
else
    echo "Illegal dataset name:${dataset}!" && exit
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
infer_out_dir=$exp_dir/infer/simu/$cpd_mode

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/$cpd_mode

# Training

if [ $stage -le 1 ]; then
    checkpoint=0
    if [ -d "$model_dir" ]; then
      for file in `ls $model_dir | grep '.th'`
      do
            if [[ `echo "$file" | tr -cd "[0-9]"` -ge "$checkpoint" ]]; then
                    checkpoint=`echo "$file" | tr -cd "[0-9]"`
            fi
      done
    fi
    echo "Evaluate at epoch {$checkpoint}"
    python run/eval.py -c $train_conf -f $feature_conf $train_dir $dev_dir $model_dir --max-epochs $max_epoch --num-speakers $num_speaker --resume $checkpoint 
fi

