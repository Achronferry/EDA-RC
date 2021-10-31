#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

dataset_name=LibriSpeech
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
    train_dir=data/$dataset_name/data/train_clean_5_ns${num_speaker}_beta2_500
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
    if [ -d "$model_dir" ]; then
      for file in `ls $model_dir | grep '.th'`
      do
            if [ `echo "$file" | tr -cd "[0-9]"` -ge "$checkpoint" ]; then
                    checkpoint=`echo "$file" | tr -cd "[0-9]"`
            fi
      done
    fi
    echo "Start training from epoch {$checkpoint}"
    python run/train.py -c $train_conf -f $feature_conf $train_dir $dev_dir $model_dir --max-epochs $max_epoch --resume $checkpoint
    cp $feature_conf $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    st=`expr $max_epoch / 10 \* 9 + 1`
    ifiles=`eval echo $model_dir/transformer{$st..$max_epoch}.th`
    python run/model_averaging.py $avg_model $ifiles
fi

# Adapting
#if [ $stage -le 3 ]; then
#    echo "Start adapting"
#    python run/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model
#fi
#
## Model averaging
#if [ $stage -le 4 ]; then
#    echo "Start model averaging"
#    ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
#    python eend/bin/model_averaging.py $test_model $ifiles
#fi

# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
    python run/infer.py -c $infer_conf -c2 $model_dir/param.yaml -f $feature_conf $dev_dir $avg_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	subsample_rate=`awk '/subsampling/ {print $2}' $model_dir/feature.yaml`
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python run/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=$subsample_rate --sampling_rate=8000 \
		$work/file_list $scoring_dir/hyp_${th}_$med.rttm
	done
	done
	. path.sh
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	md-eval.pl -c 0.25 -r $dev_dir/rttm -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

scripts/best_score.sh $scoring_dir | tee $scoring_dir/final_res.txt

echo "Finished !"
