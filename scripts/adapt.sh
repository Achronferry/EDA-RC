#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

pretrained_set=LibriSpeech_2
model_name=EEND
train_adapt_dir=data/callhome/eval/callhome1_spk2
dev_adapt_dir=data/callhome/eval/callhome2_spk2
max_epoch=100
stage=0



. utils/parse_options.sh || exit 1;

pretrain_exp_dir=exp/${pretrained_set}/${model_name}/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_100000

pretrain_model=$pretrain_exp_dir/models/avg.th

model_conf=$pretrain_exp_dir/models/param.yaml
adapt_conf=conf/${model_name}/adapt.yaml
feature_conf=conf/${model_name}/feature.yaml
infer_conf=conf/${model_name}/infer.yaml

adapt_mark=`awk '/batchsize|lr|num_frames|gradclip|noam_warmup_steps/ {sub(/: /,"_"); print $1} /optimizer/ {print $2}' $adapt_conf |
                paste -s -d '_'`
exp_dir=exp/adapt_callhome/${model_name}_${pretrained_set}/$adapt_mark
model_adapt_dir=$exp_dir/models_adapt

avg_model=$model_adapt_dir/avg.th

infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score



# Adapting
if [ $stage -le 1 ]; then
    echo "Start adapting"
    python run/train.py -c $adapt_conf -c2 $model_conf -f $feature_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir \
                        --initmodel $pretrain_model --max-epochs $max_epoch
fi


# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    st=`expr $max_epoch / 10 \* 9 + 1`
    ifiles=`eval echo $model_adapt_dir/transformer{$st..$max_epoch}.th`
    python run/model_averaging.py $avg_model $ifiles
fi

# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
	python run/infer.py -c $infer_conf -c2 $model_conf -f $feature_conf  $dev_adapt_dir $avg_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	subsample_rate=`awk '/subsampling/ {print $2}' $feature_conf`
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
	md-eval.pl -c 0.25 -r $dev_adapt_dir/rttm -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

scripts/best_score.sh $scoring_dir | tee $scoring_dir/final_res.txt

echo "Finished !"