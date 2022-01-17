#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

model_name=EEND_EDA
num_speaker=3
stage=0

pretrain_exp_dir=
suffix=

. utils/parse_options.sh || exit 1;
dev_adapt_dir=data/callhome/eval/callhome2_spk${num_speaker}

pretrain_model=$pretrain_exp_dir/adapt_callhome_3/gradclip_5_batchsize_32_num_frames_500_adam_lr_1e-4/models_adapt/avg.th

model_conf=$pretrain_exp_dir/adapt_callhome_3/gradclip_5_batchsize_32_num_frames_500_adam_lr_1e-4/models_adapt/param.yaml
adapt_conf=conf/${model_name}/adapt.yaml
feature_conf=conf/${model_name}/feature.yaml
infer_conf=conf/${model_name}/infer.yaml

adapt_mark=`awk '/^batchsize|^lr|^num_frames|^gradclip|^noam_warmup_steps/ {sub(/: /,"_"); print $1} /^optimizer/ {print $2}' $adapt_conf |
                paste -s -d '_'`

exp_dir=$pretrain_exp_dir/adapt_callhome_${num_speaker}/$adapt_mark$suffix
max_epoch=`awk '/max_epochs/ {print $2}' $adapt_conf`
cpd_mode=`awk '/change_mode/ {print $2}' $infer_conf`
model_adapt_dir=$exp_dir/models_adapt


infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score




# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
	python run/infer_v2.py -c $infer_conf -c2 $model_conf -f $feature_conf $dev_adapt_dir $pretrain_model $infer_out_dir --num-speakers $num_speaker
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	subsample_rate=`awk '/subsampling/ {print $2}' $feature_conf`
	for med in 1; do
	for th in 0.5; do
	python run/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=$subsample_rate --sampling_rate=8000 \
		$work/file_list $scoring_dir/hyp_${th}_$med.rttm
	done
	done
	. path.sh
	for med in 1; do
	for th in 0.5; do
	md-eval.pl -c 0.25 -r $dev_adapt_dir/rttm -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi

scripts/best_score.sh $scoring_dir | tee $scoring_dir/final_res.txt

echo "Finished !"