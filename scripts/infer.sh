#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

model=EDA_RC
num_speaker=4
stage=0

pretrain_exp_dir=
suffix=
dev_adapt_dir=data/LibriSpeech/data/dev_clean_ns4_beta12_500

. utils/parse_options.sh || exit 1;


pretrain_model=$pretrain_exp_dir/models/avg.th
model_conf=$pretrain_exp_dir/models/param.yaml

feature_conf=conf/${model}/feature.yaml
infer_conf=conf/${model}/infer.yaml

exp_dir=$pretrain_exp_dir/eval_${dev_adapt_dir##*/}$suffix

infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score


# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
	python run/infer.py -c $infer_conf -c2 $model_conf -f $feature_conf $dev_adapt_dir $pretrain_model $infer_out_dir --num-speakers $num_speaker
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