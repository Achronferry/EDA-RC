#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

model=EEND
num_speaker=4
stage=0

pretrain_exp_dir=exp/LibriSpeech_2/EEND/gradclip_5_batchsize_32_num_frames_500_noam_lr_1.0_noam_warmup_steps_100000_4layer_1024ff
suffix=

. utils/parse_options.sh || exit 1;
train_adapt_dir=data/callhome/eval/callhome1_spk234
dev_adapt_dir=data/callhome/eval/callhome2_spk234
# train_adapt_dir=data/LibriSpeech/data/train_clean_360_nsv_90000
# dev_adapt_dir=data/LibriSpeech/data/dev_clean_ns3_beta8_500

pretrain_model=$pretrain_exp_dir/models/avg.th

model_conf=$pretrain_exp_dir/models/param.yaml
adapt_conf=conf/${model}/adapt.yaml
feature_conf=conf/${model}/feature.yaml
infer_conf=conf/${model}/infer.yaml

adapt_mark=`awk '/^batchsize|^lr|^num_frames|^gradclip|^noam_warmup_steps/ {sub(/: /,"_"); print $1} /^optimizer/ {print $2}' $adapt_conf |
                paste -s -d '_'`

exp_dir=$pretrain_exp_dir/adapt_${train_adapt_dir##*/}/$adapt_mark$suffix
max_epoch=`awk '/max_epochs/ {print $2}' $adapt_conf`
cpd_mode=`awk '/change_mode/ {print $2}' $infer_conf`
model_adapt_dir=$exp_dir/models

avg_model=$model_adapt_dir/avg.th

infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score



# Adapting
if [ $stage -le 1 ]; then
    echo "Start adapting"
    python run/train.py -c $adapt_conf -c2 $model_conf -f $feature_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir \
                        --initmodel $pretrain_model --max-epochs $max_epoch --num-speakers $num_speaker
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
	python run/infer.py -c $infer_conf -c2 $model_conf -f $feature_conf $dev_adapt_dir $avg_model $infer_out_dir --num-speakers $num_speaker
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


if [ $stage -le 5 ]; then
    echo "Start curriculum training"
    ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    --num_speaker 2 --dev_adapt_dir data/callhome/eval/callhome2_spk2 --suffix _oracle

    ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    --num_speaker 3 --dev_adapt_dir data/callhome/eval/callhome2_spk3 --suffix _oracle

    ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    --num_speaker 4 --dev_adapt_dir data/callhome/eval/callhome2_spk4 --suffix _oracle

    ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    --num_speaker 5 --dev_adapt_dir data/callhome/eval/callhome2_spk5 --suffix _oracle

    ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    --num_speaker 6 --dev_adapt_dir data/callhome/eval/callhome2_spk6 --suffix _oracle
    # ./scripts/infer.sh --model $model --pretrain_exp_dir $exp_dir \
    # --num_speaker 5 --dev_adapt_dir data/LibriSpeech/data/dev_clean_ns5_beta16_500

fi
echo "Finished !"