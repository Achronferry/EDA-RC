#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

model_type=$1
model_one=$2
model_two=$3
model_avg=$4

dataset_name=$5

if [ $dataset_name = "mini_LibriSpeech" ]; then
    dev_adapt_dir=data/$dataset_name/data/dev_clean_2_ns2_beta2_500
elif [ $dataset_name = "LibriSpeech" ]; then
    dev_adapt_dir=data/$dataset_name/data/dev_clean_ns2_beta2_500
elif [ $dataset_name = "callhome" ]; then
    dev_adapt_dir=data/$dataset_name/eval/callhome2_spk2
else
    echo "Illegal dataset name:${dataset_name}!" && exit
fi

infer_conf=conf/$model_type/infer.yaml

# Inferring
echo "[]Start inferring model_one:$model_one"
mkdir -p .compare/infer_one
python run/infer.py -c $infer_conf $dev_adapt_dir $model_one .compare/infer_one

echo "[]Start inferring model_two:$model_two"
mkdir -p .compare/infer_two
python run/infer.py -c $infer_conf $dev_adapt_dir $model_two .compare/infer_two

echo "[]Start inferring model_avg:$model_avg"
mkdir -p .compare/infer_avg
python run/infer.py -c $infer_conf $dev_adapt_dir $model_avg .compare/infer_avg

# Scoring
echo "[]Start scoring model one:"
mkdir -p .compare/score_one
find .compare/infer_one -iname "*.h5" > .compare/infer_one/file_list
for med in 1 11; do
for th in 0.3 0.4 0.5 0.6 0.7; do
python run/make_rttm.py --median=$med --threshold=$th \
	--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
	.compare/infer_one/file_list .compare/score_one/hyp_${th}_$med.rttm
done
done

echo "[]Start scoring model two:"
mkdir -p .compare/score_two
find .compare/infer_two -iname "*.h5" > .compare/infer_two/file_list
for med in 1 11; do
for th in 0.3 0.4 0.5 0.6 0.7; do
python run/make_rttm.py --median=$med --threshold=$th \
	--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
	.compare/infer_two/file_list .compare/score_two/hyp_${th}_$med.rttm
done
done

echo "[]Start scoring model avg:"
mkdir -p .compare/score_avg
find .compare/infer_avg -iname "*.h5" > .compare/infer_avg/file_list
for med in 1 11; do
for th in 0.3 0.4 0.5 0.6 0.7; do
python run/make_rttm.py --median=$med --threshold=$th \
	--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
	.compare/infer_avg/file_list .compare/score_avg/hyp_${th}_$med.rttm
done
done

. path.sh

mkdir -p .compare/score_two_over_one
mkdir -p .compare/score_one_over_avg
mkdir -p .compare/score_two_over_avg
for med in 1 11; do
for th in 0.3 0.4 0.5 0.6 0.7; do
md-eval.pl -c 0.25 -r $dev_adapt_dir/rttm -s .compare/score_one/hyp_${th}_$med.rttm > .compare/score_one/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
md-eval.pl -c 0.25 -r $dev_adapt_dir/rttm -s .compare/score_two/hyp_${th}_$med.rttm > .compare/score_two/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
md-eval.pl -c 0.25 -r $dev_adapt_dir/rttm -s .compare/score_avg/hyp_${th}_$med.rttm > .compare/score_avg/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
md-eval.pl -c 0.25 -r .compare/score_one/hyp_${th}_$med.rttm -s .compare/score_two/hyp_${th}_$med.rttm > .compare/score_two_over_one/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
md-eval.pl -c 0.25 -r .compare/score_avg/hyp_${th}_$med.rttm -s .compare/score_one/hyp_${th}_$med.rttm > .compare/score_one_over_avg/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
md-eval.pl -c 0.25 -r .compare/score_avg/hyp_${th}_$med.rttm -s .compare/score_two/hyp_${th}_$med.rttm > .compare/score_two_over_avg/result_th${th}_med${med}_collar0.25 2>/dev/null || exit

done
done

echo "model one / dev:"
scripts/best_score.sh .compare/score_one
echo "model two / dev:"
scripts/best_score.sh .compare/score_two
echo "model avg / dev:"
scripts/best_score.sh .compare/score_avg
echo "model two / model one:"
scripts/best_score.sh .compare/score_two_over_one
echo "model one / model avg:"
scripts/best_score.sh .compare/score_one_over_avg
echo "model two / model avg:"
scripts/best_score.sh .compare/score_two_over_avg
echo "Finished !"