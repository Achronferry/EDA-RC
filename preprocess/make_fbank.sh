#!/usr/bin/env bash

# Copyright 2012-2016  Karel Vesely
# Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
fbank_config=conf/fbank.conf
compress=true
write_utt2num_frames=true  # If true writes utt2num_frames.
write_utt2dur=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
 e.g.: $0 data/train
Note: <log-dir> defaults to <data-dir>/log, and
      <fbank-dir> defaults to <data-dir>/data
Options:
  --fbank-config <config-file>         # config passed to compute-fbank-feats.
  --nj <nj>                            # number of parallel jobs.
  --cmd <run.pl|queue.pl <queue opts>> # how to run jobs.
  --write-utt2num-frames <true|false>  # If true, write utt2num_frames file.
  --write-utt2dur <true|false>         # If true, write utt2dur file.
EOF
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  fbankdir=$3
else
  fbankdir=$data/data
fi


# make $fbankdir an absolute pathname.
fbankdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbankdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbankdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $fbank_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

# utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi

for n in $(seq $nj); do
  # the next command does nothing unless $fbankdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $fbankdir/raw_fbank_$name.$n.ark
done

if $write_utt2num_frames; then
  write_num_frames_opt="--write-num-frames=ark,t:$logdir/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if $write_utt2dur; then
  write_utt2dur_opt="--write-utt2dur=ark,t:$logdir/utt2dur.JOB"
else
  write_utt2dur_opt=
fi



echo "$0: [info]: No use of segments, process the total wav.scp."
split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/wav.$n.scp"
 done
 utils/split_scp.pl $scp $split_scps || exit 1;
 $cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
   compute-fbank-feats $vtln_opts $write_utt2dur_opt --verbose=2 \
    --config=$fbank_config scp,p:$logdir/wav.JOB.scp ark:- \| \
   copy-feats --compress=$compress $write_num_frames_opt ark:- \
    ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
    || exit 1;



if [ -f $logdir/.error.$name ]; then
  echo "$0: Error producing filterbank features for $name:"
  tail $logdir/make_fbank_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $fbankdir/raw_fbank_$name.$n.scp || exit 1
done > $data/feats.scp || exit 1

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/utt2num_frames.$n || exit 1
  done > $data/utt2num_frames || exit 1
fi

if $write_utt2dur; then
  for n in $(seq $nj); do
    cat $logdir/utt2dur.$n || exit 1
  done > $data/utt2dur || exit 1
fi

# Store frame_shift and fbank_config along with features.
frame_shift=$(perl -ne 'if (/^--frame-shift=(\d+)/) {
                          printf "%.3f", 0.001 * $1; exit; }' $fbank_config)
echo ${frame_shift:-'0.01'} > $data/frame_shift
mkdir -p $data/conf && cp $fbank_config $data/conf/fbank.conf || exit 1

rm $logdir/wav_${name}.*.scp  $logdir/segments.* \
   $logdir/utt2num_frames.* $logdir/utt2dur.* 2>/dev/null

nf=$(wc -l < $data/feats.scp)
nu=$(wc -l < $data/wav.scp)
if [ $nf -ne $nu ]; then
  echo "$0: It seems not all of the feature files were successfully procesed" \
       "($nf != $nu); consider using utils/fix_data_dir.sh $data"
fi

if (( nf < nu - nu/20 )); then
  echo "$0: Less than 95% the features were successfully generated."\
       "Probably a serious error."
  exit 1
fi

echo "$0: Succeeded creating filterbank features for $name"
