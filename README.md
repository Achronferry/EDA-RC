EEND_LSTM

### Introduction

### Steps

**1. Set environment**

- Change`YOUR_KALDI_ROOT` in`/scripts/set_env.sh`.
- Run:

```bash
./scripts/set_env.sh
```

**2.Prepare_dataset**

`$dataset_name` should be one of `[LibriSpeech, mini_LibriSpeech, callhome]`.

- Change`dataset_dir` in (`/scripts/prepare_librispeech.sh` or`/scripts/prepare_callhome.sh`) to your local directory.
- Run:

```bash
./scripts/prepare_librispeech.sh $dataset_name
```

**3. Train/Infer/Score**

`$dataset_name` should be in `[Transformer_Linear, mini_LibriSpeech, callhome]`.

- Run

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" ./scripts/run.sh $dataset_name $model_name
```

**4. Adapt**

- Adapt on callhome1

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" ./scripts/adapt.sh $pretrained_dataset_name $model_name
```





TODO 代码翻新

- [ ]
