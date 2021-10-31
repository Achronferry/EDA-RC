 EEND_LSTM



### Introduction



### Steps

**1. Set environment**

- Change `YOUR_KALDI_ROOT` in `/scripts/set_env.sh`. 
- Run:

```bash
./scripts/set_env.sh
```

 

**2.Prepare_dataset**

`$dataset_name` should be one of `[LibriSpeech, mini_LibriSpeech, callhome]`.

-   Change `dataset_dir` in (`/scripts/prepare_librispeech.sh` or `/scripts/prepare_callhome.sh`) to your local directory.
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

### results
dataset stats
|    dataset  |  train #     |   train overlap %  |   dev #  |  dev overlap %  | speaker # | real/simu |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   callhome2   |    |    | | | 2 | real |
|   LibriSpeech_2   |   100000  |      |  500 | | 2 | simu |
|   LibriSpeech_3   |   100000  |      | 500 |  | 3 | simu |


train on LibriSpeech_2
|    model  |  LS2      |   CH2   |
| ---- | ---- | ---- |
|   EEND   |   3.08  |  10.81    |


train on LibriSpeech_3
oracle training strategy
|    model  |  LS3      |   CH3   |   CHv   |
| ---- | ---- | ---- | ---- |
|   EEND   |   6.78  |   16.81   |      |
|   EENDC(oracle)  |   3.00  |      |      |
|   EENDC(refine) +num_pred   |   5.86  |      |      |
|   EENDC(oracle) +num_pred   |   2.43  |   9.24   |      |
|   EENDC(fix_len_3) +num_pred   |   24.27  |    30.11  |      |
|   EENDC(fix_len_10) +num_pred   |   10.05  |   21.62   |      |
|   EENDC(fix_len_12) +num_pred   |   9.59  |   22.97   |      |
|   EENDC(fix_len_15) +num_pred   |   9.39  |  25.02    |      |
|   EENDC(fix_len_20) +num_pred   |   10.82  |   27.34   |      |

### TODO

- [ ] 代码翻新
- [ ] 




