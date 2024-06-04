# Non-Linear Inference Time Intervention: Improving LLM Truthfulness

This repo contains code from the NL-ITI paper.

## Abstract

In this work, we explore LLM’s internal representation space to identify attention heads that contain the most truthful and accurate information. Inference Time Intervention (ITI) framework, which lets bias LLM without the need for fine-tuning. We further developed the ITI framework by introducing a nonlinear multi-token probing and multi-token intervention: Non-Linear ITI (NL-ITI), which significantly improves performance on evaluation benchmarks. NL-ITI is tested on diverse multiplechoice datasets, including TruthfulQA, on which we report over 16% relative MC1 metric improvement with respect to the baseline ITI  results. Moreover, we achieved a 10% relative improvement over the recently released Truth Forest (TrFf) method that also focused on ITI improvement.

## Setup

Use `requirements.txt` to setup a python venv in which you will run the code.

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Usage step by step

### Collecting activations

First, you need to generate activations with 'qa' and 'endq' formats. The more tokens the better, as you can later use up to this number of tokens. RAM is your main limiting factor.

```
python3 get_activations.py --dataset_name truthful_qa --dataset_split valid --prompt_format qa --n_last_tokens 2
```

The script uses a single GPU, so if you have more of them, you can run this this in parallel like that (just using 0, 1, etc... for each GPU that you have).

```
CUDA_VISIBLE_DEVICES=0 python3 get_activations.py --dataset_name truthful_qa --dataset_split valid --prompt_format qa --n_last_tokens 2
```

### Probing

As a second step, you want to run probing to get scores for each head. These will be later used for selecting top heads for intervention.
```
python3 probing.py --dataset_name truthful_qa --dataset_split valid --probe_type mlp --n_last_tokens 2 [--ray_tune]
```
For faster calculation, you can type optional `--ray_tune` argument that will use parallelization. With that, there are two optional arguments that work, `--ray_cpu` and `--ray_gpu` used to manipulate the proportion of CPU and GPU used for training of single probing model.

The command above is a good choice if you are probing and intervening on a 'train' split and evaluating on a 'test' split. But for TruthfulQA, there is only a single split. In this case you can run calculation in folds, so training and test observations don't overlap.

```
python3 probing.py --dataset_name truthful_qa --dataset_split valid --probe_type mlp --n_last_tokens 2 --fold 1 [--ray_tune]
python3 probing.py --dataset_name truthful_qa --dataset_split valid --probe_type mlp --n_last_tokens 2 --fold 2 [--ray_tune]
```

The output will be found in `probings' dir. Just look up there to get the probing name which will be needed next.

### Intervention

Intervention is the final step. It will produce results that you can find later in `results` dir.

```
python3 intervention.py --probing_name truthful_qa_valid_mlp_s_last2_epochs10 --activations_direction truthful_qa --activations_std truthful_qa --activations_split valid --evaluation_dataset truthful_qa --evaluation_dataset_split valid --n_last_tokens 2 [-ht]
```

The basic case above works best if you evaluate LLM on unseen 'test' split. If you decided to run calculation in folds, you intervene on each fold separately.
```
python3 intervention.py --probing_name truthful_qa_valid_mlp_s_last2_epochs10_f1 --activations_direction truthful_qa --activations_std truthful_qa --activations_split valid --evaluation_dataset truthful_qa --evaluation_dataset_split valid --n_last_tokens 2 --fold 2 [-ht]
python3 intervention.py --probing_name truthful_qa_valid_mlp_s_last2_epochs10_f2 --activations_direction truthful_qa --activations_std truthful_qa --activations_split valid --evaluation_dataset truthful_qa --evaluation_dataset_split valid --n_last_tokens 2 --fold 1 [-ht]
```
By default, the intervention is run on a single hyperparameters set, that you can choose using `--heads` and `--alpha` arguments. If you use `-ht` argument, the calculation will be run on a set of hyperparamters and will take longer as well.

### Merging fold results (optional)

If you run the whole calculation in folds, there is a script to merge the results from both folds.

```
python3
>>> import utils
>>> utils.merge_fold_results('results_name_f1', 'results_name_f2')
```
It requires just giving the names of result files. It will join them into one.

### Collecting detailed results (optional)

For datasets with categories (MMLU), you can collect detailed (per category) intervention results, using `--detailed` option. It's intended to use with specific `--head` and `--alpha`.

It will additionaly drop a csv named `detailed_results.csv`

## Reproducing paper's results

To reproduce results from the paper, you can look into `reproduce_table1_commands` and `reproduce_table2_commands` files,
as they contain exact commands needed to calculate the scores.

## Original paper's results

You can find original paper results in `results_paper` dir. They were collected with hyperparameter tuning.
