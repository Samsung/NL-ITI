import argparse
import copy
import tempfile

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from ray import tune
from ray.air import session
from sklearn.metrics import accuracy_score

import llama
from common import (ACTIVATIONS_DIR, DATASET_NAMES, DATASET_SPLITS, DATASETS_DIR, FOLD_IDS,
                    FORMATS, HF_MODEL_NAMES, RANDOM_QA_DIR, SEED, TOKEN_OPS)
from data_loader import get_fixed_seed_data_loader
from prepare_dataset import load_activations, split_activations
from probes import PROBE_TYPES, get_default_probe_name, probe_factory
from utils import (get_probing_dir, load_probes, make_probing_dir, normalize_values,
                   save_head_accs, save_token_weights, save_probing_model, save_meta_data,
                   save_probes, set_seeds)


EARLY_STOPPING_PATIENCE = 10

TAB = '\t'


def pass_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B', choices=HF_MODEL_NAMES.keys(), help='model name')
    parser.add_argument('--activations_dir', type=str, help='dir where activations were dumped', default=ACTIVATIONS_DIR)
    parser.add_argument('--activations_format', type=str, choices=FORMATS, default='qa')
    parser.add_argument('--activations_intro_prompt', action='store_true', help='use activations collected with intro + few shot QA prompt', default=False)
    parser.add_argument('--dataset_name', type=str, choices=DATASET_NAMES, default='truthful_qa', help='feature bank for training probes')
    parser.add_argument('--dataset_split', type=str, default='valid', choices=DATASET_SPLITS)
    parser.add_argument('--probing_name', type=str, help='name that will be used to save and load the probing results', default=None)
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--ray_tune', action='store_true', help='use parallelization with ray tune', default=False)
    parser.add_argument('--ray_cpu', type=float, default=1, help='fraction of cpu resources per single neural network training')
    parser.add_argument('--ray_gpu', type=float, default=0.05, help='fraction of gpu resources per single neural network training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for neural network to train')
    parser.add_argument('--use_early_stopping', action='store_true', help='use early stopping instead of best weight across all iterations', default=False)
    parser.add_argument('--probe_type', type=str, choices=PROBE_TYPES, help='type of probing model to use', required=True)
    parser.add_argument('--probe_name', type=str, help='name of specific probing model to use', default=None)
    parser.add_argument('--n_last_tokens', type=int, help='number of last tokens to use when taking a mean activation', default=2)
    parser.add_argument('--token_op', type=str, choices=TOKEN_OPS, help='way of merging tokens into single activations vector', default='mean')
    parser.add_argument('--datasets_dir', type=str, default=DATASETS_DIR, help='dir from which to load json datasets')
    parser.add_argument('--random_qa_dir', type=str, default=RANDOM_QA_DIR, help='dir from which to load fixed random questions lists')
    parser.add_argument('--fold', type=int, choices=FOLD_IDS, help='fold id, if you want calculation in folds')
    return parser.parse_args()


def get_probing_name(
    probing_name,
    dataset_name,
    dataset_split,
    probe_type,
    probe_name,
    n_last_tokens,
    epochs,
    fold
):
    '''Use default probing name if not given'''
    # if probing name was given, simply return it
    if probing_name is not None:
        return probing_name
    
    # in other case, return default name
    default_name = f'{dataset_name}_{dataset_split}_{probe_type}_{probe_name}_last{n_last_tokens}'

    if probe_type != 'linear':
        default_name += f'_epochs{epochs}'

    if fold is not None:
        default_name += f'_f{fold}'

    return default_name


def identify_probing_model(probe_type, probe_name):
    if probe_name is None:
        return get_default_probe_name(probe_type)
    return probe_name


def get_meta_data(
    model_name,
    activations_dir,
    dataset_name,
    dataset_split,
    activations_format,
    intro_prompt,
    activations_hash,
    probing_model_type,
    probing_model_name,
    n_last_tokens,
    tokens_operation,
    epochs,
    early_stopping,
    val_ratio,
    val_split_hash,
    ray_tune,
    seed,
    train_accuracy,
    val_accuracy,
    fold,
    fold_split_hash
):
    # do not note epochs and early stopping for linear models
    if probing_model_type == 'linear':
        epochs = 0
        early_stopping = False

    meta = {
        'model': model_name,
        'activations': {
            'dir': activations_dir,
            'dataset': dataset_name,
            'dataset_split': dataset_split,
            'format': activations_format,
            'intro_prompt': intro_prompt,
            'hash': activations_hash
        },
        'training': {
            'model_type': probing_model_type,
            'model_name': probing_model_name,
            'n_last_tokens': n_last_tokens,
            'tokens_operation': tokens_operation,
            'epochs': epochs,
            'early_stopping': early_stopping
        },
        'results': {
            'acc_train': train_accuracy,
            'acc_val': val_accuracy
        },
        'split': {
            'val_ratio': val_ratio,
            'val_split_hash': val_split_hash
        },
        'ray_tune': ray_tune,
        'seed': seed
    }

    if fold is not None:
        meta['fold'] = {
            'id': fold,
            'fold_split_hash': fold_split_hash
        }
    
    return meta


def train_model(
        seed,
        tmp_dir_path,
        probe_type,
        probe_name,
        n_tokens,
        layer,
        head,
        data,
        epochs,
        early_stopping,
        ray_tune,
):
    torch.set_num_threads(1)

    set_seeds(seed)

    X_train = data['train_activations'][:,layer,head,:].copy()
    X_val = data['val_activations'][:,layer,head,:].copy()
    y_train = data['train_labels'].copy()
    y_val = data['val_labels'].copy()

    model = probe_factory(probe_type, probe_name, n_tokens, seed)

    # for cnn probe, reshape from concat to stack of vectors
    if probe_type == 'cnn':
        X_train = rearrange(X_train, 'b (v a) -> b v a', v=n_tokens)
        X_val = rearrange(X_val, 'b (v a) -> b v a', v=n_tokens)

    if probe_type == 'linear':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_acc = round(accuracy_score(y_train, y_pred), 4)
        val_acc = round(accuracy_score(y_val, y_val_pred), 4)
    else:
        train_loader = get_fixed_seed_data_loader(X_train, y_train)
        valid_loader = get_fixed_seed_data_loader(X_val, y_val)

        model.to('cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        loss_fn = nn.BCELoss()

        best_acc = -np.inf
        best_weights = None
        val_accs = []

        for epoch in range(epochs):
            model.train()

            train_losses = []
            train_preds, train_actuals = [], []

            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.cuda(), y_train_batch.cuda()
                optimizer.zero_grad()
                
                y_pred_batch = model(X_train_batch)
                y_pred_batch_binary = y_pred_batch.round()
                train_preds.extend(y_pred_batch_binary.squeeze().tolist())
                train_actuals.extend(y_train_batch.tolist())

                loss = loss_fn(y_pred_batch, y_train_batch[:, None])
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            train_loss = round(sum(train_losses) / len(train_losses), 4)

            train_correct = [i == j for i, j in zip (train_preds, train_actuals)]
            train_acc = round(sum(train_correct) / len(train_correct), 4)

            model.eval()

            val_losses = []
            val_preds, val_actuals =[], []

            for X_val_batch, y_val_batch in valid_loader:
                X_val_batch, y_val_batch = X_val_batch.cuda(), y_val_batch.cuda()

                y_pred_batch = model(X_val_batch)
                y_pred_batch_binary = y_pred_batch.round()
                val_preds.extend(y_pred_batch_binary.squeeze().tolist())
                val_actuals.extend(y_val_batch.tolist())

                loss = loss_fn(y_pred_batch, y_val_batch[:, None])
                val_losses.append(loss.item())

            val_loss = round(sum(val_losses) / len(val_losses), 4)

            val_correct = [i == j for i, j in zip (val_preds, val_actuals)]
            val_acc = round(sum(val_correct) / len(val_correct), 4)
            val_accs.append(val_acc)

            # save best weights
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            if not ray_tune:
                print(f'Epoch: {epoch}{TAB}'
                      f'Train loss: {train_loss:.2f}{TAB}Valid loss: {val_loss:.2f}{TAB}'
                      f'Train acc: {train_acc:.2f}{TAB}Valid acc: {val_acc:.2f}')

            if early_stopping and best_acc > max(val_accs[-EARLY_STOPPING_PATIENCE:]):
                print(f'Early stopped at epoch {epoch}')
                break

        model.load_state_dict(best_weights)

        X_train = torch.FloatTensor(X_train).cuda()
        y_train_pred = model(X_train)
        y_train_pred_binary = y_train_pred.round()
        train_acc = accuracy_score(y_train, y_train_pred_binary.squeeze().tolist())

        X_val = torch.FloatTensor(X_val).to('cuda')
        y_val_pred = model(X_val)
        y_val_pred_binary = y_val_pred.round()
        val_acc = accuracy_score(y_val, y_val_pred_binary.squeeze().tolist())

    results = {"layer_head": (layer, head)}
    results["accuracy train"] = train_acc
    results["accuracy val"] = val_acc

    if probe_type == 'cnn':
        token_weights = best_weights["0.weight"].cpu().numpy().squeeze()
    else:
        token_weights = np.ones(n_tokens)
    results["token weights"] = normalize_values(token_weights)

    # save model weights
    save_probing_model(model, probe_type, tmp_dir_path, head, layer)

    if ray_tune:
        session.report(results)
    else:
        return results


def train_probes(
    seed,
    probe_type,
    probe_name,
    n_tokens,
    train_activations,
    train_labels,
    val_activations,
    val_labels,
    n_model_layers,
    n_model_heads,
    epochs,
    early_stopping,
    ray_tune,
    ray_cpu,
    ray_gpu,
):
    import time
    tic  = time.time()

    data = {
        "train_activations": train_activations,
        "train_labels": train_labels,
        "val_activations": val_activations,
        "val_labels": val_labels,
    }

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = tmp_dir.name
    make_probing_dir(tmp_dir_path)

    if ray_tune:
        config = {
            "layer": tune.grid_search(list(range(n_model_layers))),
            "head":  tune.grid_search(list(range(n_model_heads))),
        }

        def get_partial():
            def partial(config, data=None):
                train_model(
                    seed,
                    tmp_dir_path,
                    probe_type,
                    probe_name,
                    n_tokens,
                    config['layer'],
                    config['head'],
                    data,
                    epochs,
                    early_stopping,
                    ray_tune,
                )
            return partial
        
        result = tune.run(
            tune.with_parameters(get_partial(), data=data),
            resources_per_trial={"cpu": ray_cpu,"gpu": ray_gpu},
            config=config,
            num_samples=1,
            reuse_actors=False,
        )

        df = result.dataframe()
        df.sort_values("layer_head", inplace=True)
        
        all_head_train_accs = df["accuracy train"].values.tolist()
        all_head_val_accs = df["accuracy val"].values.tolist()
        all_token_weights = df["token weights"].values.tolist()
        probes = []
    else:
        probes = []
        all_head_train_accs = []
        all_head_val_accs = []
        all_token_weights = []

        for layer in range(n_model_layers):
            for head in range(n_model_heads):
                print(f'Layer: {layer + 1}/{n_model_layers}{TAB}Head: {head + 1}/{n_model_heads}')

                result = train_model(
                    seed,
                    tmp_dir_path,
                    probe_type,
                    probe_name,
                    n_tokens,
                    layer,
                    head,
                    data,
                    epochs,
                    early_stopping,
                    ray_tune,
                )

                all_head_train_accs.append(result["accuracy train"])
                all_head_val_accs.append(result["accuracy val"])
                all_token_weights.append(result["token weights"])

    toc = time.time()
    print("time:", toc-tic, "[sec]")

    # output head accs, token weights & probing models
    all_head_accs_np = np.array(all_head_val_accs)
    all_token_weights_np = np.array(all_token_weights)
    probes = load_probes(
        probe_type,
        probe_name,
        n_tokens,
        tmp_dir_path,
        n_model_layers,
        n_model_heads,
        seed
    )

    # results
    mean_acc_train = round(np.mean(all_head_train_accs) * 100, 2)
    mean_acc_val = round(np.mean(all_head_val_accs) * 100, 2)

    tmp_dir.cleanup()

    return probes, all_head_accs_np, all_token_weights_np, mean_acc_train, mean_acc_val


def main(args):
    set_seeds(args.seed)

    # identify probing model
    probing_model_name = identify_probing_model(args.probe_type, args.probe_name)

    # set probing name
    probing_name = get_probing_name(
        args.probing_name,
        args.dataset_name,
        args.dataset_split,
        args.probe_type,
        probing_model_name,
        args.n_last_tokens,
        args.epochs,
        args.fold
    )

    # create new model
    hf_model_name = HF_MODEL_NAMES[args.model_name]
    model = llama.LLaMAForCausalLM.from_pretrained(hf_model_name,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16)

    # get number of layers and heads
    n_model_layers = model.config.num_hidden_layers
    n_model_heads = model.config.num_attention_heads

    # load activations
    head_wise_activations, activations_hash = load_activations(
        args.model_name,
        args.dataset_name,
        args.dataset_split,
        args.activations_format,
        args.activations_dir,
        n_model_heads,
        args.n_last_tokens,
        args.token_op,
        args.seed,
        args.activations_intro_prompt
    )

    # split activations
    (train_activations,
     train_labels,
     val_activations,
     val_labels,
     val_split_hash,
     fold_split_hash) = split_activations(
        head_wise_activations,
        args.dataset_name,
        args.dataset_split,
        args.datasets_dir,
        args.val_ratio,
        args.fold
    )

    # print trainig info
    print(f"Training with {args.probe_type} model '{probing_model_name}'")

    if args.fold:
        print(f"Training for fold {args.fold}")

    # Finally: dropped calculation in folds
    # k-fold makes sense but only for training and measuring performance on the same dataset
    # when evaluating on a different dataset, probing models should be trained on a whole datasetset to maximize their accuracy

    # train probes #
    # IN: activations
    # training and validation on a validation subset
    # OUT: accuracy for each head
    probes, all_head_accs_np, all_token_weights_np, acc_train, acc_val = train_probes(
        args.seed,
        args.probe_type,
        probing_model_name,
        args.n_last_tokens,
        train_activations,
        train_labels,
        val_activations,
        val_labels,
        n_model_layers,
        n_model_heads,
        args.epochs,
        args.use_early_stopping,
        args.ray_tune,
        args.ray_cpu,
        args.ray_gpu,
    )
    all_head_accs_np = all_head_accs_np.reshape(n_model_layers, n_model_heads)
    all_token_weights_np = all_token_weights_np.reshape(
        n_model_layers,
        n_model_heads,
        args.n_last_tokens
    )

    # make sure the dir for saving models exists
    probing_dir = get_probing_dir(probing_name)
    make_probing_dir(probing_dir)

    # save accuracies and probes
    save_head_accs(all_head_accs_np, probing_dir)
    save_token_weights(all_token_weights_np, probing_dir)
    save_probes(probes, args.probe_type, probing_dir)

    # save meta data
    meta_data = get_meta_data(
        args.model_name,
        args.activations_dir,
        args.dataset_name,
        args.dataset_split,
        args.activations_format,
        args.activations_intro_prompt,
        activations_hash,
        args.probe_type,
        probing_model_name,
        args.n_last_tokens,
        args.token_op,
        args.epochs,
        args.use_early_stopping,
        args.val_ratio,
        val_split_hash,
        args.ray_tune,
        args.seed,
        acc_train,
        acc_val,
        args.fold,
        fold_split_hash
    )
    save_meta_data(meta_data, probing_dir)


if __name__ == "__main__":
    main(pass_args())
