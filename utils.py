import hashlib
import json
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import random
import torch
import joblib

from common import FORMATS, PROBES_DIR, PROBINGS_DIR, RESULTS_JSON_NAME
from probes import probe_factory


def set_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"  # required for torch.use_deterministic_algorithms

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def get_top_heads(all_head_scores_np, n_top, n_layers, n_heads):
        top_scores = np.argsort(all_head_scores_np.reshape(n_heads*n_layers))[::-1][:n_top]
        top_heads = [flattened_idx_to_layer_head(idx, n_heads) for idx in top_scores]
        return top_heads


def get_probing_dir(probing_name):
    return f'{PROBINGS_DIR}/{probing_name}'


def get_head_accs_path(probing_dir):
    return f'{probing_dir}/all_head_accs.npy'


def get_token_weights_path(probing_dir):
    return f'{probing_dir}/all_token_weights.npy'


def get_random_questions_path(dataset_name, dir_path):
    return f'{dir_path}/{dataset_name}_questions.npy'


def get_model_path(probing_dir, layer, head):
    return f'{probing_dir}/{PROBES_DIR}/layer{layer}_head{head}'


def make_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_probing_dir(probing_dir):
    make_dir(f'{probing_dir}/{PROBES_DIR}')


def load_dataset_json(dataset_name, dataset_split, datasets_dir):
    dataset_json_path = f'{datasets_dir}/{dataset_name}_{dataset_split}.json'

    if not Path(dataset_json_path).exists():
        print(f'There is no dataset json under {dataset_json_path}')
        exit(1)

    with open(dataset_json_path) as f:
        return json.load(f)


def load_head_scores(probing_dir):
    return np.load(get_head_accs_path(probing_dir))


def load_token_weights(probing_dir):
    return np.load(get_token_weights_path(probing_dir))


def load_random_questions(dataset_name, dir_path):
    return np.load(get_random_questions_path(dataset_name, dir_path))


def load_probing_model(
    probing_model_type,
    probing_model_name,
    n_tokens,
    dir_path,
    layer,
    head,
    seed
):
    if probing_model_type == 'linear':
        with open(f'{get_model_path(dir_path, layer, head)}.joblib', 'rb') as f: 
            model = joblib.load(f)
    else:
        model = probe_factory(probing_model_type, probing_model_name, n_tokens, seed)
        model.load_state_dict(torch.load(f'{get_model_path(dir_path, layer, head)}.pt'))
    return model


def load_probes(probe_type, probe_name, n_tokens, dir_path, n_layers, n_heads, seed):
    return [
        (head,
         layer,
         load_probing_model(probe_type, probe_name, n_tokens, dir_path, layer, head, seed)
        ) for layer in range(n_layers) for head in range(n_heads)
    ]


def load_meta_data(dir_path, file_name='meta'):
    with open(f'{dir_path}/{file_name}.json') as f:
        meta = json.load(f)
    return meta


def numpy_save(array, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, array)


def save_dataset_json(dataset, dataset_name, dataset_split, datasets_dir):
    with open(f'{datasets_dir}/{dataset_name}_{dataset_split}.json', 'w') as f:
        json.dump(dataset.to_list(), f)


def save_head_accs(head_accs_np, probing_dir):
    numpy_save(head_accs_np, get_head_accs_path(probing_dir))


def save_token_weights(token_weights_np, probing_dir):
    numpy_save(token_weights_np, get_token_weights_path(probing_dir))


def save_random_questions(random_questions_list, dataset_name, dir_path):
    return numpy_save(
        random_questions_list,
        get_random_questions_path(dataset_name, dir_path)
    )


def save_probing_model(probing_model, probing_model_type, probing_dir, layer, head):
    if probing_model_type == 'linear':
        joblib.dump(probing_model, f'{get_model_path(probing_dir, layer, head)}.joblib')
        print(f'model saved: {get_model_path(probing_dir, layer, head)}.joblib')
    else:
        torch.save(deepcopy(probing_model.state_dict()), f'{get_model_path(probing_dir, layer, head)}.pt')


def save_probes(probes, probe_type, probing_dir):
    for probe in probes:
        layer, head, model = probe
        save_probing_model(model, probe_type, probing_dir, layer, head)


def save_meta_data(meta_data, dir_path, file_name='meta'):
    with open(f'{dir_path}/{file_name}.json', 'w') as f:
        json.dump(meta_data, f)


def append_key_value_to_meta_data(key, value, dir_path, file_name='meta'):
    '''Appends key-value to meta json. If the key already exists, it will be overwritten.
    It also sorts the json by key-value pairs.
    '''
    meta_filepath = Path(f'{dir_path}/{file_name}.json')

    # if the meta json doesn't exist, just save the key-value pair
    if not meta_filepath.exists():
        json_content = {key: value}
    else:
        with open(meta_filepath, 'r') as f:
            json_content = json.load(f)

        # add key-value
        json_content[key] = value
        # and sort it
        json_content = OrderedDict(sorted(json_content.items()))

    save_meta_data(json_content, dir_path, file_name)


def token_ids_to_tokens(tokenizer, token_ids_tensor):
    token_ids = token_ids_tensor.squeeze().tolist()
    id_to_token = {v: k for (k, v) in tokenizer.get_vocab().items()}
    tokens = [id_to_token[token_id] for token_id in token_ids]
    return tokens


def get_activations_name(model_name, dataset_name, dataset_split, prompt_format, n_last_tokens, seed, intro_prompt, shortened=False):
    activations_name = f'{model_name}_{dataset_name}_{dataset_split}_{prompt_format}'
    if intro_prompt:
        activations_name += '_intro_prompt'
    if not shortened:
        activations_name += f'_seed{seed}_last{n_last_tokens}'
    return activations_name


def get_head_with_top_acc(all_head_accs, n_model_heads, n_model_layers):

    def decode_layer_head_idx(idx):
        i = 0
        for layer in range(n_model_layers):
            for head in range(n_model_heads):
                if i == idx:
                    return layer + 1, head + 1
                i += 1

    top_acc = max(all_head_accs)
    top_layer_head_idx = all_head_accs.index(top_acc)

    print(f'Top acc: {top_acc} found in (layer, head) = {decode_layer_head_idx(top_layer_head_idx)}')


def hash_array(array):
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return hashlib.sha256(array).hexdigest()


def merge_fold_results(results_json_name_1, results_json_name_2, results_dir=RESULTS_JSON_NAME):
    '''The tests just check if fold ids and hash match together'''
    results_1 = load_meta_data(results_dir, results_json_name_1)
    results_2 = load_meta_data(results_dir, results_json_name_2)

    # test if folds match
    if 'fold' not in results_1 or 'fold' not in results_2:
        print('At least one of the result JSONs was not calculated in folds')
        return
    
    if results_1['fold']['fold_split_hash'] != results_2['fold']['fold_split_hash']:
        print("Fold split hash doesn't match!")
        return
    
    if results_1['probing']['fold']['fold_split_hash'] != results_1['fold']['fold_split_hash']:
        print("Fold splits between probing/intervention doesn't match for the first file!")
        return
    
    if results_2['probing']['fold']['fold_split_hash'] != results_2['fold']['fold_split_hash']:
        print("Fold splits between probing/intervention doesn't match for the second file!")
        return

    if results_1['probing']['fold']['fold_split_hash'] != results_1['fold']['fold_split_hash']:
        print("Fold splits between probing/intervention doesn't match for the first file!")
        return
    
    if results_1['probing']['fold']['id'] != 1 or results_2['probing']['fold']['id'] != 2:
        print("First file needs to be trained on fold 1, while the second on fold 2")
        return
    
    # merge and test
    merged_meta = results_1.copy()

    if results_1['model'] != results_2['model']:
        print("Model doesn't match")
        return

    # probing meta
    if results_1['probing']['activations'] != results_2['probing']['activations']:
        print("Probing activations doesn't match")
        return

    if results_1['probing']['training'] != results_2['probing']['training']:
        print("Training doesn't match")
        return

    mean_result = lambda key: round(
        (results_1['probing']['results'][key] + results_2['probing']['results'][key]) / 2, 2
    )
    merged_meta['probing']['results'] = {
        'acc_train': mean_result('acc_train'),
        'acc_val': mean_result('acc_val')
    }

    if results_1['probing']['split']['val_ratio'] != results_2['probing']['split']['val_ratio']:
        print("Val ratio doesn't match")
        return

    merged_meta['probing']['split'] = {
        'val_ratio': results_1['probing']['split']['val_ratio'],
        'val_split_hash_f1': results_1['probing']['split']['val_split_hash'],
        'val_split_hash_f2': results_2['probing']['split']['val_split_hash'],
    }

    if results_1['probing']['ray_tune'] != results_2['probing']['ray_tune']:
        del merged_meta['probing']['ray_tune']
    
    del merged_meta['probing']['fold']
    
    # intervention meta
    if results_1['intervention_activations'] != results_2['intervention_activations']:
        print("Intervention activations doesn't match")
        return
    
    if results_1['intervention'] != results_2['intervention']:
        print("Intervention doesn't match")
        return
    
    if results_1['evaluation'] != results_2['evaluation']:
        print("Evaluation doesn't match")
        return

    # results
    mean_results = []
    for result_1, result_2 in zip(results_1['results'], results_2['results']):
        if result_1['heads'] != result_2['heads'] or result_1['alpha'] != result_2['alpha']:
            print('When iterating results one by one, found a mismatch in (head, alpha) between JSONs')
            return

        mean_result = lambda key: round((float(result_1[key]) + float(result_2[key])) / 2, 2)

        mean_results.append({
            'heads': result_1['heads'],
            'alpha': result_1['alpha'],
            'mc1': mean_result('mc1'),
            'mc2': mean_result('mc2'),
            'ce': mean_result('ce'),
            'kl': mean_result('kl'),
        })
    merged_meta['results'] = mean_results

    # fold
    del merged_meta['fold']
    merged_meta['fold_split_hash'] = results_1['fold']['fold_split_hash']

    # remove the first JSON, save the merged result as the second one
    os.remove(f'{results_dir}/{results_json_name_1}.json')
    save_meta_data(merged_meta, results_dir, results_json_name_2)


def get_best_result(results_json_name, results_dir=RESULTS_JSON_NAME):
    '''Get the best results MC1-wise'''
    results = load_meta_data(results_dir, results_json_name)
    return max(results['results'], key=lambda entry: float(entry['mc1']))


def normalize_values(array):
    return array / np.sum(array)
