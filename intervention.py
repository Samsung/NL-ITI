import argparse

from collections import OrderedDict
from functools import partial
from glob import glob
from pathlib import Path

import datasets
import llama
import numpy as np
import pandas as pd
import torch

from baukit import TraceDict
from einops import rearrange
from tqdm import tqdm
from torch.nn.functional import softmax

from common import (ACTIVATIONS_DIR, DATASET_NAMES, DATASET_SPLITS, DATASETS_DIR, FOLD_IDS,
                    FORMATS, HEAD_MASK_JSON, HT_ALPHA, HT_HEADS, HF_MODEL_NAMES, PROBINGS_DIR,
                    RESULTS_DIR, RESULTS_JSON_NAME, SEED, TOKEN_OPS)
from head_masking import generate_head_mask
from prepare_dataset import (format_qa, format_endq, get_fold_split, get_intro_prompt,
                             load_activations, load_dataset, load_labels,
                             reduce_activations_to_selected_questions,
                             reduce_prompting_lists_to_selected_questions)
from utils import (get_top_heads, get_probing_dir, load_head_scores, load_meta_data,
                   load_token_weights, save_meta_data, set_seeds)


TAB = '\t'


def pass_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B', choices=HF_MODEL_NAMES.keys(), help='model name')
    parser.add_argument('--activations_dir', type=str, help='dir where activations were dumped', default=ACTIVATIONS_DIR)
    parser.add_argument('--probing_name', type=str, choices=glob('*', root_dir=PROBINGS_DIR), help='probing results to use', required=True)
    parser.add_argument('--weights_name', type=str, choices=glob('*', root_dir=PROBINGS_DIR), help='use another probing name to take token weights from', default=None)
    parser.add_argument('--results_dir', type=str, help='dir were intervention results will be saved', default=RESULTS_DIR)
    parser.add_argument('--datasets_dir', type=str, default=DATASETS_DIR, help='dir from which to load json datasets')
    parser.add_argument('--results_name', type=str, help='name for the results json', default=None)
    parser.add_argument('--activations_intro_prompt', action='store_true', help='use activations collected with intro + few shot QA prompt', default=False)
    parser.add_argument('--activations_split', type=str, default='valid', choices=DATASET_SPLITS)
    parser.add_argument('--activations_direction', type=str, default='truthful_qa', choices=DATASET_NAMES,
                        help='activations dataset from which intervention directions are calculated')
    parser.add_argument('--activations_direction_format', type=str, default='qa', choices=FORMATS)
    parser.add_argument('--activations_std', type=str, default='truthful_qa', choices=DATASET_NAMES,
                        help='activations dataset from which std along intervention directions is calculated')
    parser.add_argument('--activations_std_format', type=str, default='endq', choices=FORMATS)
    parser.add_argument('--evaluation_dataset', type=str, default='truthful_qa', choices=DATASET_NAMES,
                        help='prompts dataset on which LLM is evaluated')
    parser.add_argument('--evaluation_dataset_split', type=str, default='valid', choices=DATASET_SPLITS)
    parser.add_argument('--heads', type=int, default=48, help='number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--no_few_shot_prompt', action='store_true', help='dont add few shot QA pairs to the prompt', default=False)
    parser.add_argument('--n_last_tokens', type=int, help='number of last tokens to use when taking a mean activation', default=2)
    parser.add_argument('--token_op', type=str, choices=TOKEN_OPS, help='way of merging tokens into single activations vector', default='mean')
    parser.add_argument('--hyperparameter_tuning', '-ht', action='store_true', help='run calculation on a matrix of head & alpha parameters', default=False)
    parser.add_argument('--head_masking', action='store_true', help='run masking calculation and assign to model', default=False)
    parser.add_argument('--fold', type=int, choices=FOLD_IDS, help='for calculation in folds, use the other fold id to run iti on')
    parser.add_argument('--detailed', action='store_true', help='save detailed (per category) results', default=False)
    return parser.parse_args()


def get_default_results_name():
    """name + datetime"""
    import datetime

    tz = datetime.timezone.utc
    ft = "%Y-%m-%dT%H_%M_%S"
    dt = datetime.datetime.now(tz=tz).strftime(ft)

    dt_name = f'{RESULTS_JSON_NAME}_{dt}'
    return dt_name


def get_results_name(results_name):
    # use default results name if not given
    if results_name is None:
        return get_default_results_name()
    return results_name


def get_meta_data(
    model_name,
    probing_dir,
    activations_dir,
    activations_direction,
    activations_direction_format,
    activations_direction_hash,
    activations_std,
    activations_std_format,
    activations_std_hash,
    activations_intro_prompt,
    activations_split,
    n_last_tokens,
    tokens_operation,
    evaluation_set,
    evaluation_set_split,
    few_shot_prompt,
    results,
    head_mask_meta,
    seed,
    fold,
    fold_split_hash
):
    probing_meta = load_meta_data(probing_dir)

    meta = {
        'model': model_name,
        'probing': probing_meta,
        'intervention_activations': {
            'dir': activations_dir,
            'direction': {
                'dataset': activations_direction,
                'format': activations_direction_format,
                'hash': activations_direction_hash
            },
            'std': {
                'dataset': activations_std,
                'format': activations_std_format,
                'hash': activations_std_hash
            },
            'intro_prompt': activations_intro_prompt,
            'split': activations_split
        },
        'intervention': {
            'n_last_tokens': n_last_tokens,
            'tokens_operation': tokens_operation,
        },
        'evaluation': {
            'set': evaluation_set,
            'split': evaluation_set_split,
            'few_shot_prompt': few_shot_prompt,
        },
        'results': results,
        'head_mask': head_mask_meta,
        'seed': seed,
    }

    if fold is not None:
        meta['fold'] = {
            'id': fold,
            'fold_split_hash': fold_split_hash
        }
    
    return meta


def layer_head_to_flattened_idx(layer, head, n_model_heads):
    return layer * n_model_heads + head


def get_com_directions(n_model_layers, n_model_heads, head_wise_activations, labels):
    com_directions = []
    for layer in range(n_model_layers):
        for head in range(n_model_heads):
            layer_head_activations = head_wise_activations[:, layer, head, :]
            true_mass_mean = np.mean(layer_head_activations[labels == 1], axis=0)
            false_mass_mean = np.mean(layer_head_activations[labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions


def get_interventions_dict(top_heads, tuning_activations, n_model_heads, com_directions):
    '''Calculate intervention direction for the top heads'''
    interventions = {}
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = []
    for layer, head in top_heads:
        # use center of mass, always
        direction = com_directions[layer_head_to_flattened_idx(layer, head, n_model_heads)]

        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.head_out"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(interventions[f"model.layers.{layer}.self_attn.head_out"], key = lambda x: x[0])

    return interventions


def lt_modulated_vector_add(head_output, layer_name, device, alpha, n_model_heads, interventions, start_edit_location):
    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=n_model_heads)
    for head, direction, proj_val_std in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).to(device)
        head_output[:, start_edit_location:, head, :] += alpha * proj_val_std * direction_to_add

    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    return head_output


def get_target_question(prompt):
    """splits (mc formatted) prompt and returns part before 'A: '"""
    return prompt.split('A: ')[0]


def run_ITI(
    model,
    tokenizer,
    device,
    seed,
    evaluation_dataset,
    n_model_heads,
    n_model_layers,
    direction_activations,
    direction_labels,
    std_activations,
    few_shot_prompt,
    all_head_scores,
    heads,
    alpha,
    prompts,
    questions,
    labels,
    best_labels,
    categories=None,
    detailed=False
):
    """ Inference time intervention 
    Returns scores (log probabilities) for each Q+A pair. 
    The highest score is corresponds to the most probable answer
    """

    # get top heads with truthful/trivia/(toxic?) representation
    top_heads = get_top_heads(all_head_scores, heads, n_model_layers, n_model_heads)

    # calculate intervention vectors - truthful direction for the top heads
    com_directions = get_com_directions(n_model_layers, n_model_heads, direction_activations, direction_labels)
    interventions = get_interventions_dict(top_heads, std_activations, n_model_heads, com_directions)
    
    # run intervention and collect models scores
    scores = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            # --- intervention code --- #
            score = intervene(
                prompt,
                model,
                tokenizer,
                device,
                alpha,
                n_model_heads,
                few_shot_prompt,
                evaluation_dataset,
                interventions
            )
            scores.append(score)
        
    # calculate results
    dict_to_eval = split_answers_per_question(questions, prompts, scores, labels, best_labels)

    # MC metrics
    for q in dict_to_eval.keys():
       dict_to_eval[q]['MC1'] = MC1(dict_to_eval[q])
       dict_to_eval[q]['MC2'] = MC2(dict_to_eval[q])
    
    # per category results
    if detailed:
        save_detailed_results(dict_to_eval, categories)
       
    # mean results
    mc1, mc2 = calc_mean_MCs(dict_to_eval)
    
    # CE and KL loss
    ce_loss = calc_ce_loss(
        model,
        tokenizer,
        device,
        interventions,
        lt_modulated_vector_add,
        alpha,
        n_model_heads,
        seed
    )
    kl_wrt_orig = calc_kl_wrt_orig(
        model,
        tokenizer,
        device,
        interventions,
        lt_modulated_vector_add,
        alpha,
        n_model_heads,
        seed
    )

    # print results
    print (
        f'Heads: {heads}{TAB}Alpha: {alpha}{TAB}'
        f'MC1 score: {(mc1 * 100):.2f}{TAB}MC2 score: {(mc2 * 100):.2f}{TAB}'
        f'CE loss: {ce_loss:.2f}{TAB}KL wrt original: {kl_wrt_orig:.2f}'
    )

    return mc1, mc2, ce_loss, kl_wrt_orig


def intervene(prompt, model, tokenizer, device, alpha, n_model_heads, few_shot_prompt, evaluation_set, interventions):
    """applies intervention (if interventions != {}) and returns score for a single pair Q+A"""

    intervention_fn = lt_modulated_vector_add

    intro_prompt = get_intro_prompt(evaluation_set, few_shot_prompt)  # instruction + [few shot QA pairs]

    question = get_target_question(prompt)
    prompt_endq = intro_prompt + question                # instruction + [few shot] + Q

    prompt = intro_prompt + prompt                       # instruction + [few shot] + QA

    prompt_endq_ids = tokenizer(prompt_endq, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_starts_at_idx = prompt_endq_ids.shape[-1]  # starting from (including) 'A:'
    
    if interventions == {}:        
        def id(head_output, layer_name):
            return head_output
        intervention_fn = id
        layers_to_intervene = []
    else:
        intervention_fn = partial(
            intervention_fn,
            device=device,
            alpha=alpha,
            n_model_heads=n_model_heads,
            interventions=interventions,
            start_edit_location=answer_starts_at_idx - 1  # token_id - 1 -> head activations from moment t impact prediction of the next token (t+1)
        )
        layers_to_intervene = list(interventions.keys())

    with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
        outputs = model(prompt_ids)[0].squeeze(0)

    outputs = outputs.log_softmax(-1)  # logits to log probs
    
    # take output probabilities, but only for the tokens that make up the answer
    answer_logprobs_table = outputs[answer_starts_at_idx - 1 : -1, :]  # rows - output tokens, columns - probs for each token id (32 000 of them) (max wins)
    answer_token_ids = prompt_ids.squeeze(0)[answer_starts_at_idx:]  # exact answer token ids
    
    # get logprobs for each token from the answer
    answer_logprobs = answer_logprobs_table[range(answer_logprobs_table.shape[0]), answer_token_ids] # range(0, 2), [123, 456, 789] ->
    # -> for token0 get log_prob for token_id 123, for token1 get log prob for token_id 456, for token2 get log prob for token_id 789 
    
    answer_content_logprobs = answer_logprobs  # including 'A:' (with or without 'A:' makes no difference, so collect full answer line)

    return answer_content_logprobs.sum().item()


def split_answers_per_question(questions, prompts, scores, labels, best_labels):
    """function which creates dictionary with questions as keys prompts, scores
    needed to calculate MC per question"""
    
    dict_to_eval = OrderedDict()
    for question in questions:
        dict_to_eval[question] = {
            'prompts' : [],
            'scores' : [],
            'labels' : [],
            'best_label': [],
            'MC1': None,
            'MC2' : None
        }
    for q, p, s, l, b in zip(questions, prompts, scores, labels, best_labels):
        dict_to_eval[q]['prompts'].append(p)
        dict_to_eval[q]['scores'].append(s)
        dict_to_eval[q]['labels'].append(l)
        dict_to_eval[q]['best_label'].append(b)

    return dict_to_eval


def split_scores_mc2(question):
    
    scores_true, scores_false = [], []
    for i in range(len(question['scores'])):
        if int(question['labels'][i]) == 1:
            scores_true.append(question['scores'][i])
        elif int(question['labels'][i]) == 0:
            scores_false.append(question['scores'][i])
        else:
            raise Exception('Forbidden non-binary label detected') 
            
    return scores_true, scores_false


def split_scores_mc1(question):

    score_true, scores_false = float('-inf'), []
    for i in range(len(question['scores'])):
        if int(question['best_label'][i]) == 1:
            score_true = question['scores'][i]
        elif int(question['labels'][i]) == 1:
            continue
        elif int(question['labels'][i]) == 0:
            scores_false.append(question['scores'][i])
        else:
            raise Exception('Forbidden non-binary label detected')

    return score_true, scores_false


def MC2(question):
    """compute MC2: normalized probability mass for correct answers
    allows multiple True and False answers
    scores true - log_probability assigned to true answers given to a single question
    scores false - log probability assigned to false ansers to the same question"""
    
    scores_true, scores_false = split_scores_mc2(question)
    probs_true = np.nan_to_num(np.exp(scores_true))
    probs_false = np.nan_to_num(np.exp(scores_false))

    probs_true = sum(probs_true) / (sum(probs_true) + sum(probs_false))
    return probs_true


def MC1(question):
    """compute MC1: 1vFalse -- best correct answer vs all false answers
    allows only single True and single/multiple false answers"""
    predicted_true = 0
    score_true, scores_false = split_scores_mc1(question)

    if score_true >  max(scores_false):
        predicted_true = 1

    return predicted_true


def calc_ce_loss(
    model,
    tokenizer,
    device,
    interventions,
    intervention_fn,
    alpha,
    n_model_heads,
    seed,
    num_samples=100
):

    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = datasets.load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])

    # define intervention
    if interventions == {}:
        def id(head_output, layer_name):
            return head_output
        intervention_fn = id
        layers_to_intervene = []
    else:
        intervention_fn = partial(
            intervention_fn,
            device=device,
            alpha=alpha,
            n_model_heads=n_model_heads,
            interventions=interventions,
            start_edit_location=0
        )
        layers_to_intervene = list(interventions.keys())

    losses = []

    # fix seed
    np.random.seed(seed)
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()

    with torch.no_grad():
        for i in rand_idxs:

            input_ids = owt[i]['input_ids'][:, :128].to(device)

            with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
                loss = model(input_ids, labels=input_ids).loss

            losses.append(loss.item())

    return np.mean(losses)


def calc_kl_wrt_orig(
    model,
    tokenizer,
    device,
    interventions,
    intervention_fn,
    alpha,
    n_model_heads,
    seed,
    num_samples=100
):

    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = datasets.load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])

    # define intervention
    if interventions == {}:
        def id(head_output, layer_name):
            return head_output
        intervention_fn = id
        layers_to_intervene = []
    else:
        intervention_fn = partial(
            intervention_fn,
            device=device,
            alpha=alpha,
            n_model_heads=n_model_heads,
            interventions=interventions,
            start_edit_location=0
        )
        layers_to_intervene = list(interventions.keys())

    kl_divs = []

    # fix seed
    np.random.seed(seed)
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()

    with torch.no_grad():
        for i in rand_idxs:
            input_ids = owt[i]['input_ids'][:, :128].to(device)

            orig_logits = model(input_ids).logits.cpu().type(torch.float32)
            orig_probs = softmax(orig_logits, dim=-1)

            with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
                logits = model(input_ids).logits.cpu().type(torch.float32)
                probs  = softmax(logits, dim=-1)

            kl_div = (orig_probs * (orig_probs / probs).log()).sum() / (input_ids.shape[-1] * input_ids.shape[-2])
            kl_divs.append(kl_div.item())

    return np.mean(kl_divs)


def save_detailed_results(dict_to_eval, categories):
    df_records = []

    for question, category in zip (dict_to_eval.keys(), categories):
        record = {'category': category, 'mc1': dict_to_eval[question]['MC1'], 'mc2': dict_to_eval[question]['MC2']}
        df_records.append(record)
    
    df = pd.DataFrame(df_records)
    cat_mc1 = df.groupby('category')['mc1'].mean()
    cat_mc2 = df.groupby('category')['mc2'].mean()

    cat_mcs = []
    for cat_mc1, mc1, cat_mc2, mc2 in zip(cat_mc1.items(), cat_mc2.items()):
        assert cat_mc1 == cat_mc2
        cat_mcs.append({'category': cat_mc1, 'mc1': mc1, 'mc2': mc2})
    
    cat_mcs_df = pd.DataFrame(cat_mcs)
    cat_mcs_df.to_csv(f'{RESULTS_DIR}/results_detailed.csv')


def calc_mean_MCs(eval_dict):
    mc1 = np.array([eval_dict[q]['MC1'] for q in eval_dict.keys()])
    mc2 = np.array([eval_dict[q]['MC2'] for q in eval_dict.keys()])
    return np.mean(mc1), np.mean(mc2)


def test_if_folds_match(probing_meta, fold):
    if 'fold' not in probing_meta:
        print("Haven't trained on fold, but evaluating on fold!")
        exit(1)
    
    probing_fold = probing_meta['fold']['id']
    if probing_fold == fold:
        print("Training and evaluating on the same fold!")
        exit(1)
    
    print(f'Trained on fold {probing_fold}, evaluating on fold {fold}')


def main(args):    
    set_seeds(args.seed)

    '''Load probing results (head accs)'''
    probing_dir = get_probing_dir(args.probing_name)
    all_head_scores = load_head_scores(probing_dir)
    
    '''Generate and save mask for model to use'''
    mask_hash = generate_head_mask(args.head_masking, all_head_scores)
    
    '''Load model and tokenizer'''   
    # create new model
    hf_model_name = HF_MODEL_NAMES[args.model_name]
    tokenizer = llama.LLaMATokenizer.from_pretrained(hf_model_name)
    model = llama.LLaMAForCausalLM.from_pretrained(hf_model_name,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   head_mask=HEAD_MASK_JSON)
    model.to(args.device)

    # get number of layers and heads
    n_model_layers = model.config.num_hidden_layers
    n_model_heads = model.config.num_attention_heads

    '''Load token weights (if using weighted_sum)'''
    if args.token_op == 'weighted_sum':
        if args.weights_name is None:
            weights_dir = probing_dir
        else:
            weights_dir = get_probing_dir(args.weights_name)

        token_weights = load_token_weights(weights_dir)
    else:
        token_weights = None

    '''Load prompt dataset'''
    dataset = load_dataset(args.evaluation_dataset, args.evaluation_dataset_split, args.datasets_dir)
    _, labels, best_labels, questions, prompts = format_qa(dataset, tokenizer)

    # load categories if they exist
    categories = None
    if 'category' in dataset.features:
        categories = dataset['category']

    '''Load activations that will be used for calculating intervention vectors'''
    # direction activations: to get com directions
    direction_activations, direction_activations_hash = load_activations(
        args.model_name,
        args.activations_direction,
        args.activations_split,
        args.activations_direction_format,
        args.activations_dir,
        n_model_heads,
        args.n_last_tokens,
        args.token_op,
        args.seed,
        args.activations_intro_prompt,
        token_weights
    )
    direction_labels = load_labels(args.activations_direction, args.activations_split, args.datasets_dir)

    # std activations: to get std of activations along intervention directions
    std_activations, std_activations_hash = load_activations(
        args.model_name,
        args.activations_std,
        args.activations_split,
        args.activations_std_format,
        args.activations_dir,
        n_model_heads,
        args.n_last_tokens,
        args.token_op,
        args.seed,
        args.activations_intro_prompt,
        token_weights
    )

    # (optional) split in folds
    if args.fold is None:
        fold_split_hash = None
    else:
        # test if folds match with info print
        probing_meta = load_meta_data(probing_dir)
        test_if_folds_match(probing_meta, args.fold)

        # fold split
        if args.fold == 1:
            _, eval_question_ids, _, train_question_ids, fold_split_hash = get_fold_split(dataset)
        else:
            _, train_question_ids, _, eval_question_ids, fold_split_hash = get_fold_split(dataset)

        # calculate intervention vectors on train split
        direction_activations, direction_labels = reduce_activations_to_selected_questions(
            dataset,
            train_question_ids,
            direction_activations
        )
        std_activations, _ = reduce_activations_to_selected_questions(
            dataset,
            train_question_ids,
            std_activations
        )
        # get prompts from evaluation split
        labels, best_labels, questions, prompts = reduce_prompting_lists_to_selected_questions(
            dataset,
            eval_question_ids,
            best_labels,
            questions,
            prompts
        )

    # Run calculation on a single (head, alpha) or on a whole matrix of (head, alpha)
    if args.hyperparameter_tuning:
        head_parameters = HT_HEADS
        alpha_parameters = HT_ALPHA
    else:
        head_parameters = [ args.heads ]
        alpha_parameters = [ args.alpha ]

    '''Run LLama inference with intervention applied to the layers'''
    results = []
    for n_heads in head_parameters:
        for alpha in alpha_parameters:
            mc1, mc2, ce_loss, kl_wrt_orig = run_ITI(
                model,
                tokenizer,
                args.device,
                args.seed,
                args.evaluation_dataset,
                n_model_heads,
                n_model_layers,
                direction_activations,
                direction_labels,
                std_activations,
                not args.no_few_shot_prompt,
                all_head_scores,
                n_heads,
                alpha,
                prompts,
                questions,
                labels,
                best_labels,
                categories,
                args.detailed
            )
            result_formatted = {
                'heads': n_heads,
                'alpha': alpha,
                'mc1': round(mc1 * 100, 2),
                'mc2': round(mc2 * 100, 2),
                'ce': round(ce_loss, 2),
                'kl': round(kl_wrt_orig, 2)
            }
            results.append(result_formatted)

    '''Save results to json'''

    # head mask meta
    if args.head_masking:
        head_mask_meta = mask_hash
    else:
        head_mask_meta = False

    meta_data = get_meta_data(
        args.model_name,
        probing_dir,
        args.activations_dir,
        args.activations_direction,
        args.activations_direction_format,
        direction_activations_hash,
        args.activations_std,
        args.activations_std_format,
        std_activations_hash,
        args.activations_intro_prompt,
        args.activations_split,
        args.n_last_tokens,
        args.token_op,
        args.evaluation_dataset,
        args.evaluation_dataset_split,
        not args.no_few_shot_prompt,
        results,
        head_mask_meta,
        args.seed,
        args.fold,
        fold_split_hash
    )

    results_name = get_results_name(args.results_name)
    
    # make sure the dir for saving results exists
    Path(args.results_dir).mkdir(exist_ok=True)

    save_meta_data(meta_data, args.results_dir, results_name)


if __name__ == '__main__':
    main(pass_args())
