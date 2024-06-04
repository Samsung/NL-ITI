import argparse
import os
import tempfile

import llama
import numpy as np
import torch

from baukit import TraceDict
from pathlib import Path
from tqdm import tqdm

from common import (ACTIVATIONS_DIR, DATASET_NAMES, DATASET_SPLITS, DATASETS_DIR, FORMATS,
                    HF_MODEL_NAMES, RANDOM_QA_DIR, SEED)
from fix_randomness import get_random_questions_generator
from prepare_dataset import (format_qa, format_endq, get_intro_prompt, load_dataset)
from utils import append_key_value_to_meta_data, get_activations_name, hash_array, set_seeds


def pass_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B', choices=HF_MODEL_NAMES.keys())
    parser.add_argument('--dataset_name', type=str, default='truthful_qa', choices=DATASET_NAMES)
    parser.add_argument('--dataset_split', type=str, default='valid', choices=DATASET_SPLITS)
    parser.add_argument('--prompt_format', type=str, default='qa', choices=FORMATS)
    parser.add_argument('--activations_dir', type=str, help='dir where to save activations', default=ACTIVATIONS_DIR)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_last_tokens', type=int, help='number of last tokens to take when getting activations', default=2)
    parser.add_argument('--intro_prompt', action='store_true', help='use instruction line + few shot QA pairs before prompt', default=False)
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--datasets_dir', type=str, default=DATASETS_DIR, help='dir from which to load json datasets')
    parser.add_argument('--random_qa_dir', type=str, default=RANDOM_QA_DIR, help='dir from which to load fixed random questions lists')
    return parser.parse_args()


def get_llama_activations_bau(model, prompt, device):

    model.eval()

    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
            output = model(prompt, output_hidden_states = True)

        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def tokenize_prompt(tokenizer, prompt, prompt_format, dataset_name, use_intro_prompt):
    if use_intro_prompt:
        intro_prompt = get_intro_prompt(dataset_name)               # instruction + [few shot QA pairs]
    else:
        intro_prompt = ''

    if prompt_format == 'qa':
        delimeter = 'A: '
    elif prompt_format == 'endq':
        delimeter = 'Q: '
    else:
        print(f'Unsupported prompt format: {prompt_format}')
        exit(1)

    pre_target_part = delimeter.join(prompt.split(delimeter)[:-1])  # prompt pre last (target) line

    prompt_pre_target = intro_prompt + pre_target_part              # [instruction + few shot] + A/Q prefix

    prompt = intro_prompt + prompt                                  # [instruction + few shot] + full A/Q

    prompt_pre_target_ids = tokenizer(prompt_pre_target, return_tensors="pt").input_ids
    # print('prompt_pre_target', token_ids_to_tokens(tokenizer, prompt_pre_target_ids))
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # print('prompt', token_ids_to_tokens(tokenizer, prompt_ids))

    target_starts_at_idx = prompt_pre_target_ids.shape[-1] + 2  # +2 to omit A:/Q: prefix
    # print('target', token_ids_to_tokens(tokenizer, prompt_ids)[target_starts_at_idx:])

    return prompt_ids, target_starts_at_idx


def get_prompt_activations(
    model,
    tokenizer,
    device,
    prompt,
    prompt_format,
    dataset_name,
    use_intro_prompt,
    n_last_tokens
):
    # count non zero vectors
    non_zero_vector_count = n_last_tokens

    # tokenize prompt
    prompt_ids, target_starts_at_idx = tokenize_prompt(
        tokenizer,
        prompt,
        prompt_format,
        dataset_name,
        use_intro_prompt
    )
    # collect output activations
    _, head_wise_activations, _ = get_llama_activations_bau(model, prompt_ids, device)
    # limit the output activations to just the target part
    head_wise_activations = head_wise_activations[:, target_starts_at_idx:, :]
    # only from that, get last n tokens
    activations = head_wise_activations[:, -n_last_tokens:, :]

    # pad preceding tokens with 0s until reaching n_last_tokens length
    while activations.shape[1] < n_last_tokens:
        zeros_vector = np.zeros(activations.shape[-1])
        activations = np.insert(activations, 0, zeros_vector, axis=1)
        # for every zero vector inserted, decrese count of non_zero_vectors
        non_zero_vector_count -= 1

    return activations, non_zero_vector_count


def collect_activations(
    model,
    tokenizer,
    prompts,
    n_model_layers,
    model_hidden_size,
    dataset_name,
    prompt_format,
    device,
    n_last_tokens,
    use_intro_prompt,
):
    head_wise_activations = np.empty(
        (len(prompts), n_model_layers, n_last_tokens, model_hidden_size),
        np.float16
    )
    non_zero_vector_counts = np.empty(len(prompts), np.int8)

    for i, prompt in enumerate(tqdm(prompts)):
        prompt_activations, prompt_non_zero_vector_count = get_prompt_activations(
            model,
            tokenizer,
            device,
            prompt,
            prompt_format,
            dataset_name,
            use_intro_prompt,
            n_last_tokens
        )
        head_wise_activations[i] = prompt_activations
        non_zero_vector_counts[i] = prompt_non_zero_vector_count

    return head_wise_activations, non_zero_vector_counts


def main(args):
    set_seeds(args.seed)

    hf_model_name = HF_MODEL_NAMES[args.model_name]

    tokenizer = llama.LLaMATokenizer.from_pretrained(hf_model_name)
    model = llama.LLaMAForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    model.to(args.device)

    n_model_layers = model.config.num_hidden_layers
    model_hidden_size = model.config.hidden_size

    dataset = load_dataset(args.dataset_name, args.dataset_split, args.datasets_dir)

    print('Formatting prompts')
    if args.prompt_format == 'qa':
        _, _, _, _, prompts = format_qa(dataset, tokenizer)
    elif args.prompt_format == 'endq':
        random_q_generator = get_random_questions_generator(args.dataset_name, args.random_qa_dir)
        _, _, _, prompts = format_endq(dataset, tokenizer, random_q_generator)
    else:
        print(f"Choose prompt format from {FORMATS}")
        exit(1)

    print('Getting activations')
    head_wise_activations, non_zero_vector_counts = collect_activations(
        model,
        tokenizer,
        prompts,
        n_model_layers,
        model_hidden_size,
        args.dataset_name,
        args.prompt_format,
        args.device,
        args.n_last_tokens,
        args.intro_prompt,
    )

    # make sure the dir for saving activations exists
    Path(args.activations_dir).mkdir(exist_ok=True)

    print("Saving")
    activations_name = get_activations_name(
        args.model_name,
        args.dataset_name,
        args.dataset_split,
        args.prompt_format,
        args.n_last_tokens,
        args.seed,
        args.intro_prompt
    )
    np.savez(
        f'{args.activations_dir}/{activations_name}_activations',
        head_wise_activations=head_wise_activations,
        non_zero_vector_counts=non_zero_vector_counts
    )

    # save meta data
    activations_hash = hash_array(head_wise_activations)
    append_key_value_to_meta_data(
        key=activations_name,
        value=activations_hash,
        dir_path=args.activations_dir
    )


if __name__ == '__main__':
    main(pass_args())
