ACTIVATIONS_DIR = 'activations'

DATASETS_DIR = 'datasets'

DATASET_NAMES = {
    'arc_c',
    'commonsense_qa',
    'mmlu',
    'openbook_qa',
    'truthful_qa',
}

DATASET_SPLITS = {
    'train',
    'train_subset',
    'valid',
    'test'
}

FOLD_IDS = {1, 2}

FORMATS = {
    'qa',
    'endq'
}

HEAD_MASK_JSON = 'head-mask.json'

HF_MODEL_NAMES = {
    'llama_7B': 'decapoda-research/llama-7b-hf',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'daryl149/Llama-2-7b-chat-hf',
}

HT_ALPHA = [ 5, 10, 15, 20, 25, 30 ]

HT_HEADS = [ 16, 32, 48, 64, 80, 96 ]

N_TRAIN_SUBSET_QUESTIONS = 1000

PROBES_DIR = 'probes'

PROBINGS_DIR = 'probings'

RANDOM_QA_DIR = 'random_qa'

RESULTS_DIR = 'results'

RESULTS_JSON_NAME = 'results'

SEED = 42

TOKEN_OPS = {'concat', 'mean', 'weighted_sum'}
