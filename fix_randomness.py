import argparse

import numpy as np

from common import DATASETS_DIR, N_TRAIN_SUBSET_QUESTIONS, RANDOM_QA_DIR, SEED
from prepare_dataset import load_dataset
from utils import (load_random_questions, make_dir, save_random_questions,
                   save_dataset_json, set_seeds)


def pass_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--datasets_dir', type=str, default=DATASETS_DIR, help='dir where to save fixed random subsets')
    parser.add_argument('--random_qa_dir', type=str, default=RANDOM_QA_DIR, help='dir where to save fixed random questions / answers lists')
    return parser.parse_args()


def get_random_questions_generator(dataset_name, random_qa_dir):
    # find first questions list in the dir
    random_questions_list = load_random_questions(dataset_name, random_qa_dir)
    # return generator from this list
    return (x for x in random_questions_list)


def generate_random_questions(dataset_name, dataset_split, n, seed=SEED, datasets_dir=DATASETS_DIR, random_qa_dir=RANDOM_QA_DIR):
    # load dataset
    dataset = load_dataset(dataset_name, dataset_split, datasets_dir)

    # get a list of all questions
    questions_bank = dataset['question']

    # get random questions
    set_seeds(seed)
    np.random.shuffle(questions_bank)
    random_questions_list = np.random.choice(questions_bank, n)

    # make sure dir exists
    make_dir(random_qa_dir)

    save_random_questions(random_questions_list, dataset_name, random_qa_dir)


def generate_dataset_subset(dataset_name, dataset_split, seed=SEED, datasets_dir=DATASETS_DIR):
    # load dataset
    full_split = dataset_split.split('_')[0]
    dataset = load_dataset(dataset_name, full_split, datasets_dir)

    # get a list of all questions
    question_ids = np.array(range(len(dataset)))

    # shuffle questions ids
    set_seeds(seed)
    np.random.shuffle(question_ids)

    # get random subset
    dataset_subset = dataset.select(question_ids[:N_TRAIN_SUBSET_QUESTIONS])

    # make sure dir exists
    make_dir(datasets_dir)

    # save to file
    save_dataset_json(dataset_subset, dataset_name, dataset_split, datasets_dir)


def main(args):
    generate_dataset_subset('arc_c', 'train_subset', args.seed, args.datasets_dir)
    generate_dataset_subset('commonsense_qa', 'train_subset', args.seed, args.datasets_dir)
    generate_dataset_subset('mmlu', 'train_subset', args.seed, args.datasets_dir)
    generate_dataset_subset('openbook_qa', 'train_subset', args.seed, args.datasets_dir)

    generate_random_questions('arc_c', 'train_subset', 1100, args.seed, args.datasets_dir, args.random_qa_dir)
    generate_random_questions('commonsense_qa', 'train_subset', 1100, args.seed, args.datasets_dir, args.random_qa_dir)
    generate_random_questions('mmlu', 'train_subset', 1100, args.seed, args.datasets_dir, args.random_qa_dir)
    generate_random_questions('openbook_qa', 'train_subset', 1100, args.seed, args.datasets_dir, args.random_qa_dir)
    generate_random_questions('truthful_qa', 'valid', 1000, args.seed, args.datasets_dir, args.random_qa_dir)


if __name__ == '__main__':
    main(pass_args())
