import random

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def get_fixed_seed_data_loader(X, y, batch_size=32, shuffle=True, num_workers=1):
    return DataLoader(
        TensorDataset(Tensor(X), Tensor(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True
    )
