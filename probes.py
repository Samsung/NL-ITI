import torch.nn as nn

from sklearn.linear_model import LogisticRegression


PROBE_TYPES = {'cnn', 'linear', 'mlp'}

LINEAR_PROBES = {'logreg'}


def get_default_probe_name(probe_type):
    if probe_type == 'cnn':
        return 's'
    elif probe_type == 'linear':
        return 'logreg'
    elif probe_type == 'mlp':
        return 's'
    else:
        print(f'Use probe_type from {PROBE_TYPES}')
        return None


def build_cnn_probe(probe_name, n_tokens):
    if probe_name not in CNN_CLASS:
        print(f'For cnn probing model, use name from {CNN_CLASS.keys()}')
        return None

    return CNN_CLASS[probe_name](n_tokens)


def build_linear_probe(probe_name, seed):
    if probe_name == 'logreg':
        return LogisticRegression(random_state=seed, max_iter=1000)
    else:
        print(f'For linear probing model, use name from {LINEAR_PROBES}')
        return None


def build_mlp_probe(probe_name):
    if probe_name not in MLP_CLASS:
        print(f'For mlp probing model, use name from {MLP_CLASS.keys()}')
        return None

    return MLP_CLASS[probe_name]()


def probe_factory(probe_type, probe_name, n_tokens, seed):
    if probe_type == 'cnn':
        return build_cnn_probe(probe_name, n_tokens)
    if probe_type == 'linear':
        return build_linear_probe(probe_name, seed)
    elif probe_type == 'mlp':
        return build_mlp_probe(probe_name)
    else:
        print(f'Use probe_type from {PROBE_TYPES}')
        return None


class CNN_S(nn.Sequential):
    def __init__(self, n_tokens):
        super().__init__(
            nn.Conv1d(in_channels=n_tokens, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )


CNN_CLASS = {
    's': CNN_S
}


class MLP_S(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )


class MLP_M(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )


class MLP_L(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )


class MLP_XL(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )


class MLP_XXL(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid(),
    )


MLP_CLASS = {
    's': MLP_S,
    'm': MLP_M,
    'l': MLP_L,
    'xl': MLP_XL,
    'xxl': MLP_XXL,
}
