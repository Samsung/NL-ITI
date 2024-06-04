import json
import math

import numpy as np

from common import HEAD_MASK_JSON
from utils import hash_array


# def normalized(x, top, bottom):
#     result = (((x - bottom) / (top - bottom)) * 0.2) + 0.8
#     return result


def sigmoid_normalized(x, lower_boundary, upper_boundary, sigmoid=True):
    if x > upper_boundary:
        return 1
    
    if sigmoid:
        value = 1 / (1 + math.exp(-(15 * x - 9)))
        value = lower_boundary + (1 - lower_boundary) * value
    else:
        value = max(lower_boundary, x)
    
    return value


def generate_head_mask(use_masking, all_head_accs):
    # calculate masking values for each head
    if use_masking:
        top = 0
        bottom = 1
        for index, layer in enumerate(all_head_accs):
            if bottom > min(layer, key=float): bottom = min(layer, key=float)
            if top < max(layer, key=float): top = max(layer, key=float)

        all_head_masking_values = all_head_accs.copy()
        for index, layer in enumerate(all_head_masking_values):
            #all_head_masking_values[index] = np.array([normalized(head, top, bottom) for head in layer])
            all_head_masking_values[index] = np.array([sigmoid_normalized(head, 0.6, 0.61) for head in layer])
    else:
        # generate masks of 1 (no masking)
        all_head_masking_values = np.ones((32, 32), int)
    
    # dump them to json
    with open(HEAD_MASK_JSON, 'w', encoding='utf-8') as file:
        json.dump({'mask': all_head_masking_values.tolist()}, file, ensure_ascii=False, indent=4)
    
    mask_hash = hash_array(all_head_masking_values)
    return mask_hash
