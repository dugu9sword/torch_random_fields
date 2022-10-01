import re
from typing import List

import einops
import numpy as np
import torch
from contextlib import contextmanager


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        state_cuda = torch.cuda.random.get_rng_state()
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(state_cuda)


@contextmanager
def local_seed(seed):
    with torch_seed(seed):
        yield
        

def chain(arg, *funcs):
    result = arg
    for f in funcs:
        result = f(result)
    return result


def compute_energy(
        unary_potentials,  #
        binary_potentials=None,
        binary_edges=None,
        ternary_potentials=None,
        ternary_edges=None,
        labels=None):
    energy = torch.sum(unary_potentials[torch.arange(len(labels)), labels])

    if binary_potentials is not None:
        # for edge, pw in zip(binary_edges, binary_potentials):
        #     energy += pw[labels[edge[0]], labels[edge[1]]]
        n_state = binary_potentials.shape[-1]
        score = einops.rearrange(binary_potentials, "B S1 S2->B (S1 S2)")
        score_index = labels[binary_edges]
        score_index = score_index[:, 0] * n_state + score_index[:, 1]
        score_index = einops.rearrange(score_index, "B->B ()")
        bin_score = torch.gather(score, axis=1, index=score_index)
        energy += torch.sum(bin_score)

    if ternary_potentials is not None:
        # for edge, tw in zip(ternary_edges, ternary_potentials):
        #     energy += tw[labels[edge[0]], labels[edge[1]], labels[edge[2]]]
        n_state = ternary_potentials.shape[-1]
        score = einops.rearrange(ternary_potentials, "B S1 S2 S3->B (S1 S2 S3)")
        score_index = labels[ternary_edges]
        score_index = score_index[:, 0] * (n_state**2) + score_index[:, 1] * n_state + score_index[:, 2]
        score_index = einops.rearrange(score_index, "B->B ()")
        ter_score = torch.gather(score, axis=1, index=score_index)
        energy += torch.sum(ter_score)

    return energy


#############################################
# Data generation
#############################################

def make_grid_edges(x, neighborhood=4, return_lists=False):
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood can only be '4' or '8', got %s" %
                         repr(neighborhood))
    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    inds = inds.astype(np.int64)
    right = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    down = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = [right, down]
    if neighborhood == 8:
        upright = np.c_[inds[1:, :-1].ravel(), inds[:-1, 1:].ravel()]
        downright = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
        edges.extend([upright, downright])
    if return_lists:
        return edges
    return np.vstack(edges)


def generate_edges(length, window):
    """
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (1, 3), (2, 4), (3, 5)]
    """
    edges = []
    for w in range(1, window + 1):
        for i in range(length - w):
            edges.append((i, i + w))
    return edges


def batch_pad(idx: List[List], pad_ele=0, pad_len=None) -> List[List]:
    if pad_len is None:
        pad_len = max(map(len, idx))
    return list(map(lambda x: x + [pad_ele] * (pad_len - len(x)), idx))


def batch_mask(idx: List[List], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(map(len, idx))
    return list(
        map(lambda x: [good_ele] * len(x) + [mask_ele] * (max_len - len(x)),
            idx))


def batch_mask_by_len(lens: List[int], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(lens)
    return list(
        map(lambda x: [good_ele] * x + [mask_ele] * (max_len - x), lens))


def batch_append(idx: List[List], append_ele, before=False) -> List[List]:
    if not before:
        return list(map(lambda x: x + [append_ele], idx))
    else:
        return list(map(lambda x: [append_ele] + x, idx))


def batch_lens(idx: List[List]) -> List:
    return list(map(len, idx))


def as_batch(idx: List) -> List[List]:
    return [idx]


def flatten_lst(lst: List[List]) -> List:
    return [i for sub_lst in lst for i in sub_lst]


#############################################
# Better einsum
#############################################
def einsumx(equation, *operands):
    """
        param:
        ---
        >>> einsumx("bsz seq_len hidden , hidden o -> bsz seqlen o", *operands)
        >>> # same as
        >>> torch.einsum("zyx,xo->zyo", *operands)
    """
    equation = re.sub(",", " , ", equation.strip())
    equation = re.sub("->", " -> ", equation.strip())
    equation = re.sub(r"\s+", " ", equation.strip())
    # print(equation)
    avail_chars = list("abcdefghijklmnopqrstuvwxyz")
    for ele in equation.split(" "):
        if ele in avail_chars:
            avail_chars.remove(ele)
    maps = {}
    conversion = []
    for ele in equation.split(" "):
        if ele in (",", "->"):
            conversion.append(ele)
        elif len(ele) == 1 and ele in "abcdefghijklmnopqrstuvwxyz":
            conversion.append(ele)
        else:
            if ele not in maps:
                maps[ele] = avail_chars.pop()
            conversion.append(maps[ele])
    conversion = "".join(conversion)
    # print(conversion)
    try:
        ret = torch.einsum(conversion, *operands)
    except Exception as e:
        print(f"Error occurs! In this function, '{equation}' is converted to '{conversion}'")
        raise e
    return ret
