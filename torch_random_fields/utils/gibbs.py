import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def gibbs_sampling(
        *,
        unary_potentials=None,  # make yapf happy
        binary_potentials=None,
        binary_edges=None,
        sample_size=1000,
        burnin_size=1000):
    """
        Input:
            unary_potentials:   (num_nodes, num_states)
            binary_potentials:  (num_edges, num_states, num_states)
            binary_edges:       (num_edges, 2)
    """
    device = unary_potentials.device
    num_node = len(unary_potentials)
    num_state = len(unary_potentials[0])
    source_bin_map = {}
    target_bin_map = {}
    binary_edges = binary_edges.tolist()
    for idx, (s, t) in enumerate(binary_edges):
        if s not in source_bin_map:
            source_bin_map[s] = []
        source_bin_map[s].append(idx)
        if t not in target_bin_map:
            target_bin_map[t] = []
        target_bin_map[t].append(idx)

    cur_unary_state = {
        n: random.randint(0, num_state - 1)
        for n in range(num_node)
    }
    samples = []
    for ite in tqdm(range(sample_size + burnin_size)):
        sample = []
        order = list(range(num_node))
        random.shuffle(order)
        reorder = {o: idx for idx, o in enumerate(order)}
        for cur in order:
            logit = torch.zeros(unary_potentials[0].size(),
                                device=device,
                                requires_grad=False)
            logit += unary_potentials[cur]
            if cur in source_bin_map:
                for idx in source_bin_map[cur]:
                    _, nbr = binary_edges[idx]
                    nbr_state = cur_unary_state[nbr]
                    logit += np.transpose(binary_potentials[idx][:, nbr_state])
            if cur in target_bin_map:
                for idx in target_bin_map[cur]:
                    nbr, _ = binary_edges[idx]
                    nbr_state = cur_unary_state[nbr]
                    logit += binary_potentials[idx][nbr_state, :]
            p = F.softmax(logit, dim=0)
            cur_state = torch.multinomial(p, num_samples=1).int().item()
            cur_unary_state[cur] = cur_state
            sample.append(cur_state)
        samples.append([sample[reorder[i]] for i in range(len(sample))])
    return samples[burnin_size:]
