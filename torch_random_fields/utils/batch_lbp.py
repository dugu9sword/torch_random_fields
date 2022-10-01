import einops
import torch

from .misc import chain, einsumx


def batch_index_select(input, index):
    """
        input: (B, N, H)
        index: (B, I)

        [inp[idx] for idx, inp in enumerate(index, input)]

        case:
            
        batch_index_select(
            torch.tensor([[[0], [10], [20]], [[30], [40], [50]], [[60], [70], [80]]]),  # yapf ^_^
            torch.tensor([[1, 1], [2, 1], [0, 1]])
        ).reshape(3, 2)
    """
    device = input.device
    assert len(input.shape) == 3
    assert len(index.shape) == 2
    bsz, num, _ = input.shape
    bsz, num_index = index.shape
    index_1d = torch.arange(bsz, device=device).reshape(bsz, 1) * num + index
    index_1d = einops.rearrange(index_1d, "B N->(B N)")
    input_2d = einops.rearrange(input, "B N H->(B N) H")
    selected = einops.rearrange(input_2d[index_1d], "(B N) H->B N H", B=bsz, N=num_index)
    return selected


def batch_lbp(
    bat_unary_potentials,
    bat_unary_masks,
    bat_binary_edges,
    bat_binary_potentials,
    bat_binary_masks,
    max_iter=10,
    damping=0.5,
    tol=1e-5,
):
    # constants
    bsz, n_vertices, n_states = bat_unary_potentials.shape
    _, n_bin_edges, _, _ = bat_binary_potentials.shape

    directed_binaries = torch.stack([
        einops.rearrange(bat_binary_potentials, "B NE S T-> B NE T S"),  # make yapf happy
        bat_binary_potentials
    ])
    
    new_tensor_args = dict(dtype=bat_binary_potentials.dtype, device=bat_binary_potentials.device)
    all_incoming_msg = torch.zeros((bsz, n_vertices, n_states), **new_tensor_args)
    last_bin_msg = torch.zeros((bsz, n_bin_edges, 2, n_states), **new_tensor_args)

    for _ in range(max_iter):
        diff = 0
        for ms, mt in ((0, 1), (1, 0)):
            src_incoming = batch_index_select(all_incoming_msg, bat_binary_edges[..., ms])
            src_incoming_wo_factor = einops.rearrange(src_incoming - last_bin_msg[:, :, ms, :], "B NE S->B NE S 1")

            src_unary = batch_index_select(bat_unary_potentials, bat_binary_edges[..., ms])
            src_unary = einops.rearrange(src_unary, "B NE S -> B NE S 1")

            s2t_potential = directed_binaries[mt]

            new_msg = chain(
                src_incoming_wo_factor + src_unary + s2t_potential,
                lambda __: einops.reduce(__, "B NE S T -> B NE T", "max"),
                lambda __: __ - einops.reduce(__, "B NE T -> B NE 1", "max"),
                lambda __: damping * last_bin_msg[:, :, mt, :] + (1 - damping) * __,
            )
            """
                delta_msg: (B, NE, H)
                    it stores the message on an edge, a node can receive messages from
                    several edges.
                src_idx:   (B, NE)
                    it stores the index of the src node in an edge
                
                we construct an matrix (B, NV, NE) to aggregate messages from edges to nodes.
                for each batch and each edge, put 1 in the node dimension.
            """
            delta_msg = new_msg - last_bin_msg[:, :, mt, :]
            last_bin_msg[:, :, mt, :] = new_msg

            agg_matrix = torch.zeros([bsz, n_vertices, n_bin_edges], **new_tensor_args)
            src_idx = bat_binary_edges[..., mt]
            agg_matrix.scatter_(
                dim=1, # yapf
                index=einops.rearrange(src_idx, "B NE -> B 1 NE"),
                src=einops.rearrange(bat_binary_masks, "B NE -> B 1 NE") + 0.0,  # mask unused edges
            )

            delta_incoming = einsumx("B NV NE, B NE H->B NV H", agg_matrix, delta_msg)

            all_incoming_msg += delta_incoming

            diff += torch.abs(delta_msg * einops.rearrange(bat_binary_masks, "B NE->B NE 1")).sum() / bsz

        if diff < tol:
            break

    belief = all_incoming_msg + bat_unary_potentials
    belief = belief * einops.rearrange(bat_unary_masks, "B NV->B NV 1")
    ret = torch.argmax(belief, axis=2)
    ret[~bat_unary_masks] = -1
    return belief, ret
    