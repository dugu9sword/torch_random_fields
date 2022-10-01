import einops
import torch

from .misc import compute_energy


def loopy_belief_propagation(
        unary_potentials,  # make yapf happy
        binary_potentials=None,
        binary_edges=None,
        ternary_potentials=None,
        ternary_edges=None,
        max_iter=10,
        damping=.5,
        tol=1e-5,
        track_best=False):
    device = unary_potentials.device
    has_ternary = ternary_edges is not None
    has_binary = binary_edges is not None
    if track_best:
        best_configuration = None
        best_energy = -10000
    # constants
    n_vertices, n_states = unary_potentials.shape

    if has_binary:
        n_bin_edges = binary_edges.shape[0]

    if has_ternary:
        n_ter_edges = ternary_edges.shape[0]

    if has_binary:
        directed_binaries = torch.stack([
            einops.rearrange(binary_potentials, "B E1 E2-> B E2 E1"),  # make yapf happy
            binary_potentials
        ])

    if has_ternary:
        directed_ternaries = torch.stack([
            einops.rearrange(ternary_potentials, "B E0 E1 E2-> B E1 E2 E0"),  # make yapf happy
            einops.rearrange(ternary_potentials, "B E0 E1 E2-> B E0 E2 E1"),
            ternary_potentials
        ])

    # variables to update
    all_incoming_msg = torch.zeros((n_vertices, n_states), device=device)
    if has_binary:
        last_bin_msg = torch.zeros((n_bin_edges, 2, n_states), device=device)
    if has_ternary:
        last_ter_msg = torch.zeros((n_ter_edges, 3, n_states), device=device)

    for _ in range(max_iter):
        diff = 0

        if has_binary:
            for ms, mt in ((0, 1), (1, 0)):
                src_incoming = all_incoming_msg[binary_edges[:, ms]]
                src_incoming_wo_factor = einops.rearrange(src_incoming - last_bin_msg[:, ms], "E S->E S 1")

                src_unary = einops.rearrange(unary_potentials[binary_edges[:, ms]], "E S->E S 1")

                s2t_potential = directed_binaries[mt]

                new_msg = src_incoming_wo_factor + src_unary + s2t_potential
                new_msg = einops.reduce(new_msg, "E S T-> E T", "max")
                new_msg -= einops.reduce(new_msg, "E T-> E 1", "max")
                new_msg = damping * last_bin_msg[:, mt] + (1 - damping) * new_msg
                delta_msg = new_msg - last_bin_msg[:, mt]
                last_bin_msg[:, mt] = new_msg

                delta_incoming = torch.sparse_coo_tensor(
                    indices=torch.stack([binary_edges[:, mt], torch.arange(0, n_bin_edges, device=device)]),  # yapf
                    values=torch.ones([n_bin_edges], device=device),
                    size=(n_vertices, n_bin_edges)).to_dense() @ delta_msg
                delta_incoming = delta_incoming.to(unary_potentials)
                all_incoming_msg += delta_incoming

                diff += torch.abs(delta_msg).sum()

        if has_ternary:
            for ms1, ms2, mt in ((1, 2, 0), (0, 2, 1), (0, 1, 2)):
                s1_incoming = all_incoming_msg[ternary_edges[:, ms1]]
                s2_incoming = all_incoming_msg[ternary_edges[:, ms2]]
                s1_incoming_wo_factor = einops.rearrange(s1_incoming - last_ter_msg[:, ms1], "E S->E S 1 1")
                s2_incoming_wo_factor = einops.rearrange(s2_incoming - last_ter_msg[:, ms2], "E S->E 1 S 1")

                s1_unary = einops.rearrange(unary_potentials[ternary_edges[:, ms1]], "E S->E S 1 1")
                s2_unary = einops.rearrange(unary_potentials[ternary_edges[:, ms2]], "E S->E 1 S 1")

                to_t_potential = directed_ternaries[mt]

                new_msg = s1_incoming_wo_factor + s1_unary + s2_incoming_wo_factor + s2_unary + to_t_potential
                new_msg = einops.reduce(new_msg, "E S1 S2 T->E T", "max")
                new_msg = new_msg - einops.reduce(new_msg, "E T->E 1", "max")
                new_msg = damping * last_ter_msg[:, mt] + (1 - damping) * new_msg
                delta_msg = new_msg - last_ter_msg[:, mt]
                last_ter_msg[:, mt] = new_msg

                delta_incoming = torch.sparse_coo_tensor(
                    indices=torch.stack([ternary_edges[:, mt], torch.arange(0, n_ter_edges, device=device)]),  # yapf
                    values=torch.ones([n_ter_edges], device=device),
                    size=(n_vertices, n_ter_edges)).to_dense() @ delta_msg
                all_incoming_msg += delta_incoming
                diff += torch.abs(delta_msg).sum()
        if track_best:
            configuration = torch.argmax(all_incoming_msg + unary_potentials, axis=1)
            energy = compute_energy(
                unary_potentials=unary_potentials,  # yapf
                binary_potentials=binary_potentials,
                binary_edges=binary_edges,
                ternary_potentials=ternary_potentials,
                ternary_edges=ternary_edges,
                labels=configuration)
            if energy > best_energy:
                best_energy = energy
                best_configuration = configuration

        if diff < tol:
            break
    if track_best:
        return best_configuration
    else:
        return torch.argmax(all_incoming_msg + unary_potentials, axis=1)
