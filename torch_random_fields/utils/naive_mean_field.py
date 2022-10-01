import torch

from .misc import compute_energy, einsumx


def naive_mean_field(
        unary_potentials,  # make yapf happy
        binary_potentials,
        binary_edges,
        ternary_potentials=None,
        ternary_edges=None,
        max_iter=10,
        damping=.2,
        tol=1e-5,
        track_best=False):
    device = unary_potentials.device
    if track_best:
        best_configuration = None
        best_energy = -10000

    n_vertices = unary_potentials.shape[0]
    n_bin_edges = binary_potentials.shape[0]

    curr_potentials = unary_potentials
    q = torch.softmax(curr_potentials, 1)

    for i in range(max_iter):
        edge_t_receive = einsumx("E S1 S2, E S1->E S2", binary_potentials, q[binary_edges[:, 0]])
        edge_s_receive = einsumx("E S1 S2, E S2->E S1", binary_potentials, q[binary_edges[:, 1]])
        node_t_receive = torch.sparse_coo_tensor(
            indices=torch.stack([binary_edges[:, 1], torch.arange(0, n_bin_edges, device=device)]),  #
            values=torch.ones([n_bin_edges], device=device),
            size=(n_vertices, n_bin_edges)).to_dense() @ edge_t_receive
        node_s_receive = torch.sparse_coo_tensor(
            indices=torch.stack([binary_edges[:, 0], torch.arange(0, n_bin_edges, device=device)]),  #
            values=torch.ones([n_bin_edges], device=device),
            size=(n_vertices, n_bin_edges)).to_dense() @ edge_s_receive
        last_potentials = curr_potentials
        curr_potentials = node_t_receive + node_s_receive + unary_potentials
        delta = curr_potentials - last_potentials
        curr_potentials = last_potentials * damping + curr_potentials * (1 - damping)
        q = torch.softmax(curr_potentials, 1)
        delta = delta.sum()

        if track_best:
            configuration = torch.argmax(q, axis=1)
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

        if delta < tol:
            break
    if track_best:
        return best_configuration
    else:
        return torch.argmax(q, axis=1)
