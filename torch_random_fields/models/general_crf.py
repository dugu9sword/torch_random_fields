from typing import Optional

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from ..utils.batch_lbp import batch_index_select, batch_lbp
from ..utils.loopy_belief_propagation import loopy_belief_propagation
from ..utils.misc import batch_pad, chain, einsumx
from ..utils.naive_mean_field import naive_mean_field
from .constants import Inference, Learning


class GeneralCRF(torch.nn.Module):
    def __init__(
        self,
        num_states,
        low_rank=32,
        beam_size=64,
        support_ternary: bool = False,
        feature_size: Optional[int] = None,
        learning: str = Learning.PIECEWISE,
        inference: str = Inference.BATCH_BELIEF_PROPAGATION,
    ) -> None:
        super().__init__()
        assert learning in (
            Learning.PIECEWISE,
            Learning.PERCEPTRON,
        )
        assert inference in (
            Inference.MEAN_FIELD,
            Inference.BELIEF_PROPAGATION,
            Inference.BATCH_BELIEF_PROPAGATION,
        )
        if support_ternary:
            assert learning == Learning.PIECEWISE and \
                 inference == Inference.BELIEF_PROPAGATION, \
            f"only learning={Learning.PIECEWISE} and inference={Inference.BELIEF_PROPAGATION} supports ternary potentials!"

        self.num_states = num_states
        self.low_rank = low_rank
        self.beam_size = beam_size
        self.feature_size = feature_size
        self.learning = learning
        self.inference = inference
        self.support_ternary = support_ternary

        self.bin_embed = nn.Parameter(torch.rand(2, self.num_states, self.low_rank))
        if self.support_ternary:
            self.ter_embed = nn.Parameter(torch.rand(3, self.num_states, self.low_rank))

        if self.feature_size is not None:
            self.bin_dynamic = nn.Sequential(
                nn.Linear(2 * feature_size, low_rank**2),
                nn.ReLU(),
                Rearrange("... (D1 D2) -> ... D1 D2", D1=self.low_rank, D2=self.low_rank),
            )
            if self.support_ternary:
                self.ter_dynamic = nn.Sequential(
                    nn.Linear(3 * feature_size, low_rank**3),
                    nn.ReLU(),
                    Rearrange("... (D1 D2 D3) -> ... D1 D2 D3", D1=self.low_rank, D2=self.low_rank, D3=self.low_rank),
                )

    def forward(
        self,
        *,
        unaries: torch.Tensor,
        masks: torch.Tensor,
        binary_edges: torch.Tensor,
        binary_masks: torch.Tensor,
        ternary_edges: torch.Tensor = None,
        ternary_masks: torch.Tensor = None,
        node_features: torch.Tensor = None,
        targets: torch.Tensor = None,
    ):
        batch_size, num_nodes, _ = unaries.shape
        unaries = unaries * masks.unsqueeze(-1)

        self.beam_size = min(self.beam_size, unaries.shape[2])
        """ build unary potentials """
        if targets is not None:
            # a nice property:
            #   the target word will be the first word in the beam,
            #   although it may have a low score
            _unaries = unaries.scatter(2, targets[:, :, None], float("inf"))
            beam_targets = _unaries.topk(self.beam_size, 2)[1]
            beam_unary_potentials = unaries.gather(2, beam_targets)
        else:
            beam_targets = unaries.topk(self.beam_size, 2)[1]
            beam_unary_potentials = unaries.gather(2, beam_targets)
        """ build binary potentials """
        # beam_targets: bsz x node x beam
        # edge:         bsz x edge
        # score:        bsz x edge x beam x rank
        # bin_phis:     bsz x edge x beam x beam
        #               the [bid][i][a][b] value denotes:
        #                   potential(i-th edge connecting a & b)
        bin_edge0, bin_edge1 = binary_edges[..., 0], binary_edges[..., 1]
        bin_score0 = F.embedding(batch_index_select(beam_targets, bin_edge0), self.bin_embed[0])
        bin_score1 = F.embedding(batch_index_select(beam_targets, bin_edge1), self.bin_embed[1])
        if node_features is not None:
            bin_feat0 = batch_index_select(node_features, bin_edge0)
            bin_feat1 = batch_index_select(node_features, bin_edge1)
            bin_edge_wise = self.bin_dynamic(torch.cat([bin_feat0, bin_feat1], dim=-1))
            bin_phis = einsumx("B E K1 D1, B E D1 D2, B E K2 D2 -> B E K1 K2", bin_score0, bin_edge_wise, bin_score1)
            # bin_phis = einsumx("B E K1 D2, B E K2 D2 -> B E K1 K2", bin_score0 @ bin_edge_wise, bin_score1)
        else:
            bin_edge_wise = None
            bin_phis = einsumx("B E K1 D1, B E K2 D2 -> B E K1 K2", bin_score0, bin_score1)
        """ build ternary potentials """
        if self.support_ternary:
            ter_edge0, ter_edge1, ter_edge2 = ternary_edges[..., 0], ternary_edges[..., 1], ternary_edges[..., 2]
            ter_score0 = F.embedding(batch_index_select(beam_targets, ter_edge0), self.ter_embed[0])
            ter_score1 = F.embedding(batch_index_select(beam_targets, ter_edge1), self.ter_embed[1])
            ter_score2 = F.embedding(batch_index_select(beam_targets, ter_edge2), self.ter_embed[2])
            if node_features is not None:
                ter_feat0 = batch_index_select(node_features, ter_edge0)
                ter_feat1 = batch_index_select(node_features, ter_edge1)
                ter_feat2 = batch_index_select(node_features, ter_edge2)
                ter_edge_wise = self.ter_dynamic(torch.cat([ter_feat0, ter_feat1, ter_feat2], dim=-1))
                ter_phis = einsumx("B E K1 D1, B E D1 D2 D3, B E K2 D2, B E K3 D3 -> B E K1 K2 K3", ter_score0, ter_edge_wise, ter_score1, ter_score2)
            else:
                ter_edge_wise = None
                ter_phis = einsumx("B E K1 D1, B E K2 D2, B E K3 D3 -> B E K1 K2 K3", ter_score0, ter_score1, ter_score2)
        else:
            ter_phis = None

        if targets is not None:
            if self.learning == Learning.PIECEWISE:
                # unary
                norm_unary = beam_unary_potentials.log_softmax(-1)
                gold_unary = norm_unary[:, :, 0]
                gold_unary = gold_unary.masked_fill(~masks, 0.)
                pll = gold_unary.sum(-1)
                # binary
                norm_bin_phis = chain(
                    bin_phis,  #
                    lambda __: einops.rearrange(__, "B E K1 K2 -> B E (K1 K2)"),
                    lambda __: __.log_softmax(-1),
                    lambda __: einops.rearrange(__, "B E (K1 K2) -> B E K1 K2", K1=self.beam_size, K2=self.beam_size),
                )
                gold_bin_phis = norm_bin_phis[:, :, 0, 0]
                gold_bin_phis = gold_bin_phis.masked_fill(~binary_masks, 0.)
                pll = pll + gold_bin_phis.sum(-1)
                # ternary
                if self.support_ternary:
                    norm_ter_phis = chain(
                        ter_phis,
                        lambda __: einops.rearrange(__, "B E K1 K2 K3 -> B E (K1 K2 K3)"),
                        lambda __: __.log_softmax(-1),
                        lambda __: einops.rearrange(__, "B E (K1 K2 K3) -> B E K1 K2 K3", K1=self.beam_size, K2=self.beam_size, K3=self.beam_size),
                    )
                    gold_ter_phis = norm_ter_phis[:, :, 0, 0, 0]
                    gold_ter_phis = gold_ter_phis.masked_fill(~ternary_masks, 0.0)
                    pll = pll + gold_ter_phis.sum(-1)
                # nll
                nll = -(pll / masks.sum(-1)).mean()
                return nll
            elif self.learning == Learning.PERCEPTRON:
                # raise NotImplemented
                _, pred_idx = self(
                    unaries=unaries,
                    masks=masks,
                    binary_edges=binary_edges,
                    binary_masks=binary_masks,
                    ternary_edges=ternary_edges,
                    ternary_masks=ternary_masks,
                    node_features=node_features,
                )

                def score(index):
                    unary = unaries.gather(2, index.unsqueeze(-1)).squeeze(-1) * masks
                    bin_tran0 = F.embedding(batch_index_select(index.unsqueeze(-1), bin_edge0).squeeze(-1), self.bin_embed[0])
                    bin_tran1 = F.embedding(batch_index_select(index.unsqueeze(-1), bin_edge1).squeeze(-1), self.bin_embed[1])
                    binary = einsumx(
                        "B E D1, B E D1 D2, B E D2->B E",  #
                        bin_tran0,
                        bin_edge_wise,
                        bin_tran1) * binary_masks
                    ret = unary.sum(-1) + binary.sum(-1)
                    if self.support_ternary:
                        ter_tran0 = F.embedding(batch_index_select(index.unsqueeze(-1), ter_edge0).squeeze(-1), self.ter_embed[0])
                        ter_tran1 = F.embedding(batch_index_select(index.unsqueeze(-1), ter_edge1).squeeze(-1), self.ter_embed[1])
                        ter_tran2 = F.embedding(batch_index_select(index.unsqueeze(-1), ter_edge2).squeeze(-1), self.ter_embed[2])
                        ternary = einsumx(
                            "B E D1, B E D1 D2 D3, B E D2, B E D3, -> B E", #
                            ter_tran0, ter_edge_wise, ter_tran1, ter_tran2,
                        ) * ternary_masks
                        ret = ret + ternary.sum(-1)
                    return ret

                delta = 1 + score(pred_idx) - score(targets)
                loss = (delta / masks.sum(-1)).mean()
                return loss
            else:
                raise NotImplementedError
        else:
            if self.inference == Inference.BATCH_BELIEF_PROPAGATION:
                _, pred_idx = batch_lbp(
                    bat_unary_potentials=beam_unary_potentials,
                    bat_unary_masks=masks,
                    bat_binary_potentials=bin_phis,
                    bat_binary_edges=binary_edges,
                    bat_binary_masks=binary_masks,
                    max_iter=10,
                    damping=0.5,
                )
                pred_idx[pred_idx == -1] = 0
                pred_idx = chain(
                    pred_idx,
                    lambda __: beam_targets.gather(-1, __.unsqueeze(-1)).squeeze(-1),
                    lambda __: __.masked_fill_(~masks, 0),
                )
                return None, pred_idx
            elif self.inference in (Inference.MEAN_FIELD, Inference.BELIEF_PROPAGATION):

                def infer_one_sentence(bid):
                    node_len = masks[bid].sum(-1)
                    bin_edge_len = binary_masks[bid].sum(-1)

                    kwargs = dict(
                        unary_potentials=beam_unary_potentials[bid][:node_len],
                        binary_potentials=bin_phis[bid][:bin_edge_len],
                        binary_edges=binary_edges[bid][:bin_edge_len],
                        max_iter=10,
                        track_best=True,
                    )
                    if self.support_ternary:
                        ter_edge_len = ternary_masks[bid].sum(-1)
                        kwargs.update(dict(
                            ternary_potentials=ter_phis[bid][:ter_edge_len],
                            ternary_edges=ternary_edges[bid][:ter_edge_len],
                        ))
                    if self.inference == Inference.MEAN_FIELD:
                        ret = naive_mean_field(**kwargs)
                    elif self.inference == Inference.BELIEF_PROPAGATION:
                        ret = loopy_belief_propagation(**kwargs)
                    return ret

                pred_idx = [infer_one_sentence(bid).tolist() for bid in range(batch_size)]

                pred_idx = chain(
                    pred_idx,
                    lambda __: batch_pad(__, 0, pad_len=num_nodes),
                    lambda __: torch.LongTensor(__, device=beam_unary_potentials.device),
                    lambda __: beam_targets.gather(-1, __.unsqueeze(-1)).squeeze(-1),
                    lambda __: __.masked_fill_(~masks, 0),
                )

                return None, pred_idx

            elif self.inference == Inference.NONE:
                pred_idx = beam_unary_potentials.argmax(-1)
                return None, pred_idx
