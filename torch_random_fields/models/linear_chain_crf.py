from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .constants import Inference, Learning

from ..utils.misc import chain, einsumx


class LinearChainCRF(torch.nn.Module):
    def __init__(
        self,
        num_states,
        low_rank=32,
        beam_size=64,
        feature_size: Optional[int] = None,
        learning: str = Learning.PIECEWISE,
        inference: str = Inference.VITERBI,
    ):
        super().__init__()
        self.num_states = num_states
        self.low_rank = low_rank
        self.beam_size = beam_size
        self.feature_size = feature_size
        self.learning = learning
        self.inference = inference

        assert self.learning in (Learning.EXACT_LIKELIHOOD, Learning.PERCEPTRON,\
            Learning.PIECEWISE, Learning.PSEUDO_LIKELIHOOD)
        assert self.inference in (Inference.VITERBI, Inference.BATCH_MEAN_FIELD)

        self.E1 = nn.Embedding(num_states, low_rank)
        self.E2 = nn.Embedding(num_states, low_rank)

        if self.feature_size is not None:
            self.edge_potential = nn.Sequential(
                nn.Linear(2 * feature_size, low_rank**2),
                nn.ReLU(),
                Rearrange("... (D1 D2) -> ... D1 D2", D1=self.low_rank, D2=self.low_rank),
            )

    def forward(
        self,
        *,
        unaries,
        masks,
        node_features=None,
        targets,
    ):
        if targets is None:
            return self._decode(unaries=unaries, masks=masks, node_features=node_features)

        self.beam_size = min(self.beam_size, unaries.shape[2])
        batch_size, seq_len, _ = unaries.shape

        if node_features is not None:
            # B x (T-1) x D x D
            edge_wise = self.edge_potential(torch.cat([node_features[:, :-1, :], node_features[:, 1:, :]], dim=-1))  
        else:
            edge_wise = torch.eye(self.low_rank, device=unaries.device)
            edge_wise = edge_wise.expand(batch_size, seq_len - 1, self.low_rank, self.low_rank)

        _unaries = unaries.scatter(2, targets[:, :, None], np.float("inf"))
        beam_targets = _unaries.topk(self.beam_size, 2)[1]
        beam_node_phi = unaries.gather(2, beam_targets)

        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])  # B x (T-1) x K x D

        # [checked] B T K1 D1 @ B T D1 D2 -> B T K1 D2
        beam_transition_score1 = beam_transition_score1 @ edge_wise
        beam_edge_phi = einsumx(
            "B T K1 D, B T K2 D -> B T K1 K2",
            beam_transition_score1,
            beam_transition_score2,
        )

        if self.learning == Learning.EXACT_LIKELIHOOD:
            # compute logP
            numerator = self._compute_score(unaries, targets, masks, return_potential=False)

            # compute logZ
            score = unaries[:, 0]  # B x K
            for i in range(1, unaries.size()[1]):
                next_score = score[:, :, None] + beam_edge_phi[:, i - 1]
                next_score = torch.logsumexp(next_score, dim=1) + unaries[:, i]

                if masks is not None:
                    score = torch.where(masks[:, i:i + 1], next_score, score)
                else:
                    score = next_score
            denominator = torch.logsumexp(score, dim=1)

            ll = numerator - denominator
            nll = -(ll / masks.sum(-1)).mean()
            return nll
        elif self.learning == Learning.PERCEPTRON:
            """
            structured perceptron is a type of structural svm (w/o regularization):
            supposing y'=argmax_s, y* is ground truth. (s'>=s*)
            L = max_y' {L_{0-1}(y*, y')+score(y')} - score(y*)
              = max {   0 + score(y*) - score(y*) if y*==y'
                        1 + score(y') - score(y*) if y*!=y'    }
              = max { 0, 1+s'-s* }
              = 1+s'-s*
            """
            _, predictions = self._decode(unaries=unaries, masks=masks, node_features=node_features)
            pred_scores = self._compute_score(unaries, predictions, masks, return_potential=False)
            gold_scores = self._compute_score(unaries, targets, masks, return_potential=False)
            delta = 1 + pred_scores - gold_scores
            loss = (delta / masks.sum(-1)).mean()
            return loss
        if self.learning == Learning.PIECEWISE:
            # normalize logits in a beam, use the gold [:, :, 0, 0] to compute cross entropy
            norm_beam_node_phi = beam_node_phi.log_softmax(dim=-1)
            node_gold_phi = norm_beam_node_phi[:, :, 0].masked_fill(~masks, 0.0)

            norm_beam_edge_phi = chain(
                beam_edge_phi,
                lambda __: einops.rearrange(__, "B T K1 K2 -> B T (K1 K2)"),
                lambda __: __.log_softmax(dim=-1),
                lambda __: einops.rearrange(
                    __,
                    "B T (K1 K2) -> B T K1 K2",
                    K1=self.beam_size,
                    K2=self.beam_size,
                ),
            )
            edge_gold_phi = norm_beam_edge_phi[:, :, 0, 0].masked_fill(~masks[:, 1:], 0.0)
            ll = node_gold_phi.sum(-1) + edge_gold_phi.sum(-1)
            nll = -(ll / masks.sum(-1)).mean()
            return nll
        elif self.learning == Learning.PSEUDO_LIKELIHOOD:
            norm_beam_node_phi = beam_node_phi.log_softmax(dim=-1)
            node_gold_phi = norm_beam_node_phi[:, :, 0].masked_fill(~masks, 0.0)

            l2r_score = chain(
                self.E1(targets[:, :-1]),  # B x (T-1) x D
                lambda __: einsumx("B T D1, B T D1 D2 -> B T D2", __, edge_wise),
                lambda __: einsumx("B T D, V D -> B T V", __, self.E2.weight),
                lambda __: __.log_softmax(dim=-1),
                lambda __: __.gather(-1, targets[:, 1:, None]).squeeze(-1),
                lambda __: __.masked_fill(~masks[:, 1:], 0.0))

            r2l_score = chain(
                self.E2(targets[:, 1:]),  # B x (T-1) x D
                lambda __: einsumx("B T D2, B T D1 D2 -> B T D1", __, edge_wise),
                lambda __: einsumx("B T D, V D -> B T V", __, self.E1.weight),
                lambda __: __.log_softmax(dim=-1),
                lambda __: __.gather(-1, targets[:, :-1, None]).squeeze(-1),
                lambda __: __.masked_fill(~masks[:, :-1], 0.0))

            ll = l2r_score.sum(-1) + r2l_score.sum(-1) + node_gold_phi.sum(-1)
            nll = -(ll / masks.sum(-1)).mean()
            return nll

    def _decode(
        self,
        *,
        unaries,
        masks=None,
        node_features=None,
    ):
        self.beam_size = min(self.beam_size, unaries.shape[2])
        batch_size, seq_len = unaries.size()[:2]

        if node_features is not None:
            # B x (T-1) x D x D
            edge_wise = self.edge_potential(torch.cat([node_features[:, :-1, :], node_features[:, 1:, :]], dim=-1))  
        else:
            edge_wise = torch.eye(self.low_rank, device=unaries.device)
            edge_wise = edge_wise.expand(batch_size, seq_len - 1, self.low_rank, self.low_rank)

        beam_emission_scores, beam_targets = unaries.topk(self.beam_size, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])  # B x (T-1) x K x D

        beam_transition_score1 = beam_transition_score1 @ edge_wise
        beam_transition_matrix = einsumx("B T K1 D, B T K2 D -> B T K1 K2", beam_transition_score1, beam_transition_score2)

        # beam_transition_matrix = torch.zeros_like(beam_transition_matrix)
        beam_transition_matrix = beam_transition_matrix \
            .view(batch_size, seq_len - 1, self.beam_size * self.beam_size) \
            .log_softmax(dim=-1) \
            .view(batch_size, seq_len - 1, self.beam_size, self.beam_size)\
            .type_as(unaries)
        beam_emission_scores = beam_emission_scores \
            .log_softmax(dim=-1) \
            .type_as(beam_transition_matrix)

        if self.inference == Inference.VITERBI:
            traj_tokens, traj_scores = [], []
            finalized_tokens, finalized_scores = [], []

            # compute the normalizer in the log-space
            score = beam_emission_scores[:, 0]  # B x K
            dummy = (torch.arange(self.beam_size, device=score.device).expand(*score.size()).contiguous())

            for i in range(1, seq_len):
                traj_scores.append(score)
                _score = score[:, :, None] + beam_transition_matrix[:, i - 1]
                _score, _index = _score.max(dim=1)
                _score = _score + beam_emission_scores[:, i]

                if masks is not None:
                    score = torch.where(masks[:, i:i + 1], _score, score)
                    index = torch.where(masks[:, i:i + 1], _index, dummy)
                else:
                    score, index = _score, _index
                traj_tokens.append(index)

            # now running the back-tracing and find the best
            best_score, best_index = score.max(dim=1)
            finalized_tokens.append(best_index[:, None])
            finalized_scores.append(best_score[:, None])

            for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
                previous_index = finalized_tokens[-1]
                finalized_tokens.append(idx.gather(1, previous_index))
                finalized_scores.append(scs.gather(1, previous_index))

            finalized_tokens.reverse()
            finalized_tokens = torch.cat(finalized_tokens, 1)
            finalized_tokens = beam_targets.gather(2, finalized_tokens.unsqueeze(-1)).squeeze(-1)

            finalized_scores.reverse()
            finalized_scores = torch.cat(finalized_scores, 1)
            finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

            return finalized_scores, finalized_tokens
        elif self.inference == Inference.BATCH_MEAN_FIELD:
            beam_potential = beam_emission_scores
            beam_q = F.softmax(beam_potential, -1)

            max_iter = 5
            damping = 0.5

            for _ in range(max_iter):
                beam_left_sum = torch.zeros_like(beam_potential)
                beam_right_sum = torch.zeros_like(beam_potential)

                beam_left_sum[:, 1:] = einsumx("B L D1, B L D1 D2->B L D2", beam_q[:, :-1, :], beam_transition_matrix)
                beam_right_sum[:, :-1] = einsumx("B L D2, B L D1 D2->B L D1", beam_q[:, 1:, :], beam_transition_matrix)

                beam_sum = beam_left_sum + beam_right_sum

                last_beam_potential = beam_potential
                beam_potential = beam_potential + beam_sum
                beam_potential = last_beam_potential * damping + beam_potential * (1 - damping)
                beam_potential = beam_potential * masks.unsqueeze(-1)
                beam_q = F.softmax(beam_potential, -1)
            finalized_scores, finalized_tokens = beam_potential.max(-1)
            finalized_tokens = beam_targets.gather(2, finalized_tokens.unsqueeze(-1)).squeeze(-1)
            return finalized_scores, finalized_tokens
        else:
            raise NotImplementedError

    def _compute_score(self, unaries, paths, masks, return_potential=False):
        # unary
        unaries = unaries.gather(-1, paths.unsqueeze(-1)).squeeze(-1)
        unaries = unaries * masks
        # binary
        left = self.E1(paths[:, :-1])
        right = self.E2(paths[:, 1:])
        binaries = (left * right).sum(-1)
        binaries = binaries * masks[:, 1:]

        if return_potential:
            return unaries, binaries
        else:
            scores = unaries.sum(-1) + binaries.sum(-1)
            return scores
