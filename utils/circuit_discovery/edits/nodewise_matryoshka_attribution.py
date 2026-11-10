"""Matryoshka attribution (MAttr) over sentence-pair attention masks.

Port of the optimiser of https://github.com/aryamanarora/matryoshka-attribution
(Arora et al., 2026, arXiv:2609.25518; commit af82b2e, ``trainer.learn_scores``,
``sigmoid_topk.SigmoidTopK``, ``schedules.sample_k``) to this repository's
setup. The repository's algorithm, unchanged:

- one real-valued score per unit, initialised at zero;
- every step draws a number of kept units ``k`` (``uniform``:
  ``k ~ Uniform(1, n)``; ``log``: log-uniform on ``[1, n]``);
- the mask of the step is the sigmoid top-k of the scores,
  ``m_i = sigmoid((s_i - tau) / T)``, with the threshold ``tau`` found by
  bisection so that ``sum_i m_i = k``;
- the gradient reaches the scores through the implicit derivative of that
  constraint, ``dL/ds_j = (sp_j / T) * (g_j - sum_i g_i sp_i / sum_i sp_i)``
  with ``sp_i = m_i (1 - m_i)`` and ``g = dL/dm``;
- Adam on the task loss only. There is no sparsity penalty and no target
  sparsity: one run gives an ordering of the units, and the mask at any
  sparsity keeps the top of that ordering.

What differs from the repository is the environment, which is this
repository's and is not changed here:

- a unit is one eligible sentence pair (query sentence i, earlier key
  sentence j) of the reasoning prefix, shared across all layers and heads;
- the mask value enters every attention layer as ``log m`` added to the
  pre-softmax attention scores of the pair's tokens (the repository instead
  interpolates between clean and counterfactual activations);
- the task loss is the configured local objective (``answer_probe_kl``) on
  ``continuations_per_step`` continuations of the rollout bank per step (the
  repository's edge-level script uses one example per step).

Implementation: subclass of ``NodewiseSubnetworkProbingHCBatched`` that reuses
its training loop, rollout-bank plumbing, checkpoints and logging, and
overrides only how the mask of a step is produced (``_sample_masks``), the
optimiser (``_build_optimizer``, to expose Adam's eps) and the diagnostics.
The scores live in ``log_alpha`` so the saved mask (``save_log_alpha: true``)
and the evaluators' top-k readout work unchanged. No existing trainer is
modified.

Config (``masking_algorithm: nodewise_matryoshka_attribution``):

- required: ``optimizer: adam``, ``l0_lambda: 0.0``, ``log_alpha_init: 0.0``,
  ``save_log_alpha: true``, ``mask_granularity: pair``; ``target_sparsity``
  is not used and should be left unset.
- ``learning_rate``, ``num_training_steps``, ``continuations_per_step`` as for
  the other trainers. ``num_hc_samples_per_step`` is the number of
  independent ``k`` draws per step (the repository's ``k_avg``; default 1),
  batched along the batch dimension.
- ``algorithm_kwargs`` (all optional):
  ``mattr_temperature`` (T, default 0.5), ``mattr_k_schedule`` (``uniform`` |
  ``log``), ``mattr_bisection_iters`` (50), ``mattr_adam_eps`` (1e-8),
  ``mattr_eval_sparsities`` (list; the binary top-k mask at each is scored on
  the whole bank every ``bank_eval_every`` steps, rows in
  ``<log_dir>/bank_topk_eval.jsonl``), ``mattr_snapshot_steps`` (list of
  step counts at which the scores are stored; written to
  ``<log_dir>/score_snapshots.npz``),
  ``mattr_init_mask_path`` + ``mattr_init_scale`` (NOT part of the
  repository's recipe: start the scores from another mask's ranking, linearly
  in rank over ``[-scale, +scale]``, instead of from zero).

Registered as ``nodewise_matryoshka_attribution``.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional

import numpy as np
import torch
from torch.autograd import Function

from utils.circuit_discovery.edits.nodewise_deterministic_mask_trainers import (
    n_keep_at_target,
    topk_keep_mask,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _HC_BETA,
)

_EPS = 1e-8
DEFAULT_EVAL_SPARSITIES = (0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Tensor-level pieces (ported from the repository; unit-tested without a model)
# ---------------------------------------------------------------------------

class SigmoidTopK(Function):
    """Sigmoid top-k mask: ``m = sigmoid((s - tau) / T)`` with ``tau`` chosen by
    bisection so that ``m.sum(-1) == k``; backward by implicit differentiation
    of that constraint. ``scores`` is ``[..., n]``; ``k`` is a float or a
    tensor broadcastable to ``[..., 1]``. Computed in float64, returned in the
    dtype of ``scores``."""

    @staticmethod
    def forward(ctx, scores, k, T, n_iters):
        s = scores.double()
        if torch.is_tensor(k):
            k = k.to(device=s.device, dtype=s.dtype)
        lo = s.min(dim=-1, keepdim=True).values - 10 * T
        hi = s.max(dim=-1, keepdim=True).values + 10 * T
        for _ in range(n_iters):
            mid = (lo + hi) / 2
            f_mid = torch.special.expit((s - mid) / T).sum(dim=-1, keepdim=True)
            # the sum is decreasing in tau: too many kept -> raise the lower end
            lo = torch.where(f_mid > k, mid, lo)
            hi = torch.where(f_mid > k, hi, mid)
        tau = (lo + hi) / 2
        mask = torch.special.expit((s - tau) / T)
        ctx.save_for_backward(mask)
        ctx.T = T
        ctx.out_dtype = scores.dtype
        return mask.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        g = grad_output.double()
        sp = mask * (1.0 - mask)
        sp_sum = sp.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        gsp = (g * sp).sum(dim=-1, keepdim=True)
        grad_scores = (sp / ctx.T) * (g - gsp / sp_sum)
        return grad_scores.to(ctx.out_dtype), None, None, None


def sigmoid_topk(scores: torch.Tensor, k, T: float = 0.5, n_iters: int = 50) -> torch.Tensor:
    return SigmoidTopK.apply(scores, k, T, n_iters)


def sample_k(total: int, schedule: str = "uniform") -> float:
    """Number of kept units for one step (a float, as in the repository).
    One ``torch.rand(1)`` draw from the global generator."""
    if schedule == "log":
        return math.exp(math.log(total) * torch.rand(1).item())
    if schedule == "uniform":
        return 1.0 + (total - 1.0) * torch.rand(1).item()
    raise ValueError(f"mattr_k_schedule must be 'uniform' or 'log', got {schedule!r}")


def rank_linear_init(source_scores: torch.Tensor, scale: float) -> torch.Tensor:
    """Scores that reproduce the ordering of ``source_scores`` (1-D), spaced
    linearly in rank over ``[-scale, +scale]`` (highest source score ->
    ``+scale``). Ties keep their index order."""
    n = source_scores.numel()
    order = torch.argsort(source_scores, descending=True, stable=True)
    out = torch.empty(n, dtype=torch.float32, device=source_scores.device)
    if n == 1:
        out[order] = 0.0
        return out
    out[order] = torch.linspace(scale, -scale, n, device=source_scores.device)
    return out


def scatter_pool_mask(pool_mask: torch.Tensor, pool_index: torch.Tensor, num_sents: int) -> torch.Tensor:
    """``(K, n)`` mask values of the learnable pool -> ``(K, S, S)`` mask with
    ones on every other cell. Differentiable in ``pool_mask``."""
    K = pool_mask.shape[0]
    full = torch.ones(K, num_sents * num_sents, dtype=pool_mask.dtype, device=pool_mask.device)
    full = full.index_copy(1, pool_index, pool_mask)
    return full.view(K, num_sents, num_sents)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NodewiseMatryoshkaAttribution(NodewiseSubnetworkProbingHCBatched):
    """Sigmoid top-k mask at a freshly sampled ``k`` every step; Adam on the
    task loss; the result is one ordering of the eligible sentence pairs."""

    ALGORITHM_NAME = "nodewise_matryoshka_attribution"

    def __init__(
        self,
        mattr_temperature: float = 0.5,
        mattr_k_schedule: str = "uniform",
        mattr_bisection_iters: int = 50,
        mattr_adam_eps: float = 1e-8,
        mattr_eval_sparsities: Optional[List[float]] = None,
        mattr_snapshot_steps: Optional[List[int]] = None,
        mattr_init_mask_path: Optional[str] = None,
        mattr_init_scale: float = 2.0,
        bank_eval_every: int = 500,
        **kwargs,
    ):
        kwargs.setdefault("num_hc_samples_per_step", 1)
        kwargs.setdefault("optimizer", "adam")
        kwargs.setdefault("l0_lambda", 0.0)
        kwargs.setdefault("log_alpha_init", 0.0)
        kwargs.setdefault("save_log_alpha", True)
        super().__init__(**kwargs)
        name = self.ALGORITHM_NAME
        if self.mask_granularity != "pair":
            raise NotImplementedError(f"{name} supports pair granularity only")
        if self.l0_lambda != 0.0:
            raise ValueError(f"{name} has no sparsity penalty: set l0_lambda: 0")
        if self.optimizer != "adam":
            raise ValueError(f"{name} uses Adam on the task loss: set optimizer: adam (got {self.optimizer!r})")
        if self.hc_beta_anneal or self.dropout_p > 0.0:
            raise ValueError(f"{name}: hc_beta_anneal and dropout_p do not apply")
        if self.log_alpha_init_mask_path:
            raise ValueError(f"{name}: use algorithm_kwargs.mattr_init_mask_path, not log_alpha_init_mask_path "
                             f"(which binarises the source mask at a target sparsity)")
        if not self.save_log_alpha:
            raise ValueError(f"{name}: save_log_alpha must be true (the saved scores are the ordering)")
        if isinstance(self.log_alpha_init, str) or float(self.log_alpha_init) != 0.0:
            raise ValueError(f"{name}: scores start at zero (log_alpha_init: 0.0); "
                             f"for another start use mattr_init_mask_path")
        if mattr_temperature <= 0.0:
            raise ValueError(f"mattr_temperature must be > 0, got {mattr_temperature}")
        if mattr_k_schedule not in ("uniform", "log"):
            raise ValueError(f"mattr_k_schedule must be 'uniform' or 'log', got {mattr_k_schedule!r}")
        self.mattr_temperature = float(mattr_temperature)
        self.mattr_k_schedule = mattr_k_schedule
        self.mattr_bisection_iters = int(mattr_bisection_iters)
        self.mattr_adam_eps = float(mattr_adam_eps)
        self.mattr_eval_sparsities = [float(s) for s in (mattr_eval_sparsities or DEFAULT_EVAL_SPARSITIES)]
        self.mattr_snapshot_steps = sorted({int(s) for s in (mattr_snapshot_steps or [])})
        self.mattr_init_mask_path = mattr_init_mask_path
        self.mattr_init_scale = float(mattr_init_scale)
        self.bank_eval_every = int(bank_eval_every)
        self._step = 0
        self._combined_filter = None
        self._pool_index = None
        self._theta_ref = None
        self._bank_input_ids = None
        self._bank_kl_all_removed = None
        self._last_ks: List[float] = []
        self._snapshots = {}

    # -- hooks into the base training loop (no base changes) -----------------

    def _current_beta(self, step: int) -> float:
        # The loop calls this first in every step; it is our step counter.
        self._step = int(step)
        return _HC_BETA

    def _build_optimizer(self, params):
        return torch.optim.Adam(params, lr=self.learning_rate, eps=self.mattr_adam_eps)

    def _init_log_alpha(self, granularity, num_heads, num_sents, device, combined_filter=None):
        assert combined_filter is not None, f"{self.ALGORITHM_NAME} needs the eligibility filter"
        self._combined_filter = combined_filter
        learnable = (~combined_filter.bool()).to(device)
        self._pool_index = learnable.flatten().nonzero(as_tuple=True)[0]
        base = torch.zeros(num_sents, num_sents, dtype=torch.float32, device=device)
        if self.mattr_init_mask_path:
            with open(self.mattr_init_mask_path) as f:
                src = torch.tensor(json.load(f)["scores"], dtype=torch.float32, device=device)
            if src.shape != (num_sents, num_sents):
                raise ValueError(f"init mask {self.mattr_init_mask_path} has shape {tuple(src.shape)}, "
                                 f"expected {(num_sents, num_sents)}")
            init = rank_linear_init(src.flatten()[self._pool_index], self.mattr_init_scale)
            base = base.flatten().index_copy(0, self._pool_index, init).view(num_sents, num_sents)
            print(f"  {self.ALGORITHM_NAME}: scores start from the ranking of "
                  f"{os.path.basename(self.mattr_init_mask_path)} (linear in rank, +/-{self.mattr_init_scale:g})")
        return base.unsqueeze(0).clone().requires_grad_(True)

    def _get_clean_logits(self, input_ids, continuations):
        self._bank_input_ids = input_ids
        return super()._get_clean_logits(input_ids, continuations)

    def _learnable(self, device) -> torch.Tensor:
        assert self._combined_filter is not None, "_init_log_alpha must run first"
        return (~self._combined_filter.bool()).to(device)

    # -- the mask of a step ---------------------------------------------------

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        self._theta_ref = log_alpha
        theta = log_alpha[0]
        num_sents = theta.shape[-1]
        idx = self._pool_index.to(theta.device)
        pool_scores = theta.flatten()[idx]
        n = pool_scores.numel()
        K = self._hc_total_samples
        ks = [sample_k(n, self.mattr_k_schedule) for _ in range(K)]
        self._last_ks = ks
        k_t = torch.tensor(ks, dtype=torch.float64, device=theta.device).view(K, 1)
        pool_mask = sigmoid_topk(pool_scores.unsqueeze(0).expand(K, -1), k_t,
                                 T=self.mattr_temperature, n_iters=self.mattr_bisection_iters)
        mask = scatter_pool_mask(pool_mask, idx, num_sents)
        return {l: mask for l in self.layers}

    # -- diagnostics the base loop logs --------------------------------------

    def _current_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        """Sparsity of this step's forward mask: 1 - mean(k) / n."""
        if not self._last_ks or self._pool_index is None:
            return 0.0
        return 1.0 - float(np.mean(self._last_ks)) / max(1, int(self._pool_index.numel()))

    def _expected_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        return self._current_sparsity(log_alpha, granularity, combined_filter, beta)

    def _write_jsonl(self, name: str, row: dict):
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, name), "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- periodic binary evaluation on the whole bank, score snapshots -------

    def _continuations_for_step(self, step, continuations, clean_logits_list, branch_rewards,
                                position_mask_overrides):
        # ``step`` updates have been applied to the scores at this point.
        last = step + 1 == self.num_training_steps
        if step in self.mattr_snapshot_steps and self._theta_ref is not None:
            self._snapshots[int(step)] = self._theta_ref[0].detach().float().cpu().numpy().copy()
        if self.bank_eval_every > 0 and (step % self.bank_eval_every == 0 or last):
            self._bank_topk_eval(step, continuations, clean_logits_list, position_mask_overrides)
        return super()._continuations_for_step(step, continuations, clean_logits_list,
                                               branch_rewards, position_mask_overrides)

    def _bank_losses(self, hard, continuations, clean_logits_list, position_mask_overrides):
        device = hard.device
        input_ids = self._bank_input_ids
        prefix_len = input_ids.shape[-1]
        losses = []
        NodewiseSubnetworkProbingSDPA._install_masks(self, {l: hard.unsqueeze(0) for l in self.layers})
        with torch.no_grad():
            for i, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                position_mask = self._build_position_mask(full_len, prefix_len, device)
                if position_mask_overrides is not None and position_mask_overrides[i] is not None:
                    position_mask = position_mask_overrides[i].to(device)
                clean_logits = clean_logits_list[i][:, :full_len].to(device)
                with torch.amp.autocast("cuda"):
                    logits = self.model(full_input).logits
                loss = self.objective_fn(clean_logits, logits.float(), position_mask, token_ids=full_input)
                losses.append(float(loss.detach().item()))
                del logits, loss
        return losses

    def _bank_topk_eval(self, step, continuations, clean_logits_list, position_mask_overrides):
        """Task loss of the binary top-k mask (exact zeros, the evaluator's
        rule) at every sparsity of ``mattr_eval_sparsities`` on every
        continuation of the bank; the loss with every eligible pair removed
        is computed once and stored with each row."""
        scores = self._theta_ref[0].detach()
        device = scores.device
        learnable = self._learnable(device)
        n_valid = int(learnable.sum().item())
        ones = torch.ones_like(scores, dtype=torch.float32)
        try:
            if self._bank_kl_all_removed is None:
                hard = torch.where(learnable, torch.zeros_like(ones), ones)
                self._bank_kl_all_removed = self._bank_losses(
                    hard, continuations, clean_logits_list, position_mask_overrides)
            for s in self.mattr_eval_sparsities:
                n_keep = n_keep_at_target(n_valid, s)
                keep = topk_keep_mask(scores, learnable, n_keep)
                hard = torch.where(learnable, keep.float(), ones)
                losses = self._bank_losses(hard, continuations, clean_logits_list, position_mask_overrides)
                t = torch.tensor(losses)
                self._write_jsonl("bank_topk_eval.jsonl", {
                    "step": step, "sparsity": s, "n_keep": n_keep, "n_valid": n_valid,
                    "n_continuations": len(losses), "mean": float(t.mean()), "median": float(t.median()),
                    "max": float(t.max()), "per_continuation": losses,
                    "all_removed_per_continuation": self._bank_kl_all_removed,
                })
                print(f"  [bank top-k eval] step {step} sparsity {s:g}: n_keep {n_keep}/{n_valid}, "
                      f"mean KL {float(t.mean()):.5f}, median {float(t.median()):.5f}")
        finally:
            if self._installed_mask_dict is not None:
                NodewiseSubnetworkProbingSDPA._install_masks(self, self._installed_mask_dict)

    def _step_global(self, *args, **kwargs):
        raise NotImplementedError(f"{self.ALGORITHM_NAME} supports local objectives only")

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        if self._snapshots and self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            steps = sorted(self._snapshots)
            np.savez_compressed(os.path.join(self.log_dir, "score_snapshots.npz"),
                                steps=np.array(steps), scores=np.stack([self._snapshots[s] for s in steps]))
        node_mask.algorithm = self.ALGORITHM_NAME
        node_mask.metadata.update({
            "mattr_temperature": self.mattr_temperature,
            "mattr_k_schedule": self.mattr_k_schedule,
            "mattr_bisection_iters": self.mattr_bisection_iters,
            "mattr_adam_eps": self.mattr_adam_eps,
            "mattr_k_draws_per_step": self._hc_total_samples,
            "mattr_eval_sparsities": self.mattr_eval_sparsities,
            "mattr_snapshot_steps": self.mattr_snapshot_steps,
            "mattr_init_mask_path": self.mattr_init_mask_path,
            "mattr_init_scale": self.mattr_init_scale if self.mattr_init_mask_path else None,
            "mattr_source": "github.com/aryamanarora/matryoshka-attribution@af82b2e",
            "bank_eval_every": self.bank_eval_every,
            "num_hc_samples_per_step": 1,
            "hc_sample_batching": False,
            "target_sparsity": None,
            "sparsity_loss_mode": "none (ordering; sigmoid top-k at a sampled k)",
        })
        return node_mask
