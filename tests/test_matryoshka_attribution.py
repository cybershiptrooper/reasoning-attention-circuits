"""Unit tests for the tensor-level pieces of the Matryoshka attribution trainer
(no model needed)."""
import math

import pytest
import torch

from utils.circuit_discovery.edits.nodewise_matryoshka_attribution import (
    rank_linear_init, sample_k, scatter_pool_mask, sigmoid_topk,
)


def _reference_topk(scores, k, T, n_iters=200):
    """Sigmoid top-k written with plain autograd: tau solves sum sigmoid((s - tau)/T) = k,
    and the dependence of tau on the scores is put back with one Newton step."""
    s = scores.double()
    lo, hi = s.min() - 10 * T, s.max() + 10 * T
    with torch.no_grad():
        for _ in range(n_iters):
            mid = (lo + hi) / 2
            if torch.sigmoid((s - mid) / T).sum() > k:
                lo = mid
            else:
                hi = mid
        tau0 = (lo + hi) / 2
    m0 = torch.sigmoid((s - tau0) / T)
    sp = (m0 * (1 - m0)).detach()
    # tau as a differentiable function of s around tau0: d tau / d s_j = sp_j / sum(sp)
    tau = tau0 + ((s - s.detach()) * sp).sum() / sp.sum()
    return torch.sigmoid((s - tau) / T)


@pytest.mark.parametrize("k", [1.0, 3.7, 10.0, 24.2])
def test_mask_sums_to_k_and_is_ordered(k):
    torch.manual_seed(0)
    scores = torch.randn(25)
    m = sigmoid_topk(scores, k, T=0.5, n_iters=50)
    assert m.dtype == scores.dtype
    assert abs(float(m.double().sum()) - k) < 1e-4
    assert torch.all((m > 0) & (m < 1))
    order = torch.argsort(scores)
    assert torch.all(m[order][1:] >= m[order][:-1])          # monotone in the score


def test_zero_scores_give_the_uniform_mask():
    m = sigmoid_topk(torch.zeros(40), 10.0, T=0.5)
    assert torch.allclose(m, torch.full((40,), 0.25), atol=1e-6)


def test_gradient_matches_plain_autograd_reference():
    torch.manual_seed(1)
    scores = torch.randn(30, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(30, dtype=torch.float64)
    k, T = 11.3, 0.5
    (sigmoid_topk(scores, k, T=T, n_iters=80) * upstream).sum().backward()
    ref = scores.detach().clone().requires_grad_(True)
    (_reference_topk(ref, k, T) * upstream).sum().backward()
    assert torch.allclose(scores.grad, ref.grad, atol=1e-8)
    # a shift of all upstream gradients by a constant leaves the score gradient unchanged
    scores2 = scores.detach().clone().requires_grad_(True)
    (sigmoid_topk(scores2, k, T=T, n_iters=80) * (upstream + 5.0)).sum().backward()
    assert torch.allclose(scores.grad, scores2.grad, atol=1e-8)
    assert abs(float(scores.grad.sum())) < 1e-8             # the constraint removes the mean direction


def test_gradient_matches_finite_differences():
    torch.manual_seed(2)
    scores = torch.randn(12, dtype=torch.float64, requires_grad=True)
    w = torch.randn(12, dtype=torch.float64)
    f = lambda s: (sigmoid_topk(s, 4.5, T=0.5, n_iters=100) ** 2 * w).sum()
    f(scores).backward()
    eps = 1e-5
    for j in range(12):
        d = torch.zeros(12, dtype=torch.float64)
        d[j] = eps
        fd = (f(scores.detach() + d) - f(scores.detach() - d)) / (2 * eps)
        assert abs(float(fd) - float(scores.grad[j])) < 1e-6


def test_batched_rows_take_their_own_k():
    torch.manual_seed(3)
    scores = torch.randn(20)
    ks = torch.tensor([[2.0], [9.5], [17.0]], dtype=torch.float64)
    m = sigmoid_topk(scores.unsqueeze(0).expand(3, -1), ks, T=0.5)
    assert m.shape == (3, 20)
    assert torch.allclose(m.double().sum(-1, keepdim=True), ks, atol=1e-4)
    for r in range(3):
        assert torch.allclose(m[r], sigmoid_topk(scores, float(ks[r]), T=0.5), atol=1e-6)


def test_log_mask_gradient_is_finite_for_small_mask_values():
    """The trainer's mask enters the model as log(mask): the 1/m factor of that
    log cancels against the m(1-m) slope of the sigmoid, so far-below-threshold
    scores still get a finite gradient."""
    scores = torch.tensor([30.0, 0.5, 0.0, -0.5, -30.0], requires_grad=True)
    m = sigmoid_topk(scores, 2.0, T=0.5)
    leaf = m.detach().requires_grad_(True)
    (torch.log(leaf.clamp_min(1e-30)) * torch.tensor([1.0, -2.0, 3.0, 0.5, 1.5])).sum().backward()
    m.backward(gradient=leaf.grad)
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().max() < 1e3


def test_sample_k_ranges():
    torch.manual_seed(0)
    u = [sample_k(500, "uniform") for _ in range(2000)]
    assert min(u) >= 1.0 and max(u) <= 500.0
    assert 230 < sum(u) / len(u) < 270
    lg = [sample_k(500, "log") for _ in range(2000)]
    assert min(lg) >= 1.0 and max(lg) <= 500.0
    assert abs(sum(math.log(x) for x in lg) / len(lg) - math.log(500) / 2) < 0.2
    with pytest.raises(ValueError):
        sample_k(10, "natural")


def test_rank_linear_init_keeps_the_ordering():
    src = torch.tensor([0.3, 9.0, -1.0, 0.3, 2.0])
    init = rank_linear_init(src, scale=2.0)
    assert float(init.max()) == 2.0 and float(init.min()) == -2.0
    assert torch.equal(torch.argsort(init, descending=True), torch.argsort(src, descending=True, stable=True))
    assert torch.allclose(init.sort().values.diff(), torch.full((4,), 1.0))


def test_scatter_pool_mask_leaves_other_cells_at_one_and_passes_gradient():
    S = 4
    learnable = torch.zeros(S, S, dtype=torch.bool)
    learnable[2, 1] = learnable[3, 1] = learnable[3, 2] = True
    idx = learnable.flatten().nonzero(as_tuple=True)[0]
    pool = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], requires_grad=True)
    full = scatter_pool_mask(pool, idx, S)
    assert full.shape == (2, S, S)
    assert torch.equal(full[:, ~learnable], torch.ones(2, S * S - 3))
    assert torch.allclose(full[:, learnable], pool)
    full.sum().backward()
    assert torch.equal(pool.grad, torch.ones_like(pool))


def test_algorithm_is_registered():
    from utils.circuit_discovery.factory import get_available_algorithms
    assert "nodewise_matryoshka_attribution" in get_available_algorithms()
