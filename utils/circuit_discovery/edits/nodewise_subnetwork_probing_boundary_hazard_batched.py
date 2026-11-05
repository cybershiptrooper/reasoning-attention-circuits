"""Batched single-pass boundary-hazard subnetwork probing.

Computes exactly the same gradient as
``NodewiseSubnetworkProbingBoundaryHazard`` but restructures the per-step
work to avoid redundant compute. The parent's ``_step_global`` runs, per
training step, 12 no-grad full-length forwards (pass 1, to obtain
per-boundary weights) followed by up to 12 with-grad full-length forwards
+ backwards (pass 2) — every one of which re-encodes the identical
prefix. This class replaces that with:

1. **Single pass.** Every hazard objective in
   ``utils.objectives.HAZARD_OBJECTIVES`` is a plain mean of per-candidate
   terms (no cross-candidate normalisation, unlike the SNIS objectives the
   two-pass structure was built for), so the loss can be computed directly
   on with-grad boundary readouts and backpropagated once. Pass 1
   disappears entirely.

2. **One prefix forward per step, shared by all candidates.** Within a
   step the Hard-Concrete mask sample is fixed and all candidates share
   the same prefix, so the prefix is encoded once (mask installed, with
   grad) and its per-layer K/V are reused by every candidate forward.
   This is exact, not an approximation: with ``mask_mode="prefix"`` the
   circuit mask biases only prefix-position queries — the mask converter's
   fast path returns ``None`` whenever every query token maps to sentence
   -1 (see ``_expand_mask_to_log_additive``) — so candidate-position rows
   never see the mask and causal attention factorises cleanly into
   (prefix self-attention) + (candidate rows attending to cached prefix
   K/V plus themselves).

3. **Length-bucketed batched candidate forwards.** Candidates are sorted
   by length and forwarded in right-padded batches against the shared
   prefix K/V. Right padding is safe under causal attention: real tokens
   never attend to the padded tail, and only real boundary rows are read.

4. **No per-candidate ``gc.collect()/empty_cache()``** (the parent calls
   ``clear_cuda`` after every pass-2 candidate — 6,000 full GC sweeps over
   a 500-step run).

Both passes run under per-layer ``torch.utils.checkpoint`` (non-reentrant)
when the model has gradient checkpointing enabled, mirroring the parent's
memory behaviour; per-layer K/V of the prefix are checkpoint *outputs*, so
they stay in the autograd graph and gradients flow through the shared
prefix into the mask gates.

The forward math is Qwen3-specific in one small place: the prefix pass
recomputes each layer's roped K/V (k_proj -> k_norm -> RoPE, mirroring the
patched attention in ``utils/circuit_discovery/sdpa_forward.py``) so they
can be captured as checkpoint outputs. A guard raises on other model
types.

Registered as ``nodewise_subnetwork_probing_boundary_hazard_batched`` and
(with the continuous answer guard) ``..._probe_weighted_batched``. The
existing classes are untouched; previous runs remain reproducible under
the old algorithm names.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard import (
    NodewiseSubnetworkProbingBoundaryHazard,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard_probe_weighted import (
    NodewiseSubnetworkProbingBoundaryHazardProbeWeighted,
)
from utils.circuit_discovery.sdpa_forward import apply_rotary_pos_emb


class _PrefixConcatKV:
    """Minimal stand-in for a HF ``Cache`` inside the candidate forward.

    ``update()`` returns the (batch-expanded) prefix K/V concatenated with
    the incoming candidate K/V. It never mutates state, so it is safe
    under non-reentrant checkpoint re-runs (unlike ``DynamicCache``, whose
    ``update`` appends in place). The expand is a view; the concat is the
    only materialisation and lives transiently inside one layer's forward.
    """

    def __init__(self, k_prefix: torch.Tensor, v_prefix: torch.Tensor):
        self.k_prefix = k_prefix
        self.v_prefix = v_prefix

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        bsz = key_states.shape[0]
        kp = self.k_prefix.expand(bsz, -1, -1, -1)
        vp = self.v_prefix.expand(bsz, -1, -1, -1)
        return (
            torch.cat([kp, key_states], dim=-2),
            torch.cat([vp, value_states], dim=-2),
        )


class NodewiseSubnetworkProbingBoundaryHazardBatched(
    NodewiseSubnetworkProbingBoundaryHazard
):
    """Boundary-hazard SNP with a single-pass, prefix-KV-shared step."""

    def __init__(self, candidate_batch_size: int = 6, **kwargs):
        self.candidate_batch_size = int(candidate_batch_size)
        self._batched_prepared = False
        self._causal_mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def _prepare_batched(self, continuations: List[torch.Tensor], device):
        if self.model.config.model_type != "qwen3":
            raise NotImplementedError(
                "nodewise_subnetwork_probing_boundary_hazard_batched "
                "implements the prefix-K/V recompute for Qwen3 only "
                f"(got model_type={self.model.config.model_type!r}); use "
                "nodewise_subnetwork_probing_boundary_hazard instead."
            )
        if getattr(self.model.config, "sliding_window", None):
            raise NotImplementedError(
                "Batched hazard step assumes full causal attention; "
                "sliding-window attention is not supported."
            )
        lengths = [int(c.shape[-1]) for c in continuations]
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        bs = self.candidate_batch_size
        if bs <= 0:
            bs = len(order)
        self._cand_buckets: List[List[int]] = [
            order[i : i + bs] for i in range(0, len(order), bs)
        ]
        self._cand_lengths = lengths
        self._batched_prepared = True

    # ------------------------------------------------------------------
    # Attention-mask helpers (constant across steps; cached)
    # ------------------------------------------------------------------

    def _additive_causal_mask(
        self, q_len: int, kv_offset: int, device, dtype
    ) -> torch.Tensor:
        """(1, 1, q_len, kv_offset + q_len) additive causal mask.

        Query row ``i`` (absolute position ``kv_offset + i``) may attend
        to key columns ``0 .. kv_offset + i``. ``kv_offset = 0`` gives the
        standard causal mask for the prefix pass.
        """
        key = (q_len, kv_offset)
        cached = self._causal_mask_cache.get(key)
        if cached is not None and cached.dtype == dtype:
            return cached
        k_len = kv_offset + q_len
        q_pos = torch.arange(q_len, device=device).unsqueeze(-1) + kv_offset
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
        mask.masked_fill_(k_pos > q_pos, torch.finfo(dtype).min)
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._causal_mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self, fn, *args):
        if getattr(self.model, "is_gradient_checkpointing", False):
            return _torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _forward_prefix_collect_kv(self, input_ids: torch.Tensor):
        """Encode the prefix once (mask installed) and return per-layer
        roped K/V as checkpoint outputs (graph-connected).

        The K/V recompute mirrors the patched Qwen3 attention exactly:
        k_proj -> view -> k_norm -> transpose -> RoPE (see
        ``_make_qwen3_forward_sdpa``); recomputing costs 2 of the layer's
        7 projections and avoids touching the shared patched forward.
        """
        base = self.model.model
        device = input_ids.device
        P = input_ids.shape[-1]
        hidden = base.embed_tokens(input_ids)
        position_ids = torch.arange(P, device=device).unsqueeze(0)
        cache_position = torch.arange(P, device=device)
        pos_emb = base.rotary_emb(hidden, position_ids)
        mask = self._additive_causal_mask(P, 0, device, hidden.dtype)

        n_kv = self.model.config.num_key_value_heads
        head_dim = getattr(
            self.model.config, "head_dim",
            self.model.config.hidden_size
            // self.model.config.num_attention_heads,
        )
        cos, sin = pos_emb
        kv_per_layer = []

        for layer in base.layers:
            def _layer_fn(h, _layer=layer):
                attn = _layer.self_attn
                # device_map="auto" (Qwen3-32B over 2 GPUs): the K/V
                # recompute calls submodules directly, so move the input
                # and RoPE tables to this layer's device first.
                ldev = attn.k_proj.weight.device
                h = h.to(ldev)
                c, s = cos.to(ldev), sin.to(ldev)
                hs = _layer.input_layernorm(h)
                k = attn.k_proj(hs).view(1, P, n_kv, head_dim)
                k = attn.k_norm(k).transpose(1, 2)
                v = attn.v_proj(hs).view(1, P, n_kv, head_dim).transpose(1, 2)
                _, k = apply_rotary_pos_emb(k, k, c, s)
                # NB: call .forward directly. In transformers >= 4.56 the
                # decoder layer's __call__ (GradientCheckpointingLayer)
                # intercepts when gradient checkpointing is enabled and the
                # model is in train mode: it would both re-checkpoint this
                # already-checkpointed call and — fatally for the candidate
                # pass — silently set past_key_values=None.
                out = _layer.forward(
                    h,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=pos_emb,
                )
                return out, k, v

            hidden, k, v = self._maybe_checkpoint(_layer_fn, hidden)
            kv_per_layer.append((k, v))
        return kv_per_layer

    def _forward_candidates(
        self,
        bucket_ids: torch.Tensor,
        kv_per_layer,
        prefix_len: int,
    ) -> torch.Tensor:
        """Forward a right-padded candidate batch against the prefix K/V.

        Returns the final-norm hidden states (B, C_max, d). Padded rows
        compute garbage that is never read (causal attention; boundary
        rows are always real tokens).
        """
        base = self.model.model
        device = bucket_ids.device
        B, C = bucket_ids.shape
        hidden = base.embed_tokens(bucket_ids)
        position_ids = (
            torch.arange(prefix_len, prefix_len + C, device=device)
            .unsqueeze(0)
        )
        cache_position = torch.arange(
            prefix_len, prefix_len + C, device=device
        )
        pos_emb = base.rotary_emb(hidden, position_ids)
        mask = self._additive_causal_mask(
            C, prefix_len, device, hidden.dtype
        )

        for layer, (k_p, v_p) in zip(base.layers, kv_per_layer):
            def _layer_fn(h, kp, vp, _layer=layer):
                # .forward directly — see the note in the prefix pass; the
                # intercepting __call__ would strip past_key_values here.
                return _layer.forward(
                    h,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=_PrefixConcatKV(kp, vp),
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=pos_emb,
                )

            hidden = self._maybe_checkpoint(_layer_fn, hidden, k_p, v_p)
        return base.norm(hidden)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _step_global(
        self,
        input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
    ):
        if not self._hazard_prepared:
            self._prepare_hazard_tensors(device, len(continuations))
        if not self._batched_prepared:
            self._prepare_batched(continuations, device)

        lm_w = self.model.lm_head.weight
        log_h_list: List[Optional[torch.Tensor]] = [None] * len(continuations)

        with torch.amp.autocast("cuda"):
            kv_per_layer = self._forward_prefix_collect_kv(input_ids)

            for bucket in self._cand_buckets:
                c_max = max(self._cand_lengths[i] for i in bucket)
                ids = torch.zeros(
                    len(bucket), c_max, dtype=torch.long, device=device,
                )
                for row, i in enumerate(bucket):
                    ids[row, : self._cand_lengths[i]] = continuations[i][0]
                hidden = self._forward_candidates(
                    ids, kv_per_layer, prefix_len,
                )
                # .to() calls are no-ops on one GPU; with device_map="auto"
                # the final hidden states live on the last layer's device.
                hdev = hidden.device
                lm_w_h = lm_w.to(hdev)
                event_ids = self._event_token_ids.to(hdev)
                for row, i in enumerate(bucket):
                    rows = hidden[row, self._bd_positions[i].to(hdev)]  # (B_i, d)
                    logits = (rows @ lm_w_h.T).float()             # fp32
                    log_probs = torch.log_softmax(logits, dim=-1)
                    log_h_list[i] = torch.logsumexp(
                        log_probs[:, event_ids], dim=-1,
                    ).to(device)
                del hidden

        # Per-boundary gradient hooks reproduce the parent's pass-1 weight
        # diagnostics without a second forward.
        grads: List[Optional[torch.Tensor]] = [None] * len(log_h_list)

        def _make_hook(idx):
            def _hook(g):
                grads[idx] = g.detach()
            return _hook

        for i, lh in enumerate(log_h_list):
            lh.register_hook(_make_hook(i))

        loss = self._hazard_fn(
            log_h=log_h_list,
            eligible=self._bd_eligible,
            clean_log_h=self._bd_clean_log_h,
            gaps=self._bd_gaps,
            horizon=self._horizon,
            positions=self._bd_positions,
        )
        task_loss_val = float(loss.detach().item())
        loss.backward()

        with torch.no_grad():
            parts = [g.abs().flatten() for g in grads if g is not None]
            if parts:
                absw = torch.cat(parts)
                absw_sum = float(absw.sum().item())
            else:
                absw_sum = 0.0
            if absw_sum > 0:
                p_absw = absw / absw_sum
                w_entropy = float(
                    -(p_absw * (p_absw + 1e-12).log()).sum().item()
                )
                w_max_share = float(p_absw.max().item())
            else:
                w_entropy, w_max_share = 0.0, 0.0
            self._last_global_diag = {
                "per_boundary_weight_abs_sum": absw_sum,
                "per_boundary_weight_entropy": w_entropy,
                "per_boundary_weight_max_share": w_max_share,
            }
        return task_loss_val


class NodewiseSubnetworkProbingBoundaryHazardProbeWeightedBatched(
    NodewiseSubnetworkProbingBoundaryHazardProbeWeighted,
    NodewiseSubnetworkProbingBoundaryHazardBatched,
):
    """Batched step + continuous answer guard (``probe_p_trace``).

    MRO: the step comes from the batched class, the hazard-tensor
    preparation and objective wrapping from the probe-weighted class.
    """
