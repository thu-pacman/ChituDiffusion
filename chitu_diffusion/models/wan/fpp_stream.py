"""Cross-step FPP feedback from SmartDiffusion exp/fpp@b64ddaa.

A segment spans consecutive asynchronous denoising steps. Only its first and
last boundaries synchronize the world. The last stage owns sampler history;
the first stage consumes candidate latents with a delay of P microbatches.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...parallel.cp.nccl.agkv import all_gather_sequence
from ...parallel.pp import (
    FppAttentionContext,
    FppState,
    PipelineTransport,
    fpp_attention_context,
)
from ...parallel.pp.layout import FppTokenLayout
from .fpp_scheduler import fpp_scheduler_step, validate_fpp_scheduler


class _WanStage:
    def __init__(self, model, state, mesh, layout, *, reference):
        self.model, self.state, self.mesh, self.layout = model, state, mesh, layout
        self.reference = reference
        self.device = state.latents.device
        self.transport = PipelineTransport(mesh.pipeline)
        self.features = {}
        raw_rope = model.rope(state.latents.to(model.dtype))
        pad = layout.padded_tokens - layout.tokens
        # Legacy complex RoPE padding is 1: cosine=1, sine=0, with no mask.
        self.rope = tuple(
            torch.cat(
                [
                    r,
                    torch.full(
                        (r.shape[0], pad, *r.shape[2:]),
                        fill,
                        dtype=r.dtype,
                        device=r.device,
                    ),
                ],
                dim=1,
            )
            for r, fill in zip(raw_rope, (1, 0), strict=True)
        )
        self.cp_rank = None if reference else mesh.cp_rank
        self.cache_tokens = layout.padded_tokens if reference else layout.local_tokens
        self.branches = (
            (
                "uncond"
                if mesh.cfg_rank and state.do_classifier_free_guidance
                else "cond",
            )
            if mesh.cfg_degree == 2
            else (
                ("cond", "uncond") if state.do_classifier_free_guidance else ("cond",)
            )
        )

    def prepare_step(self, step):
        t = self.state.timesteps[step].expand(self.state.actual_batch_size)
        self.conditioning = {}
        for branch in self.branches:
            embeds = (
                self.state.prompt_embeds
                if branch == "cond"
                else self.state.negative_prompt_embeds
            )
            temb, projection, context, _ = self.model.condition_embedder(
                t, embeds, None, timestep_seq_len=None
            )
            self.conditioning[branch] = (
                temb,
                projection.unflatten(1, (6, -1)),
                context,
            )

    def forward(self, latents, patch):
        indices = self.layout.indices(patch, self.cp_rank, device=self.device)
        local_indices = (
            indices
            if self.reference
            else indices - self.mesh.cp_rank * self.layout.local_tokens
        )
        topology, model = self.mesh.pipeline, self.model
        if topology.first:
            embedded = (
                model.patch_embedding(latents.to(model.dtype))
                .flatten(2)
                .transpose(1, 2)
            )
            embedded = F.pad(
                embedded, (0, 0, 0, self.layout.padded_tokens - self.layout.tokens)
            )
            inputs = [embedded.index_select(1, indices)] * len(self.branches)
        else:
            shape = (
                len(self.branches),
                self.state.actual_batch_size,
                indices.numel(),
                model.config.num_attention_heads * model.config.attention_head_dim,
            )
            inputs = self.transport.receive(
                shape, dtype=model.dtype, device=self.device
            ).unbind(0)
        rotary = tuple(value.index_select(1, indices) for value in self.rope)
        begin, end = model.pipeline_layer_range
        outputs = []
        for branch, tokens in zip(self.branches, inputs, strict=True):
            _, projection, context = self.conditioning[branch]
            with fpp_attention_context(
                FppAttentionContext(
                    self.state.fpp_state.cache,
                    branch,
                    None,
                    self.cache_tokens,
                    None if patch is None else local_indices,
                    self.mesh.cp_group,
                )
            ):
                for block in model.blocks[begin:end]:
                    tokens = block(tokens, context, projection, rotary)
            outputs.append(tokens)
        if not topology.last:
            self.transport.send(torch.stack(outputs))
            return None
        predictions = []
        for branch, tokens in zip(self.branches, outputs, strict=True):
            if patch is None:
                self.features[branch] = tokens.clone()
            else:
                self.features[branch].index_copy_(1, local_indices, tokens)
            full = all_gather_sequence(
                self.features[branch].unsqueeze(-1), self.mesh.cp_group
            ).squeeze(-1)
            predictions.append(self.project(full[:, : self.layout.tokens], branch))
        if self.mesh.cfg_degree == 2:
            gathered = [torch.empty_like(predictions[0]) for _ in range(2)]
            dist.all_gather(
                gathered, predictions[0].contiguous(), group=self.mesh.cfg_group
            )
            predictions = (
                gathered if self.state.do_classifier_free_guidance else gathered[:1]
            )
        if len(predictions) == 1:
            return predictions[0]
        positive, negative = predictions
        return negative + self.state.guidance_scale * (positive - negative)

    def project(self, tokens, branch):
        model = self.model
        temb = self.conditioning[branch][0]
        shift, scale = (
            model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
        ).chunk(2, dim=1)
        tokens = (model.norm_out(tokens.float()) * (1 + scale) + shift).type_as(tokens)
        output = model.proj_out(tokens)
        batch, _, frames, height, width = self.state.latents.shape
        pt, ph, pw = model.config.patch_size
        output = output.reshape(
            batch, frames // pt, height // ph, width // pw, pt, ph, pw, -1
        )
        return (
            output.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
            .contiguous()
        )


class _Feedback:
    def __init__(self, mesh, depth):
        self.mesh, self.depth = mesh, depth
        self.pending = deque()

    def send(self, tensor):
        # Waiting for the oldest send is safe after P candidates: the head
        # has already posted its corresponding receive before microbatch P.
        if len(self.pending) >= self.depth:
            work, buffer = self.pending.popleft()
            work.wait()
            del buffer
        buffer = tensor.contiguous()
        work = dist.isend(
            buffer, dst=self.mesh.pipeline.ranks[0], group=self.mesh.feedback_group
        )
        self.pending.append((work, buffer))

    def receive(self, template):
        output = torch.empty_like(template)
        dist.irecv(
            output, src=self.mesh.pipeline.ranks[-1], group=self.mesh.feedback_group
        ).wait()
        return output

    def drain(self):
        while self.pending:
            work, buffer = self.pending.popleft()
            work.wait()
            del buffer


@torch.inference_mode()
def run_wan_fpp(pipeline, state, *, step_callback=None):
    """Run a complete static request; virtual degrees provide a serial reference.

    Callbacks observe committed latents on last-stage ranks only. Dynamic EPE
    migration and per-step serving are intentionally outside this contract.
    """
    config = pipeline._epe_fpp_config
    if config.schedule != "stream":
        raise ValueError("run_wan_fpp requires stream scheduling")
    if state.step_index != 0:
        raise ValueError("cross-step FPP requires a fresh request")
    mesh = pipeline._epe_fpp_mesh
    topology = mesh.pipeline
    virtual = (
        config.reference_stages is not None or config.reference_context_degree != 1
    )
    if virtual and (topology.degree * mesh.cp_degree * mesh.cfg_degree != 1):
        raise ValueError("virtual FPP reference degrees require a single process")
    stages = config.reference_stages or topology.degree
    if config.patches != 1 and config.patches < stages:
        raise ValueError("cross-step FPP requires patches >= pipeline stages")
    validate_fpp_scheduler(state.scheduler)
    layout = FppTokenLayout(
        state.image_tokens,
        config.patches,
        config.reference_context_degree if virtual else mesh.cp_degree,
    )
    state.fpp_state = FppState(config)
    fpp = state.fpp_state
    kernel = _WanStage(pipeline.transformer, state, mesh, layout, reference=virtual)
    feedback = _Feedback(mesh, stages)
    total = len(state.timesteps)

    def synchronize():
        # NCCL Work.wait() orders CUDA streams; it does not wait on the host.
        # Quiesce every PP/CP/CFG communicator before entering a world
        # collective, otherwise a fast head can launch the broadcast while
        # a slower tail is still trying to launch its final collectives.
        if state.latents.is_cuda:
            torch.cuda.synchronize(state.latents.device)
        if mesh.boundary_group is not None:
            dist.barrier(group=mesh.boundary_group)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(state.latents, src=mesh.output_rank)
        if state.latents.is_cuda:
            torch.cuda.synchronize(state.latents.device)

    def committed(step, full):
        fpp.last_step = step
        fpp.full_steps += int(full)
        fpp.patch_steps += int(not full)
        state.step_index = step + 1
        if step_callback is not None and topology.last:
            step_callback(state)

    try:
        while state.step_index < total:
            step = state.step_index
            if config.full_step(step, total):
                kernel.prepare_step(step)
                prediction = kernel.forward(state.latents, None)
                kernel.transport.drain()
                if topology.last:
                    state.latents = fpp_scheduler_step(
                        state.scheduler,
                        prediction,
                        state.timesteps[step],
                        state.latents,
                        commit=True,
                    )
                synchronize()
                committed(step, True)
                continue
            end = step + 1
            while end < total and not config.full_step(end, total):
                end += 1
            head_latents = state.latents
            local_feedback = deque()
            micro = 0
            for current in range(step, end):
                kernel.prepare_step(current)
                # All candidate previews for this logical step share this base.
                base = state.latents
                for index, patch in enumerate(config.order(current, stages)):
                    if topology.first and micro >= stages:
                        head_latents = (
                            local_feedback.popleft()
                            if topology.last
                            else feedback.receive(head_latents)
                        )
                    prediction = kernel.forward(head_latents, patch)
                    if topology.last:
                        final_patch = index == config.patches - 1
                        candidate = fpp_scheduler_step(
                            state.scheduler,
                            prediction,
                            state.timesteps[current],
                            base,
                            commit=final_patch,
                        )
                        if final_patch:
                            state.latents = candidate
                        if topology.first:
                            local_feedback.append(candidate)
                        else:
                            feedback.send(candidate)
                    micro += 1
                committed(current, False)
            if topology.first and not topology.last:
                for _ in range(stages):
                    head_latents = feedback.receive(head_latents)
                state.latents = head_latents
            kernel.transport.drain()
            feedback.drain()
            synchronize()
        pipeline.last_fpp_stats = {
            "schedule": "stream",
            "pipeline_degree": topology.degree,
            "context_degree": mesh.cp_degree,
            "cfg_degree": mesh.cfg_degree,
            "reference_stages": config.reference_stages,
            "reference_context_degree": config.reference_context_degree,
            "patches": config.patches,
            "padded_tokens": layout.padded_tokens,
            "full_steps": fpp.full_steps,
            "patch_steps": fpp.patch_steps,
        }
        return state
    finally:
        fpp.cache.clear()
        kernel.features.clear()
