from types import MethodType

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
)

from concept_erasure import ConceptScrubber, ErasureMethod, LeaceFitter
from concept_erasure.utils import assert_type


def _fast_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: tuple[torch.Tensor, ...] | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
    """Fast version of LLaMA attention."""
    assert not output_attentions, "output_attentions not implemented for fast_attn"
    del attention_mask  # unused, but it's non-None in the base class

    bsz, q_len, _ = hidden_states.size()
    q = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    k = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    v = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = k.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        k = torch.cat([past_key_value[0], k], dim=2)
        v = torch.cat([past_key_value[1], v], dim=2)

    past_key_value = (k, v) if use_cache else None
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def patch_attention_llama_(model: torch.nn.Module):
    """Patch the attention layers of a LLaMA model to use fast attention kernels."""
    for mod in model.modules():
        if isinstance(mod, LlamaAttention):
            mod.forward = MethodType(_fast_attn, mod)


@torch.no_grad()
def scrub_llama(
    model: LlamaForCausalLM,
    train: Dataset,
    z_column: str | None,
    batch_size: int = 1,
    method: ErasureMethod = "leace",
    sublayers: bool = True,
    affine: bool = True,
) -> tuple[ConceptScrubber | None, float]:
    base = assert_type(LlamaModel, model.base_model)
    d = assert_type(int, base.config.hidden_size)

    if z_column is None:
        k = -1
        scrubber = None
    else:
        k = assert_type(int, train.features[z_column].feature.num_classes)
        scrubber = ConceptScrubber()

    losses = []

    xs = []
    zs = []
    N = len(train) // batch_size
    train = train.with_format("torch", device=model.device)

    # Embed all the inputs
    embed_fn = base.get_input_embeddings()
    for i, batch in enumerate(tqdm(train.iter(batch_size), desc="Embedding", total=N)):
        assert isinstance(batch, dict)

        tokens = assert_type(torch.Tensor, batch["input_ids"])
        x = embed_fn(tokens)
        xs.append(x.to("cpu", non_blocking=True))

        # We don't actually need to move these to the CPU since they're small
        if z_column is not None:
            zs.append(F.one_hot(batch[z_column], num_classes=k))

    # Enumerate the layers
    for j, layer in enumerate(tqdm(base.layers, unit="layer")):
        assert isinstance(layer, LlamaDecoderLayer)

        attn_eraser = None
        if scrubber is not None:
            attn_fitter = LeaceFitter(
                d, k, affine=affine, device=model.device, method=method
            )

            # Fit the next eraser on the previous hidden states
            for x, z in tqdm(zip(xs, zs), desc="Fitting (attn)", total=N):
                assert scrubber is not None

                # Discard post-LN output and recompute during application to save RAM
                x = layer.input_layernorm(x.to(model.device))
                attn_fitter.update(x, z)

            attn_eraser = attn_fitter.eraser
            scrubber.erasers[f"layers-{j}-input_layernorm"] = attn_eraser
            del attn_fitter  # Save VRAM

        # Run attention & MLP with the erasers we just fit
        for i, x in tqdm(enumerate(xs), desc="Applying (attn)", total=N):
            # Bring back to the accelerator
            x = x.to(model.device)
            h = layer.input_layernorm(x)  # Recomputing from above

            # Apply the eraser
            if attn_eraser is not None and scrubber is not None:
                h = attn_eraser(h).type_as(h)

            pos_ids = torch.arange(0, h.shape[-2], device=h.device, dtype=torch.long)
            pos_ids = pos_ids.unsqueeze(0).view(-1, h.shape[-2])

            h, _, __ = layer.self_attn(h, position_ids=pos_ids)
            h = x = x + h  # Post-attention residual connection

            # We're not scrubbing the sublayers, so run the rest of the layer
            if not sublayers:
                h = layer.post_attention_layernorm(h)
                h = layer.mlp(h)
                h = x + h  # Post-MLP residual connection

            xs[i] = h.to("cpu", non_blocking=True)

        # Skip the part where we scrub the MLP if we're not doing that
        if not sublayers:
            continue

        mlp_eraser = None
        if scrubber is not None:
            mlp_fitter = LeaceFitter(
                d, k, affine=affine, device=model.device, method=method
            )

            # Fit the next eraser on the previous hidden states
            for x, z in tqdm(zip(xs, zs), desc="Fitting (MLP)", total=N):
                assert scrubber is not None

                # Discard post-LN output and recompute during application to save RAM
                x = layer.post_attention_layernorm(x.to(model.device))
                mlp_fitter.update(x, z)

            mlp_eraser = mlp_fitter.eraser
            scrubber.erasers[f"layers-{j}-post_attention_layernorm"] = mlp_eraser
            del mlp_fitter  # Save VRAM

        for i, x in tqdm(enumerate(xs), desc="Applying (MLP)", total=N):
            # Bring back to the accelerator
            x = x.to(model.device)
            h = layer.post_attention_layernorm(x)  # Recomputing from above

            # Apply the eraser
            if mlp_eraser is not None and scrubber is not None:
                h = mlp_eraser.eraser(h).type_as(h)

            h = layer.mlp(h)
            h = x + h  # Post-MLP residual connection
            xs[i] = h.to("cpu", non_blocking=True)

    for batch, x in tqdm(zip(train.iter(batch_size), xs), total=N):
        assert isinstance(batch, dict)
        tokens = assert_type(torch.Tensor, batch["input_ids"])

        x = x.to(model.device)
        x = base.norm(x)
        logits = model.lm_head(x)

        labels = tokens.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )
        losses.append(lm_loss)

    return scrubber, torch.stack(losses).mean().item()
