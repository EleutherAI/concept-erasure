from typing import Literal

from datasets import Dataset
from transformers import GPTNeoXForCausalLM, LlamaForCausalLM, PreTrainedModel

from ..concept_scrubber import ConceptScrubber
from .llama import patch_attention_llama_, scrub_llama
from .neox import patch_attention_neox_, scrub_neox


def patch_attention_(model: PreTrainedModel):
    """Patch the attention layers of a model to use fast attention kernels."""
    ty = model.config.model_type

    if ty == "gpt_neox":
        patch_attention_neox_(model)
    elif ty == "llama":
        patch_attention_llama_(model)
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")


def scrub(
    model: PreTrainedModel,
    train: Dataset,
    z_column: str | None,
    affine: bool = True,
    batch_size: int = 1,
    proj_type: Literal["leace", "loracs", "orth"] = "leace",
    sublayers: bool = True,
) -> tuple[ConceptScrubber | None, float]:
    """Scrub a model to remove the fast attention kernels."""
    if isinstance(model, GPTNeoXForCausalLM):
        return scrub_neox(model, train, z_column, batch_size, proj_type, affine)
    elif isinstance(model, LlamaForCausalLM):
        return scrub_llama(
            model, train, z_column, batch_size, proj_type, sublayers, affine
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model).__name__}")
