import math

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from concept_erasure.scrubbing import patch_attention_, scrub


@pytest.mark.slow
@pytest.mark.parametrize("model_str", ["EleutherAI/pythia-160m", "huggyllama/llama-7b"])
@torch.inference_mode()
def test_scrubbing(model_str: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # It turns out to be important to use the actual pretrained weights and not random
    # weights, since random models tend to get near-constant loss even if you implement
    # them incorrectly.
    model = AutoModelForCausalLM.from_pretrained(model_str).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    ## Test attention patching by itself ##
    # Get the output pre-patching
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    clean_logits = model(input_ids).logits

    # Patch the model
    patch_attention_(model)

    # Check that the outputs are the same
    patched_logits = model(input_ids).logits
    torch.testing.assert_close(clean_logits, patched_logits, atol=5e-5, rtol=5e-5)

    ## Test attention patching + concept erasure ##
    tokenized = tokenizer(
        ["The quick brown fox jumps over the lazy dog", "Hello world!"]
    )
    ds = Dataset.from_dict(dict(tokenized)).with_format("torch")
    gt_losses = []

    for record in ds:
        input_ids = record["input_ids"].unsqueeze(0).to(device)
        gt_losses.append(model(input_ids, labels=input_ids).loss)

    gt_loss = torch.stack(gt_losses).mean().item()
    _, loss = scrub(model, ds, z_column=None)
    assert math.isclose(gt_loss, loss, rel_tol=1e-4)
