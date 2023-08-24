import math
import os
from argparse import ArgumentParser

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from concept_erasure.scrubbing import patch_attention_, random_scrub, scrub
from concept_erasure.utils import assert_type


@torch.inference_mode()
def evaluate(ds, model, args, nats_to_bpb):
    losses = []
    pbar = tqdm(
        ds.iter(args.batch_size),
        desc="Evaluating",
        total=len(ds) // args.batch_size,
    )

    for batch in pbar:
        assert isinstance(batch, dict)

        tokens = assert_type(torch.Tensor, batch["input_ids"])
        loss = model(tokens, labels=tokens).loss
        losses.append(loss)

        pbar.set_postfix(
            loss=torch.stack(losses).mean().item() * nats_to_bpb,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--proj-type", type=str, default="leace")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--skip-sublayers", action="store_true")
    parser.add_argument("--z-column", type=str, default="pos")
    args = parser.parse_args()

    # Step 1: Load the model
    print(f"Using device {args.device}")
    model = assert_type(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            load_in_8bit=args.int8,
            torch_dtype=torch.float16,
        ),
    )
    tokenizer = assert_type(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(args.model, use_fast="llama" not in args.model),
    )
    patch_attention_(model)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)

    # Step 2: Load the dataset
    ds = Dataset.load_from_disk(args.dataset).with_format(
        "torch",
        device=model.device,
    )
    splits = ds.train_test_split(train_size=2**15, seed=42)
    train = splits["train"].select(range(2**14))
    test = splits["test"].select(range(2**14))

    byte_lengths = assert_type(torch.Tensor, ds["num_bytes"])
    length_ratio = float(2048 / byte_lengths.float().mean())
    nats_to_bpb = length_ratio / math.log(2)

    if args.random:
        k = assert_type(int, train.features[args.z_column].feature.num_classes)

        with torch.inference_mode():
            losses = []
            pbar = tqdm(
                ds.iter(args.batch_size),
                desc="Evaluating",
                total=len(ds) // args.batch_size,
            )
            base = assert_type(PreTrainedModel, model.base_model)

            for batch in pbar:
                assert isinstance(batch, dict)

                tokens = assert_type(torch.Tensor, batch["input_ids"])
                with random_scrub(base, subspace_dim=k):
                    loss = model(tokens, labels=tokens).loss
                    losses.append(loss)

                pbar.set_postfix(
                    loss=torch.stack(losses).mean().item() * nats_to_bpb,
                )

    scrubber, dirty_loss = scrub(
        model,
        train,
        affine=not args.no_bias,
        batch_size=args.batch_size,
        method=args.method,
        sublayers=not args.skip_sublayers,
        z_column=args.z_column,
    )
    print(f"Train loss (bits per byte): {dirty_loss * nats_to_bpb:.2f}")

    if scrubber:
        with scrubber.scrub(model):
            evaluate(test, model, args, nats_to_bpb)
