from itertools import chain
from typing import TypeVar

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from concept_erasure.utils import assert_type

tokenizer = assert_type(
    PreTrainedTokenizerBase, AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
)

T = TypeVar("T")


def chunk(seq: list[T], chunk_size: int) -> list[list[T]]:
    """Chunk a sequence into chunks of size `chunk_size`."""

    return [
        seq[i * chunk_size : (i + 1) * chunk_size]
        for i in range(len(seq) // chunk_size)
    ]


def uniform_chunks(batch: dict[str, list]) -> dict[str, list]:
    eos_pos = ds.features["pos"].feature.str2int("X")

    pos_iter = chain.from_iterable(ids + [eos_pos] for ids in batch["pos"])
    token_iter = chain.from_iterable(
        ids + [tokenizer.eos_token_id] for ids in batch["input_ids"]
    )
    return {
        "input_ids": chunk(list(token_iter), 2048),
        "pos": chunk(list(pos_iter), 2048),
    }


ds = Dataset.load_from_disk("/mnt/ssd-2/nora/pile-hf/")
chunked_ds = ds.map(
    uniform_chunks,
    batched=True,
    batch_size=1000,
    num_proc=24,
    remove_columns=["attention_mask"],
).map(
    lambda x: {"num_bytes": len(tokenizer.decode(x["input_ids"]).encode("utf-8"))},
    num_proc=24,
)
chunked_ds.save_to_disk("/mnt/ssd-2/nora/pile-hf-chunked/")
