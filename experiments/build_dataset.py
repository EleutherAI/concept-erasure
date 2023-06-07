import os
from multiprocessing import Pool
from pathlib import Path

import spacy
from datasets import (
    ClassLabel,
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
)
from spacy.tokens import Doc
from transformers import AutoTokenizer

from concept_erasure.utils import assert_type

TOKENIZER_NAME = "huggyllama/llama-7b"


def spacy_to_hf(spacy_dir: Path):
    # Make sure that the tokenizer doesn't use multiple threads
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        model_max_length=None,
        use_fast=True,
    )
    vocab = spacy.load("en_core_web_trf").vocab

    # Load the spacy docs
    for path in spacy_dir.rglob("*.spacy"):
        doc = Doc(vocab).from_disk(path)

        encoding = tokenizer(
            doc.text,
            max_length=None,
            return_offsets_mapping=True,
        )
        encoding.pop("token_type_ids", None)

        # In general, BPE tokens will be subspans of SpaCy tokens. Each BPE token
        # should get its label from the SpaCy token that contains it.
        spacy_spans = [
            doc.char_span(*span, alignment_mode="expand")
            for span in encoding.pop("offset_mapping")
        ]
        # In rare cases, the BPE token may be split across multiple SpaCy tokens. In
        # those cases, we assign the token to the first SpaCy token that contains it.
        encoding["pos"] = [
            # Whitespace might get tokenized into a BPE token, but it doesn't have
            # a part of speech. Assign it the "X" (other) tag.
            span[0].pos_ if span and span[0].pos_ != "SPACE" else "X"
            for span in spacy_spans
        ]
        yield encoding


def process_shard(rank_path: Path) -> Dataset:
    ds = Dataset.from_generator(
        spacy_to_hf,
        features=Features(
            {
                "input_ids": Sequence(Value(dtype="int32")),
                "attention_mask": Sequence(Value(dtype="int8")),
                "pos": Sequence(
                    # Universal Dependency POS tags
                    ClassLabel(
                        names=[
                            "ADJ",
                            "ADP",
                            "ADV",
                            "AUX",
                            "CCONJ",
                            "DET",
                            "INTJ",
                            "NOUN",
                            "NUM",
                            "PART",
                            "PRON",
                            "PROPN",
                            "PUNCT",
                            "SCONJ",
                            "SYM",
                            "VERB",
                            "X",
                        ]
                    ),
                ),
            }
        ),
        gen_kwargs=dict(spacy_dir=rank_path),
    )
    return assert_type(Dataset, ds)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("spacy_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    folders = list(args.spacy_dir.glob("rank_*"))

    with Pool(len(folders)) as pool:
        shards = pool.map(process_shard, folders)
        master = concatenate_datasets(shards)

    # Save the dataset
    master.save_to_disk(args.output_dir)
