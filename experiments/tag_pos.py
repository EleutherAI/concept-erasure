from pathlib import Path

import spacy
import torch
from datasets import Dataset
from torch.multiprocessing import spawn
from tqdm.auto import tqdm

from concept_erasure.utils import assert_type


def worker(rank: int, out_dir: Path):
    world_size = torch.cuda.device_count()

    dataset = (
        assert_type(Dataset, Dataset.from_json("/mnt/ssd-1/igor/data/pile/val.jsonl"))
        .filter(lambda x: len(x["text"]) < 500_000)
        .shuffle(seed=42)
        .shard(world_size, rank)
    )

    # Make our own directory for the output
    out_dir = out_dir / f"rank_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spacy.require_gpu(gpu_id=rank)
    nlp = spacy.load(
        "en_core_web_trf",
        # Important for performance; otherwise Tensor Cores aren't used
        config=dict(
            components=dict(transformer=dict(model=dict(mixed_precision=True)))
        ),
    )

    doc_iter = nlp.pipe(
        map(lambda x: x["text"], dataset),
        batch_size=16,
    )

    for i, doc in tqdm(
        enumerate(doc_iter), position=rank, smoothing=0.0, total=len(dataset)
    ):
        # This is a little sketchy and depends how Dataset.shard works
        doc_id = rank + i * world_size
        doc.to_disk(out_dir / f"doc_{doc_id}.spacy", exclude=["tensor", "user_data"])


if __name__ == "__main__":
    out_dir = Path("/mnt/ssd-2/nora/pile-spacy")
    out_dir.mkdir(parents=True, exist_ok=True)

    spawn(
        worker,
        args=(out_dir,),
        nprocs=torch.cuda.device_count(),
        join=True,
    )
