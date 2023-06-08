# Least-Squares Concept Erasure (LEACE)
Concept erasure aims to remove specified features from a representation. It can be used to improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). This is the repo for **LEAst-squares Concept Erasure (LEACE)**, a closed-form method which provably prevents all linear classifiers from detecting a concept while inflicting the least possible damage to the representation. You can check out the paper [here](https://arxiv.org/abs/2306.03819).

# Installation

```bash
pip install concept-erasure
```

# Usage

`ConceptEraser` is the central class in this repo. It keeps track of the covariance and cross-covariance statistics needed to erase a concept, and lazily computes the LEACE parameters when needed.

## Batch usage

In most cases, you probably have a batch of feature vectors `X` and concept labels `Z` and want to erase the concept from `X`. The easiest way to do this is using `ConceptEraser.fit()` followed by `ConceptEraser.forward()`:

```python
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from concept_erasure import ConceptEraser

n, d, k = 2048, 128, 2

X, Y = make_classification(
    n_samples=n,
    n_features=d,
    n_classes=k,
    random_state=42,
)
X_t = torch.from_numpy(X)
Y_t = torch.from_numpy(Y)

# Logistic regression does learn something before concept erasure
real_lr = LogisticRegression(max_iter=1000).fit(X, Y)
beta = torch.from_numpy(real_lr.coef_)
assert beta.norm(p=torch.inf) > 0.1

eraser = ConceptEraser.fit(X_t, Y_t)
X_ = eraser(X_t)

# But learns nothing after
null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(X_.numpy(), Y)
beta = torch.from_numpy(null_lr.coef_)
assert beta.norm(p=torch.inf) < 1e-4
```

## Streaming usage
If you have a **stream** of data, you can use `ConceptEraser.update()` to update the statistics and `ConceptEraser.forward()` to erase the concept. This is useful if you have a large dataset and want to avoid storing it all in memory.

```python
from concept_erasure import ConceptEraser
from sklearn.datasets import make_classification
import torch

n, d, k = 2048, 128, 2

X, Y = make_classification(
    n_samples=n,
    n_features=d,
    n_classes=k,
    random_state=42,
)
X_t = torch.from_numpy(X)
Y_t = torch.from_numpy(Y)

eraser = ConceptEraser(d, 1, dtype=X_t.dtype)

# Compute cross-covariance matrix using batched updates
for x, y in zip(X_t.chunk(2), Y_t.chunk(2)):
    eraser.update(x, y)

# Erase the concept from the data
x_ = eraser(X_t[0])
```

# Paper replication

Scripts used to generate the part-of-speech tags for the concept scrubbing experiments can be found in [this repo](https://github.com/EleutherAI/tagged-pile). We plan to upload the tagged datasets to the HuggingFace Hub shortly.

## Concept scrubbing

The concept scrubbing code is a bit messy right now, and will probably be refactored soon. We found it necessary to write bespoke implementations for different HuggingFace model families. So far we've implemented LLaMA and GPT-NeoX. These can be found in the `concept_erasure.scrubbing` submodule.
