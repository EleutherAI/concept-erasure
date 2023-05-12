# concept-erasure
Erasing concepts from neural representations with provable guarantees

Example usage:
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
