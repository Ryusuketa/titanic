import torch
import numpy as np
from sklearn.svm import SVC
from typing import Union


class WrappedSVC(SVC):
    def predict(self, X: Union[np.array, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return torch.Tensor(super().predict(X))

    def predict_proba(self, X: np.array) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return torch.Tensor(super().predict_proba(X))
