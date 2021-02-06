from lightgbm import LGBMClassifier
from typing import Union
import numpy as np
import torch


class WrappedLGBMClassifier(LGBMClassifier):
    def predict(self, X: Union[np.array, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return torch.Tensor(super().predict(X, *args, **kwargs))

    def predict_proba(self, X: np.array, *args, **kwargs) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return torch.Tensor(super().predict_proba(X, *args, **kwargs))
