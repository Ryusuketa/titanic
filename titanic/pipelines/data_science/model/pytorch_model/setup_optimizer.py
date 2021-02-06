from typing import Any
from torch.optim import Adam
import torch.nn as nn


def setup_optimizer(model: nn.Module, objective_function_name: str, **kwargs) -> Any:
    def get_adam():
        yield Adam(model.params, lr=kwargs['lr'], betas=kwargs['betas'], weight_decay=kwargs['weight_decay'])

    function_map = dict(adam=get_adam())
    return function_map.get(objective_function_name)
