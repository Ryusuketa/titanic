from omegaconf import DictConfig
from typing import Any
import inspect


def init_model(model_class: Any, cfg: DictConfig) -> Any:
    model_param_keys = inspect.signature(model_class).parameters.keys()
    model_param = {k: v for k, v in cfg.items() if k in model_param_keys}
    model = model_class(**model_param)
    return model
