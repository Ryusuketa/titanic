from typing import Any

from .sklearn_model.models import WrappedSVC
from .lightgbm_model.models import WrappedLGBMClassifier


def get_model_class(name: str) -> Any:
    class_map = dict(svc=WrappedSVC, lightgbm=WrappedLGBMClassifier)
    return class_map.get(name)
