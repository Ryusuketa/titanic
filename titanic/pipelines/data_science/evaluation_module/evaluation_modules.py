from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler
from typing import Any, Callable
import torch


def calculate_precision_score(model: Any, features: torch.Tensor, outputs: torch.Tensor) -> float:
    predicted: torch.Tensor = model.predict(features)
    precision = precision_score(predicted.numpy(), outputs.numpy())
    return float(precision)


def calculate_recall_score(model: Any, features: torch.Tensor, outputs: torch.Tensor) -> float:
    predicted: torch.Tensor = model.predict(features)
    recall = recall_score(predicted.numpy(), outputs.numpy())
    return float(recall)


def calculate_f1_score(model: Any, features: torch.Tensor, outputs: torch.Tensor) -> float:
    predicted: torch.Tensor = model.predict(features)
    f1 = f1_score(predicted.numpy(), outputs.numpy())
    return float(f1)


def calculate_accuracy_score(model: Any, features: torch.Tensor, outputs: torch.Tensor) -> float:
    predicted: torch.Tensor = model.predict(features)
    accuracy = accuracy_score(predicted.numpy(), outputs.numpy())
    return float(accuracy)


def get_evaluation_function(name: str) -> Callable[[Any, SupervisedLearningDataHandler], float]:
    function_map = dict(
        precision=calculate_precision_score,
        recall=calculate_recall_score,
        f1=calculate_f1_score,
        accuracy=calculate_accuracy_score,
    )
    assert name in function_map.keys(), f'{name} is not in map keys {list(function_map.keys())}.'
    return function_map[name]
