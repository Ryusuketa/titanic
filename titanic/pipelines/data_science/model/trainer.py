from typing import Any, Callable, Optional, Union
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from titanic.pipelines.data_science.model.pytorch_model.setup_optimizer import setup_optimizer
from titanic.pipelines.data_science.model.pytorch_model.setup_objective_function import setup_objective_function
from titanic.pipelines.data_handlers.data_handler import CVDataLoaderPair, CVPair
from abc import ABC, abstractmethod
import torch.optim


class Trainer(ABC):
    @abstractmethod
    def train(self, wandb_hooks: Any = None) -> None:
        pass

    def get_model(self) -> Any:
        return self._model


class PytorchTrainerBase(Trainer):
    def __init__(
        self, model: Any, data_loader: Union[CVDataLoaderPair, DataLoader], cfg: DictConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        objective_function: Optional[Callable[..., torch.Tensor]] = None, **kwargs
    ) -> None:
        self._model = model
        self._optimizer = optimizer or setup_optimizer(model, **cfg)
        self._objective_function = objective_function or setup_objective_function(**cfg)
        self._data_loader = data_loader
        self._cfg = cfg

    @abstractmethod
    def train(self, wandb_hooks: Optional[Any] = None) -> None:
        pass


class SklearnTrainer(Trainer):
    def __init__(
        self, model: Any, dataset: Union[CVPair, Dataset], cfg: DictConfig, **kwargs
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._cfg = cfg

    def train(self, wandb_hooks: Optional[Any] = None) -> None:
        if isinstance(self._dataset, CVPair):
            train_features, train_outputs = self._get_numpy_features_and_outputs_for_train()
        else:
            train_features, train_outputs = self._get_numpy_features_and_outputs()
        self._model.fit(train_features, train_outputs)

    def _get_numpy_features_and_outputs_for_train(self):
        features = self._dataset.train[:][0]
        outputs = self._dataset.train[:][1]
        return features.numpy(), outputs.numpy()

    def _get_numpy_features_and_outputs_for_validation(self):
        features = self._dataset.validation[:][0]
        outputs = self._dataset.validation[:][1]
        return features.numpy(), outputs.numpy()

    def _get_numpy_features_and_outputs(self):
        features = self._dataset[:][0]
        outputs = self._dataset[:][1]
        return features, outputs


class LightGBMTrainer(Trainer):
    def __init__(
        self, model: Any, dataset: CVPair, cfg: DictConfig, **kwargs
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._cfg = cfg


def get_trainer_class(name: str) -> Trainer:
    class_map = dict(sklearn=SklearnTrainer)
    return class_map.get(name)
