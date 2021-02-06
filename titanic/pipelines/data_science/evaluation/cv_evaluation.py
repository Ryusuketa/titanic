from typing import Callable
from omegaconf.dictconfig import DictConfig
import torch
import pandas as pd

from titanic.pipelines.data_science.model.trainer import Trainer
from titanic.pipelines.data_science.utils import init_model
from titanic.pipelines.data_handlers.data_handler import CVDataLoaderPair, CVPair, SupervisedLearningDataHandler
from typing import Any, Dict


class CVEvaluator:
    def __init__(
        self,
        model: Any,
        data_handler: SupervisedLearningDataHandler,
        evaluation_functions: Dict[str, Callable],
        trainer: Trainer,
        wandb
    ) -> None:
        self._model = model
        self._data_handler = data_handler
        self._evaluation_functions = evaluation_functions
        self._trainer = trainer
        self._wandb = wandb

    def _run_one_cv_evaluation(
        self, data_loader: CVDataLoaderPair, dataset: CVPair, cfg: DictConfig
    ) -> float:
        model = init_model(self._model, cfg)
        trainer_instance = self._trainer(model, data_loader=data_loader, dataset=dataset, cfg=cfg)
        trainer_instance.train()
        features = dataset.validation[:][0]
        outputs = dataset.validation[:][1]
        model = trainer_instance.get_model()
        results = self._apply_evaluation_functions(model, features, outputs)
        return results

    def _apply_evaluation_functions(
        self, model: Any, features: torch.Tensor, outputs: torch.Tensor
    ) -> Dict[str, float]:
        return pd.Series({k: f(model, features, outputs) for k, f in self._evaluation_functions.items()})

    def _send_wandb(self, values: pd.Series) -> None:
        self._wandb.log(values.to_dict())

    def run_evaluation(self, cfg: DictConfig):
        evaluated = 0
        cnt = 0
        for loader, dataset in zip(self._data_handler.get_data_loader_of_cv_pairs(),
                                   self._data_handler.get_cv_subsets()):
            evaluated += self._run_one_cv_evaluation(loader, dataset, cfg)
            cnt += 1
        assert ~pd.isnull(evaluated).any(), f'evaluated results includes nan value, actual {evaluated}'
        self._send_wandb(evaluated / cnt)
