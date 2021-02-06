from typing import Callable
from omegaconf.dictconfig import DictConfig

from titanic.pipelines.data_science.model.trainer import Trainer
from titanic.pipelines.data_handlers.data_handler import CVDataLoaderPair, CVPair, SupervisedLearningDataHandler
from typing import Any
from titanic.pipelines.data_science.utils import init_model


class HydraCVEvaluator:
    def __init__(
        self,
        model: Any,
        data_handler: SupervisedLearningDataHandler,
        evaluation_function: Callable,
        trainer: Trainer
    ) -> None:
        self._model = model
        self._data_handler = data_handler
        self._evaluation_function = evaluation_function
        self._trainer = trainer

    def _run_one_cv_evaluation(
        self, data_loader: CVDataLoaderPair, dataset: CVPair, cfg: DictConfig
    ) -> float:
        model = init_model(self._model, cfg)
        trainer_instance = self._trainer(model, data_loader=data_loader, dataset=dataset, cfg=cfg)
        trainer_instance.train()
        features = dataset.validation[:][0]
        outputs = dataset.validation[:][1]
        model = trainer_instance.get_model()
        results = self._evaluation_function(model, features, outputs)
        return results

    def run_evaluation(self, cfg: DictConfig):
        evaluated = 0
        cnt = 0
        for loader, dataset in zip(self._data_handler.get_data_loader_of_cv_pairs(),
                                   self._data_handler.get_cv_subsets()):
            evaluated += self._run_one_cv_evaluation(loader, dataset, cfg)
            cnt += 1
        return evaluated / cnt
