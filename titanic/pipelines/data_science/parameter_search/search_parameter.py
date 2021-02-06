from typing import Any, Callable
import logging

import hydra
from omegaconf import DictConfig
import sys

from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler
from kedro.config import ConfigLoader
from .hydra_cv_evaluation import HydraCVEvaluator
from ..model.trainer import get_trainer_class
from ..model.model_mapper import get_model_class
from ..evaluation_module import get_evaluation_function

log = logging.getLogger(__name__)
MODEL_PARAMETERS_PATH = 'search_results/'


def generate_task_function(cv_evaluater: Any) -> Callable[[DictConfig], Any]:
    @hydra.main(config_name="config.yml")
    def hydra_main(cfg: DictConfig) -> Any:
        result = cv_evaluater.run_evaluation(cfg)
        log.info(f"params({cfg})={result}")
        return result
    return hydra_main


def _get_model_params() -> DictConfig:
    conf = ConfigLoader(MODEL_PARAMETERS_PATH).get('*.yaml')
    return conf['best_evaluated_params']


def run_parameter_search(model_name: str,
                         trainer_name: str,
                         data_handler: SupervisedLearningDataHandler,
                         evaluation_function_name: str,
                         *args,
                         **kwargs) -> Any:
    model = get_model_class(model_name)
    trainer = get_trainer_class(trainer_name)
    evaluation_function = get_evaluation_function(evaluation_function_name)
    evaluator = HydraCVEvaluator(model, data_handler, evaluation_function, trainer)
    sys.argv = sys.argv[0:1] + ['-m']
    run_task = generate_task_function(evaluator)
    run_task()
    optim_param = _get_model_params()
    return optim_param
