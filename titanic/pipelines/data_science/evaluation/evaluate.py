from omegaconf.dictconfig import DictConfig
from kedro.config import ConfigLoader
from typing import Dict, Any, List

from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler
from .cv_evaluation import CVEvaluator
from ..evaluation_module import get_evaluation_function
from ..model.model_mapper import get_model_class
from ..model.trainer import get_trainer_class
import wandb

MODEL_PARAMETERS_PATH = 'search_results/'
PROJECT_ENV_NAME = 'KAGGLE_COMPETITION_NAME'


def get_model_params() -> Dict[str, Any]:
    conf = ConfigLoader(MODEL_PARAMETERS_PATH).get('*.yaml')
    return conf['best_evaluated_params']


def set_wandb_config() -> None:
    conf = get_model_params()
    wandb.config.update(conf)


def run_evaluation(
    model_name: str, trainer_name: str, data_handler: SupervisedLearningDataHandler, evaluation_task_names: List[str],
    cfg: DictConfig, wandb_project_name: str, **kwargs
) -> None:
    """[summary]

    Parameters
    ----------
    model_name : str
        Supervised model.
    trainer_name : str
        Trainer name.
    data_handler : SupervisedLearningDataHandler
        data handler object which has evaluation data.
    evaluation_task_names : List[str]
        The string names of you want to calculate evaluation metric
    """
    wandb.init(project=wandb_project_name)
    set_wandb_config()
    evaluation_functions = {name: get_evaluation_function(name) for name in evaluation_task_names}
    model = get_model_class(model_name)
    trainer = get_trainer_class(trainer_name)
    evaluator = CVEvaluator(model, data_handler, evaluation_functions, trainer, wandb)
    evaluator.run_evaluation(cfg)
    return 'done'
