from typing import Any

from omegaconf.dictconfig import DictConfig
from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler
from titanic.pipelines.data_science.utils import init_model
from .model_mapper import get_model_class
from .trainer import get_trainer_class


def train(model_name: str, trainer_name: str, data_handler: SupervisedLearningDataHandler, cfg: DictConfig,
          *args, **kwargs) -> Any:
    model = init_model(get_model_class(model_name), cfg)
    data_loader = data_handler.get_data_loader()
    dataset = data_handler.get_dataset()
    trainer = get_trainer_class(trainer_name)(model, data_loader=data_loader, dataset=dataset, cfg=cfg)
    trainer.train()
    return trainer.get_model()
