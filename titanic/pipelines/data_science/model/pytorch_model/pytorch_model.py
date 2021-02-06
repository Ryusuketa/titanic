from typing import Dict, Any
from torchvision.models import resnet101
import torch
from torch import nn


def _generate_fcn(resnet_input_feature_size: int, fc_parameters: Dict[str, Any]) -> nn.Module:
    pass


class ResNet101:
    def __init__(self, fcn_parameters: Dict[str, Any]) -> None:
        self._model = resnet101(pretrained=True)
        self._model.fc = _generate_fcn(self.__get_resnet_fcn_input_size(), fcn_parameters)

    def __get_resnet_fcn_input_size(self):
        return self._model.fc.weight.shape[0]

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def predict(self, input: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(self._model(input), dim=1)

    def predict_proba(self, input: torch.Tensor) -> torch.Tensor:
        return self._model(input)
