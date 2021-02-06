from titanic.pipelines.data_science.parameter_search.hydra_cv_evaluation import HydraCVEvaluator
from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler
from titanic.pipelines.data_science.model.trainer import Trainer
from torch.utils.data import TensorDataset
import torch
import pytest


@pytest.fixture
def dataset_fixture():
    features = torch.ones([10, 5])
    outputs = torch.randn([10])
    dataset = TensorDataset(features, outputs)
    n_fold = 5
    return dict(dataset=dataset, n_fold=n_fold)


def test_run_one_cv_evaluation(mocker):
    evaluation_function_mock = mocker.Mock(return_value=1)
    evaluator = HydraCVEvaluator(model=mocker.Mock(),
                                 data_handler=mocker.Mock(SupervisedLearningDataHandler),
                                 evaluation_function=evaluation_function_mock,
                                 trainer=mocker.Mock(Trainer))
    results = evaluator._run_one_cv_evaluation(mocker.MagicMock(), mocker.MagicMock(), {})
    assert results == 1


def test_run_evaluatoin(mocker, dataset_fixture):
    handler = SupervisedLearningDataHandler(**dataset_fixture)
    evaluation_function_mock = mocker.Mock(return_value=1)
    evaluator = HydraCVEvaluator(model=mocker.Mock(),
                                 data_handler=handler,
                                 evaluation_function=evaluation_function_mock,
                                 trainer=mocker.Mock(Trainer))
    results = evaluator.run_evaluation({})
    assert results == 1
