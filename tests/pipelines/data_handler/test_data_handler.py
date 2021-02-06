import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from titanic.pipelines.data_handlers.data_handler import SupervisedLearningDataHandler


@pytest.fixture
def dataset_fixture():
    features = torch.ones([10, 5])
    outputs = torch.randn([10])
    dataset = TensorDataset(features, outputs)
    n_fold = 5
    return dict(dataset=dataset, n_fold=n_fold)


def test_split_data_to_k_fold_cv_subsets(dataset_fixture):
    results = SupervisedLearningDataHandler._split_data_to_k_fold_cv_subsets(**dataset_fixture)
    assert len(results[0].train) == 8
    assert len(results[0].validation) == 2
    assert len(results) == 5
    assert (results[0].validation[0][0] == results[1].validation[0][0]) is not True


def test_init_supervised_learning_data_handler(dataset_fixture):
    SupervisedLearningDataHandler(**dataset_fixture)


def test_get_data_loader_of_cv_pairs_list(dataset_fixture):
    results = list(SupervisedLearningDataHandler(**dataset_fixture).get_data_loader_of_cv_pairs())
    assert len(list(results)) == 5


def test_get_data_loader(dataset_fixture):
    results = SupervisedLearningDataHandler(**dataset_fixture).get_data_loader_of_cv_pairs()
    assert isinstance(next(results).train, DataLoader)


def test_get_cv_subsets(dataset_fixture):
    results = SupervisedLearningDataHandler(**dataset_fixture).get_cv_subsets()
    assert len(list(results)) == 5
    results = SupervisedLearningDataHandler(**dataset_fixture).get_cv_subsets()
    assert isinstance(next(results).train, Dataset)
