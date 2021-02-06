from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from typing import Dict, Any, Generator, NamedTuple, Optional, List
from torchvision import transforms
import torch
import numpy as np
import pandas as pd


from logging import getLogger

logger = getLogger(__name__)


class CVPair(NamedTuple):
    train: Subset
    validation: Subset


class CVDataLoaderPair(NamedTuple):
    train: DataLoader
    validation: DataLoader


class SupervisedLearningDataHandler:
    def __init__(
        self, dataset: Dataset, n_fold: Optional[int] = None, train_data_loader_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self._dataset = dataset
        self._cv_subsets = self._split_data_to_k_fold_cv_subsets(dataset, n_fold)
        self._train_data_loader_params = train_data_loader_params

    @staticmethod
    def _split_data_to_k_fold_cv_subsets(dataset: Dataset, n_fold: int) -> List[CVPair]:
        n_validation_data = len(dataset) // n_fold
        perm = np.random.permutation(len(dataset))
        cv_subsets = []
        for i in range(n_fold):
            boolean_index = np.zeros(len(dataset)).astype(bool)
            p = perm[i * n_validation_data: (i + 1) * n_validation_data]
            boolean_index[p] = True
            train_subset = Subset(dataset, np.where(~boolean_index)[0])
            validation_subset = Subset(dataset, np.where(boolean_index)[0])
            cv_subsets.append(CVPair(train=train_subset, validation=validation_subset))
        return cv_subsets

    def get_data_loader_of_cv_pairs(self) -> Generator[CVDataLoaderPair, None, None]:
        for cv_pair in self._cv_subsets:
            if self._train_data_loader_params:
                yield CVDataLoaderPair(train=DataLoader(cv_pair.train, **self._train_data_loader_params),
                                       validation=DataLoader(cv_pair.validation, **self._train_data_loader_params))
            else:
                yield CVDataLoaderPair(train=DataLoader(cv_pair.train), validation=DataLoader(cv_pair.validation))

    def get_cv_subsets(self):
        for cv_pair in self._cv_subsets:
            yield cv_pair

    def get_data_loader(self):
        if self._train_data_loader_params:
            return DataLoader(self._dataset, **self._train_data_loader_params)
        else:
            return DataLoader(self._dataset)

    def get_dataset(self):
        return self._dataset


class ImageDataset(Dataset):
    def __init__(
        self, original_images: torch.Tensor, labels: Optional[torch.LongTensor] = None, inference_mode: bool = False
    ) -> None:
        self._original_images = original_images
        self._labels = labels
        self._transforms = self._setup_transforms()
        self._inference_mode = inference_mode
        self._resize = transforms.Resize((256, 256))

    def _setup_transforms(self):
        transform_seqential = torch.nn.Sequential(
            transforms.RandomRotation((-180, 180)),
            transforms.RandomResizedCrop((256, 256))
        )
        scripted_transforms = torch.jit.script(transform_seqential)
        return scripted_transforms

    def __getitem__(self, index: int) -> None:
        tensor = (torch.Tensor(self._original_images[index]) - 128) / 128
        if not self._inference_mode:
            tensor = self._transforms(tensor)
        return (tensor, self._labels[index])

    def __len__(self):
        return len(self._original_images)


class SupervisedLearningInferenceDataHandlerMixin:
    def __init__(self, data: pd.DataFrame, primary_id_column: str, output_column_name: str) -> None:
        self._data = data
        self._id_column = primary_id_column
        self._id_series = data.pop(self._id_column)
        self._output_column_name = output_column_name

    def format_outputs(self, predicted: torch.Tensor) -> pd.DataFrame:
        df = pd.DataFrame({self._id_column: self._id_series.reset_index(drop=True),
                           self._output_column_name: predicted.numpy().astype(int)})
        return df


class SupervisedLearningInferenceDataHandler(SupervisedLearningInferenceDataHandlerMixin):
    def get_dataset(self) -> torch.Tensor:
        return torch.Tensor(self._data.to_numpy())


class ImageSupervisedLearningInferenceDataHandler(SupervisedLearningInferenceDataHandlerMixin):
    def get_dataset(self) -> torch.Tensor:
        return ImageDataset(torch.LongTensor(self._data['image']), inference_mode=True)[:]


def generate_supervised_learning_data_handler(
    features: pd.DataFrame, outputs: pd.DataFrame, n_fold: int, id_column: str,
    train_data_loader_params: Optional[Dict[str, Any]] = None, dataset_type: str = 'tensor'
) -> SupervisedLearningDataHandler:
    inputs = dict(
        features=features, outputs=outputs, n_fold=n_fold, id_column=id_column,
        train_data_loader_params=train_data_loader_params
    )
    function_map = dict(
        tensor=_generate_table_supervised_learning_data_handler(**inputs),
        image=_generate_image_supervised_learning_data_handler(**inputs)
    )
    return function_map[dataset_type]


def _generate_table_supervised_learning_data_handler(
    features: pd.DataFrame, outputs: pd.DataFrame, n_fold: int, id_column: str,
    train_data_loader_params: Optional[Dict[str, Any]] = None,
) -> SupervisedLearningDataHandler:
    logger.info(f'features.shape: {features.shape}')
    logger.info(f'outputs.shape: {outputs.shape}')

    features = features.sort_values(by=id_column)
    features.pop(id_column)
    features = torch.Tensor(features.to_numpy())
    outputs = outputs.sort_values(by=id_column)
    outputs.pop(id_column)
    outputs = torch.Tensor(np.squeeze(outputs.to_numpy()))
    return SupervisedLearningDataHandler(TensorDataset(features, outputs), n_fold, train_data_loader_params)


def _generate_image_supervised_learning_data_handler(
    features: pd.DataFrame, outputs: pd.DataFrame, n_fold: int, id_column: str,
    train_data_loader_params: Optional[Dict[str, Any]] = None,
) -> SupervisedLearningDataHandler:
    assert 'image' in features.columns, 'features must have `image` column.'
    features = features.sort_values(by=id_column)
    outputs = outputs.sort_values(by=id_column)

    features = features.pop('image').to_list()
    features = torch.Tensor(features)

    outputs = outputs.pop(id_column)
    outputs = torch.LongTensor(outputs)
    return SupervisedLearningDataHandler(ImageDataset(features, outputs), n_fold, train_data_loader_params)


def generate_supervised_learning_inference_data_handler(
    features: pd.DataFrame, id_column: str, output_column_name: str, dataset_type: str
) -> SupervisedLearningInferenceDataHandler:
    return SupervisedLearningInferenceDataHandler(features, id_column, output_column_name)
