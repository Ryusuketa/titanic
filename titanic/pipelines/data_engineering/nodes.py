from typing import Any, Callable, Dict
import logging
import numpy as np

import pandas as pd


log = logging.getLogger(__name__)


def format_dataset(images: Dict[Any, Callable]):
    image_list = []
    for image_id, image_loader in list(images.items()):
        image = np.asarray(image_loader(), dtype=np.uint8)
        image_list.append(dict(image_id=image_id, image=image))

    images = pd.DataFrame.from_dict(image_list)
    return images


def load_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    return df


def extract_basic_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


def transform_to_dummy(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'])


def fillna(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(0)
