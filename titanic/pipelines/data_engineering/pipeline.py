from kedro.pipeline import Pipeline, node
from kedro.config import ConfigLoader
from titanic.pipelines.data_engineering.nodes import format_dataset, load_inference_data
from titanic.pipelines.data_engineering.format_features import merge_features, format_features_for_inference,\
    extract_feature_columns
from enum import Enum


class Mode(Enum):
    TRAIN = 1
    INFERNECE = 2


def create_data_wrangler_pipeline(mode: str = 'train', **kwargs):
    if mode == 'train':
        data_load_node = node(
            format_dataset, inputs='train_images', outputs='features'
        )
    elif mode == 'inference':
        data_load_node = node(load_inference_data, inputs='test_images', outputs='features')
    else:
        raise ValueError('mode must be selected from `train` or `inference`.')

    return Pipeline([
        data_load_node
    ])


def create_data_merge_pipeline(mode: str = 'train', **kwargs):
    conf_loader = ConfigLoader('conf/local/parameters/')
    use_feature_names = conf_loader.get('data_engineering_parameters.yml')['use_feature_names']
    if mode == 'train':
        return Pipeline([
            node(
                merge_features, inputs=['params:data_primary_key', *use_feature_names], outputs='model_input_features'
            ),
            node(
                extract_feature_columns, inputs='model_input_features', outputs='output_columns'
            )
        ])
    elif mode == 'inference':
        return Pipeline([
            node(
                merge_features, inputs=['params:data_primary_key', *use_feature_names],
                outputs='tmp_model_input_features'
            ),
            node(
                format_features_for_inference,
                inputs=['tmp_model_input_features', 'output_columns'],
                outputs='model_input_features'
            )
        ])
