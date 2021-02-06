from kedro.pipeline import Pipeline, node
from .data_handler import generate_supervised_learning_data_handler,\
    generate_supervised_learning_inference_data_handler


def create_data_handler(**kwargs):
    return Pipeline([
        node(
            generate_supervised_learning_data_handler,
            inputs=['model_input_features', 'train_outputs', 'params:n_fold', 'params:data_primary_key',
                    'params:train_data_loader_params', 'params:dataset_type'],
            outputs='train_data_handler'
        )
    ])


def create_inference_data_handler(**kwargs):
    return Pipeline([
        node(
            generate_supervised_learning_inference_data_handler,
            inputs=['model_input_features',
                    'params:data_primary_key',
                    'params:output_column_name',
                    'params:dataset_type'],
            outputs='inference_data_handler'
        )
    ])
