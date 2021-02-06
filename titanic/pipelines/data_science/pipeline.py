from kedro.pipeline import Pipeline, node
from .evaluation.evaluate import run_evaluation
from .parameter_search.search_parameter import run_parameter_search
from .model.inference import run_inference
from .model.train import train


def create_training_pipeline(**kwargs):
    return Pipeline([
        node(
            run_parameter_search,
            inputs=['params:model', 'params:trainer', 'train_data_handler', 'params:evaluation_for_parameter_search'],
            outputs='searched_params'
        ),
        node(
            train,
            inputs=['params:model',  'params:trainer', 'train_data_handler', 'searched_params'],
            outputs='trained_model'
        ),
    ])


def create_evaluation_pipeline(**kwargs):
    return Pipeline([
        node(
            run_evaluation,
            inputs=['params:model', 'params:trainer', 'train_data_handler', 'params:evaluation_task_names',
                    'searched_params', 'params:project_name'],
            outputs='done_evaluation'
        )
    ])


def create_inference_pipeline(**kwargs):
    return Pipeline([
        node(
            run_inference,
            inputs=['trained_model', 'inference_data_handler'],
            outputs='inference_output'
        )
    ])
