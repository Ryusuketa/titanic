import pandas as pd
from typing import Any
from titanic.pipelines.data_handlers.data_handler import SupervisedLearningInferenceDataHandler


def run_inference(model: Any, test_data_handler: SupervisedLearningInferenceDataHandler) -> pd.DataFrame:
    inputs = test_data_handler.get_dataset()
    predicted = model.predict(inputs)
    return test_data_handler.format_outputs(predicted)
