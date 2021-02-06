import pandas as pd
from functools import reduce


def merge_features(primary_key: str, *args):
    if len(args) == 1:
        return args[0]
    return reduce(lambda left, right: pd.merge(left, right, on=primary_key), args)


def format_features_for_inference(inference_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    return inference_df.reindex(columns=train_columns, fill_value=0)


def extract_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.columns
