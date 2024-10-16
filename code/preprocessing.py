import os
import pandas as pd
import numpy as np

from preprocessing_fn import *

def time_feature_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = create_temporal_feature(df)
    df = create_sin_cos_season(df)
    df = create_floor_area_interaction(df)
    #df = remove_built_year_2024(df)
    #df = feature_selection(df)
    return df

# def test1_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
#     df = create_temporal_feature(df)
#     df = create_sin_cos_season(df)
#     df = create_floor_area_interaction(df)
#     df = feature_selection(df)

#     return df

# def test2_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
#     df = create_temporal_feature(df)
#     df = create_sin_cos_season(df)
#     df = create_floor_area_interaction(df)
#     df = feature_selection(df)

#     return df
