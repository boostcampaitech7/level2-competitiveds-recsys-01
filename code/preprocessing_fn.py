import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

from utils.constant_utils import Config


def create_temporal_feature(df: pd.DataFrame)-> pd.DataFrame:
    df_preprocessed = df.copy()
    
    df_preprocessed['year'] = df_preprocessed['contract_year_month'].astype(str).str[:4].astype(int)
    df_preprocessed['month'] = df_preprocessed['contract_year_month'].astype(str).str[4:].astype(int)
    df_preprocessed['date'] = pd.to_datetime(df_preprocessed['year'].astype(str) + df_preprocessed['month'].astype(str).str.zfill(2) + df_preprocessed['contract_day'].astype(str).str.zfill(2))

    # 기본 특성 생성 (모든 데이터셋에 동일하게 적용 가능)
    df_preprocessed['day_of_week'] = df_preprocessed['date'].dt.dayofweek
    #df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df_preprocessed['quarter'] = df_preprocessed['date'].dt.quarter
    df_preprocessed['is_month_end'] = (df_preprocessed['date'].dt.is_month_end).astype(int)
    df_preprocessed['season'] = df_preprocessed['month'].map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                                    5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
                                    9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})
    return df_preprocessed


def create_sin_cos_season(df: pd.DataFrame)-> pd.DataFrame:
    df_preprocessed = df.copy()
    # Cyclical encoding for seasons
    season_dict = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    df_preprocessed['season_numeric'] = df_preprocessed['season'].map(season_dict)
    df_preprocessed['season_sin'] = np.sin(2 * np.pi * df_preprocessed['season_numeric'] / 4)
    df_preprocessed['season_cos'] = np.cos(2 * np.pi * df_preprocessed['season_numeric'] / 4)

    df_preprocessed = df_preprocessed.drop(['season_numeric'], axis=1)
    return df_preprocessed


def create_floor_area_interaction(df: pd.DataFrame)-> pd.DataFrame:
    df_preprocessed = df.copy()

    df_preprocessed['floor_area_interaction'] = df_preprocessed['floor'] * df_preprocessed['area_m2']
    return df_preprocessed


def feature_selection(train_data_scaled: pd.DataFrame, valid_data_scaled: pd.DataFrame, test_data_scaled: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drop_columns = ['type', 'season', 'date']
    train_data_scaled.drop(drop_columns, axis = 1, inplace = True)
    valid_data_scaled.drop(drop_columns, axis = 1, inplace = True)
    test_data_scaled.drop(drop_columns + ['deposit'], axis = 1, inplace = True)

    return train_data_scaled, valid_data_scaled, test_data_scaled


def standardization(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exclude_cols = ['type', 'date', 'season', 'deposit']


    features_to_scale = [col for col in train_data.columns 
                     if col not in exclude_cols and train_data[col].dtype in ['int64', 'float64']]


    scaler = StandardScaler()
    train_data_scaled = train_data.copy()
    train_data_scaled[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])

    valid_data_scaled = valid_data.copy()
    valid_data_scaled[features_to_scale] = scaler.transform(valid_data[features_to_scale])

    test_data_scaled = test_data.copy()
    test_data_scaled[features_to_scale] = scaler.transform(test_data[features_to_scale])

    return train_data_scaled, valid_data_scaled, test_data_scaled

def handle_outliers(total_df):
    new_df = total_df.copy()
    deposit = total_df['deposit']
    weight = 1.5

    Q1 = 0
    Q3 = np.percentile(deposit.values, 75)

    iqr = Q3 - Q1
    iqr_weight = iqr * weight

    lowest_val = Q1 - iqr_weight
    # 최솟값
    highest_val = Q3 + iqr_weight
    # 최댓값

    low_outlier_index = deposit[(deposit < lowest_val)].index
    high_outlier_index = deposit[(deposit > highest_val)].index

    # 최솟값보다 작고, 최댓값보다 큰 이상치 데이터들의 인덱스
    new_df.loc[low_outlier_index,'deposit'] = lowest_val
    new_df.loc[high_outlier_index,'deposit'] = highest_val

    # 전체 데이터에서 이상치 데이터 제거
    new_df.reset_index(drop = True, inplace = True)

    return new_df

def handle_duplicates(df):
    df.drop_duplicates(subset=['area_m2', 'contract_year_month', 'contract_day', 'contract_type', 'floor', 'latitude', 'longitude', 'age', 'deposit'], inplace = True)
    return df


def log_transform(df, column):
    df.loc[df[column] > 0, column] = np.log(df[column][df[column] > 0]).astype(float)
    df.loc[df[column] < 0, column] = -np.log(abs(df[column][df[column] < 0])).astype(float)
    return df

def numeric_to_categoric(df, column, map_dict):
    df[column] = df[column].map(map_dict).astype('category')
    return df

# 리스트 형태로 전달할 것
def drop_columns(df, columns):
    df.drop(columns=columns, inplace = True)
    return df