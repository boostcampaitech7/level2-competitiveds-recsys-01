import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans

from utils.constant_utils import Config



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

def handle_outliers(df):
    train_df = df[df['type'] == 'train']
    test_df = df[df['type'] == 'test']

    Q1 = train_df['deposit'].quantile(0.20)
    Q3 = train_df['deposit'].quantile(0.80)
    IQR = Q3 - Q1

    factor=1.5
    lower_limit=train_df['deposit'].min()

    lower_bound = max(Q1 - factor * IQR, lower_limit)
    upper_bound = Q3 + factor * IQR

    low_outlier_index = train_df[(train_df['deposit'] < lower_bound)].index
    high_outlier_index = train_df[(train_df['deposit'] > upper_bound)].index

    train_df.loc[low_outlier_index, 'deposit'] = lower_bound
    train_df.loc[high_outlier_index, 'deposit'] = upper_bound

    total_df = pd.concat([train_df, test_df], axis = 0)
    return total_df

def handle_duplicates(df):
    df.drop_duplicates(subset=['area_m2', 'contract_year_month', 'contract_day', 'contract_type', 'floor', 'latitude', 'longitude', 'age', 'deposit'], inplace = True)
    return df

def area_square(df):
    df['area_m2'] = df['area_m2'] ** 2
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

def handle_age_outliers(df):
    df = df[df['age']>=0]
    df.reset_index(inplace=True)
    return df



# 방법 2: sklearn의 PowerTransformer 사용 (Box-Cox 방법)
def boxcox_transform(y_train):
    # PowerTransformer 초기화 (Box-Cox 방법 사용)
    pt = PowerTransformer(method='box-cox')
    
    # 훈련 데이터로 변환기 학습 및 변환
    y_train_transformed = pt.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
    
    return y_train_transformed, pt

def boxcox_re_transform(prediction, pt):
    y_train_pred = pt.inverse_transform(prediction.reshape(-1, 1)).flatten()

    return y_train_pred

def target_yeo_johnson(data):
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(np.array(data['deposit']).reshape(-1, 1))

    data['transformed_deposit'] = y_transformed
    
    data.drop(['deposit'], axis = 1, inplace = True)
    data.rename(columns = {'transformed_deposit' : 'deposit'}, inplace = True)

    return pt, data


def handle_outliers(df):
    factor=1.5
    lower_limit=0

    train_df = df[df['type'] == 'train']
    test_df = df[df['type'] == 'test']

    Q1 = train_df['deposit'].quantile(0.25)
    Q3 = train_df['deposit'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - factor * IQR, lower_limit)
    upper_bound = Q3 + factor * IQR

    filtered_train_df = train_df[(train_df['deposit'] >= lower_bound) & (train_df['deposit'] <= upper_bound)]
    total_df = pd.concat([filtered_train_df, test_df], axis = 0)
    
    return total_df