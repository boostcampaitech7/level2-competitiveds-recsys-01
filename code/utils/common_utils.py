import pandas as pd
import os
import time
import numpy as np

from utils.constant_utils import Directory, Config

# 데이터 병합 함수
def merge_data(train_data, test_data):
    train_data['type'] = 'train'
    test_data['type'] = 'test'

    df = pd.concat([train_data, test_data], axis = 0)
    df.drop(['index'], axis = 1, inplace = True)

    interest_rate = Directory.interest_rate.rename(columns = {'year_month' : 'contract_year_month'})

    # 202406 데이터 202405 값으로 추가
    interest_rate.loc[len(interest_rate)] = [202406, 3.56]
    interest_rate['contract_year_month'] = interest_rate['contract_year_month'].astype(int)
    df = df.merge(interest_rate, on='contract_year_month', how='left')
    return df


# 데이터 분할 함수
def train_valid_test_split(df, log_transform: str = None):
    # 데이터 분할
    train_data = df[df['type'] == 'train']
    test_data = df[df['type'] == 'test']

    valid_start = Config.SPLIT_YEAR_START
    valid_end = Config.SPLIT_YEAR_END

    valid_data = train_data[(train_data['contract_year_month'] >= valid_start) & (train_data['contract_year_month'] <= valid_end)]
    train_data = train_data[~((train_data['contract_year_month'] >= valid_start) & (train_data['contract_year_month'] <= valid_end))]
    
    # 인덱스 정리
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # log 변환
    if log_transform == 'log':
        train_data['deposit'] = np.log1p(train_data['deposit'])
        valid_data['deposit'] = np.log1p(valid_data['deposit'])

    return train_data, valid_data, test_data


# target 분리 함수
def split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled):
    X_train = train_data_scaled.drop(columns=['deposit'])
    y_train = train_data_scaled['deposit']
    X_valid = valid_data_scaled.drop(columns=['deposit'])
    y_valid = valid_data_scaled['deposit']
    X_test = test_data_scaled.drop(columns=['deposit'], errors='ignore')
    
    return X_train, y_train, X_valid, y_valid, X_test


# train과 valid 병합 함수(total dataset 구축)
def train_valid_concat(X_train, X_valid, y_train, y_valid):
    X_total, y_total = pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0)
    return X_total, y_total


# submission 저장 함수
def submission_to_csv(submit_df, file_name):
    submission_path = os.path.join(Directory.result_path, "submission")
    os.makedirs(submission_path, exist_ok=True)

    file_name += '_' + time.strftime('%x', time.localtime())[:5].replace('/','') + '.csv'

    submission_file_path = os.path.join(submission_path, file_name)
    submit_df.to_csv(submission_file_path, index=False, encoding='utf-8-sig')


'''
name : 실험 진행자
title : 실험 구성요소 (모델명, 피처들)
hyperparams : 하이퍼파라미터 구성
MAE score : MAE score
'''
def mae_to_csv(name, title, hyperparams, mae):
    mae_path = os.path.join(Directory.result_path, "mae.csv")
    new_dict = {"name":[name], "title": [title], "hyperparams":[hyperparams], "MAE score":[mae]}

    if not os.path.exists(mae_path):
        mae_df = pd.DataFrame(new_dict)

    else:
        mae_df = pd.read_csv(mae_path)
        mae_df = pd.concat([mae_df, pd.DataFrame(new_dict)], axis=0)    

    mae_df.to_csv(mae_path, index=False, encoding='utf-8-sig')

    
# saving data and load function
def save_and_load_function(data: list, file_name: str, mode: str) -> list:
    # 저장 경로
    save_path = os.path.join(Directory.root_path, "level2-competitiveds-recsys-01/data/transaction_data", file_name + ".txt")
    
    # save 모드
    if mode == 'save':
        with open(save_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n") 
    
    # load 모드
    elif mode == 'load':
        loaded_data = []
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                loaded_data = f.read().splitlines()  
        return loaded_data
    
    else:
        raise ValueError("mode should be either 'save' or 'load'") 
