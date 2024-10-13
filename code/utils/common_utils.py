import pandas as pd
import os
import time

from utils.constant_utils import Directory


def merge_data(train_data, test_data):
    train_data['type'] = 'train'
    test_data['type'] = 'test'

    df = pd.concat([train_data, test_data], axis = 0)
    df.drop(['index'], axis = 1, inplace = True)

    interest_rate = Directory.interest_rate.rename(columns = {'year_month' : 'contract_year_month'})
    df = df.merge(interest_rate, on='contract_year_month', how='left')
    return df


def train_valid_test_split(df):
    # 데이터 분할
    train_data = df[df['type'] == 'train']
    test_data = df[df['type'] == 'test']

    valid_start = 202307
    valid_end = 202312

    valid_data = train_data[(train_data['contract_year_month'] >= valid_start) & (train_data['contract_year_month'] <= valid_end)]
    train_data = train_data[~((train_data['contract_year_month'] >= valid_start) & (train_data['contract_year_month'] <= valid_end))]

    return train_data, valid_data, test_data

def train_valid_concat(train, valid):
    total = pd.concat([train, valid])
    return total
def split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled):
    X_train = train_data_scaled.drop(columns=['deposit'])
    y_train = train_data_scaled['deposit']
    X_valid = valid_data_scaled.drop(columns=['deposit'])
    y_valid = valid_data_scaled['deposit']
    X_test = test_data_scaled.copy()
    
    return X_train, y_train, X_valid, y_valid, X_test


def submission_to_csv(submit_df, file_name):
    submission_path = os.path.join(Directory.result_path, "submission")
    os.makedirs(submission_path, exist_ok=True)

    file_name += '_' + time.strftime('%x', time.localtime())[:5].replace('/','') + '.csv'

    submission_file_path = os.path.join(submission_path, file_name)
    submit_df.to_csv(submission_file_path, index=False, encoding='utf-8-sig')


def mae_to_csv(mae_df, file_name):
    mae_path = os.path.join(Directory.result_path, "mae")
    os.makedirs(mae_path, exist_ok=True)

    file_name += '_' + time.strftime('%x', time.localtime())[:5].replace('/','') + '.csv'
    mae_file_path = os.path.join(mae_path, file_name)
    mae_df.to_csv(mae_file_path, index=False, encoding='utf-8-sig')
