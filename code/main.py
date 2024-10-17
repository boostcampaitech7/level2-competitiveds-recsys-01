from utils.constant_utils import Directory
from utils import common_utils
import preprocessing
import features

import preprocessing_fn
import features
import pandas as pd


import model
from inference import *

def main():
    print("Start the main.py successfully!")

    '''
    name : 실험자 이름입니다.
    title : result 폴더에 저장될 실험명을 지정합니다.
    '''
    name = 'jinnk0'
    title = 'cluster,timefeature,categorical,drop,gangnam,xgb1000'

    df = common_utils.merge_data(Directory.train_data, Directory.test_data)

    # 클러스터 피처 apply
    for info_df_name in ['subway_info', 'school_info', 'park_info']:
        info_df = getattr(Directory, info_df_name)
        df = features.clustering(df, info_df, feat_name=info_df_name, n_clusters=15)

    # 이상치 처리
    df = preprocessing.handle_outliers(df)

    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)

    # 중복 제거
    train_data_ = preprocessing.handle_duplicates(train_data_)
    valid_data_ = preprocessing.handle_duplicates(valid_data_)

    # 전처리 적용
    train_data_, valid_data_, test_data_ = features.create_temporal_feature(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_sin_cos_season(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_floor_area_interaction(train_data_, valid_data_, test_data_)

    # 새로운 피처 추가
    train_data_, valid_data_, test_data_ = features.create_clustering_target(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_nearest_subway_distance(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_subway_within_radius(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_nearest_park_distance(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_school_within_radius(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_sum_park_area_within_radius(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_school_counts_within_radius_by_school_level(train_data_, valid_data_, test_data_)

    # 정규화
    train_data_, valid_data_, test_data_ = preprocessing.standardization(train_data_, valid_data_, test_data_)

    # feature selection
    train_data_scaled, valid_data_scaled, test_data_scaled = preprocessing.feature_selection(train_data_, valid_data_, test_data_)

    # feature split
    X_train, y_train, X_valid, y_valid, X_test = common_utils.split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled)
    
    # train model
    print("Train the model")
    model_ = model.xgboost(X_train, y_train)

    prediction, mae = inference(model_, 'validation', X_valid, y_valid)

    feature_names = X_train.columns

    # 피처 중요도를 DataFrame으로 변환
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model_.feature_importances_
    })

    # 중요도를 기준으로 정렬
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 출력
    print(importance_df)

    # record MAE score as csv
    hyperparams = "learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED"
    common_utils.mae_to_csv(name, title, hyperparams=hyperparams, mae = mae)

    # train with total dataset
    print("Train with total dataset")
    X_total, y_total = common_utils.train_valid_concat(X_train, X_valid, y_train, y_valid)
    model_ = model.xgboost(X_total, y_total)

    # inference with test data
    submission = inference(model_, 'submission', X_test)

    # save sample submission
    common_utils.submission_to_csv(submission, title)

    return prediction, mae

if __name__ == "__main__":
    prediction, mae = main()
    print(mae)
