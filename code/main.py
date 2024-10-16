from utils.constant_utils import Directory
from utils import common_utils
import preprocessing
import features

import model
from inference import *
from utils.common_utils import submission_to_csv, mae_to_csv

def main():
    print("Start the main.py successfully!")
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
    train_data_ = features.create_temporal_feature(train_data_)
    valid_data_ = features.create_sin_cos_season(valid_data_)
    test_data_ = features.create_floor_area_interaction(test_data_)

    # 새로운 피처 추가
    train_data, valid_data, test_data = features.create_clustering_target(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_nearest_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_subway_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_nearest_park_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_school_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_sum_park_area_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)

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

    # train with total dataset
    print("Train with total dataset")
    X_total, y_total = common_utils.train_valid_concat(X_train, X_valid, y_train, y_valid)
    model_ = model.xgboost(X_total, y_total)

    # inference with test data
    submission = inference(model_, 'submission', X_test)

    # save sample submission
    common_utils.submission_to_csv(submission, 'cluster(20),timefeature,feat2_feature,xgboost(1000)')

    return prediction, mae

if __name__ == "__main__":
    prediction, mae = main()
    print(mae)