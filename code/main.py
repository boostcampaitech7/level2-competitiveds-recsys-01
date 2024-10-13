from utils.constant_utils import Directory
from utils import common_utils
import preprocessing
import preprocessing_fn
import features

import model
from inference import *

def main():
    print("Start the main.py successfully!")
    df = common_utils.merge_data(Directory.train_data, Directory.test_data)

    # 클러스터 피처 apply
    for info_df_name in ['subway_info', 'school_info', 'park_info']:
        info_df = getattr(Directory, info_df_name)  
        df = features.clustering(df, info_df, feat_name=info_df_name, n_clusters=20)

    # 이상치 처리
    df = preprocessing_fn.handle_outliers(df)
    
    # 로그 변환
    #df = preprocessing_fn.log_transform(df, 'floor')

    # 계약 유형 피처 카테고리 변환
    df = preprocessing_fn.numeric_to_categoric(df, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})

    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)

    # 중복 제거
    train_data_ = preprocessing_fn.handle_duplicates(train_data_)
    valid_data_ = preprocessing_fn.handle_duplicates(valid_data_)

    # 전처리 적용
    train_data_ = preprocessing.time_feature_preprocessing(train_data_)
    valid_data_ = preprocessing.time_feature_preprocessing(valid_data_)
    test_data_ = preprocessing.time_feature_preprocessing(test_data_)

    # 정규화
    train_data_, valid_data_, test_data_ = preprocessing_fn.standardization(train_data_, valid_data_, test_data_)

    # feature selection
    train_data_scaled, valid_data_scaled, test_data_scaled = preprocessing_fn.feature_selection(train_data_, valid_data_, test_data_)

    # feature split
    X_train, y_train, X_valid, y_valid, X_test = common_utils.split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled)
    
    # train model
    print("Train the model")
    model_ = model.xgboost(X_train, y_train)

    prediction, mae = inference(model_, 'validation', X_valid, y_valid)
    print(mae)

    # train with total dataset
    print("Train with total dataset")
    X_total, y_total = common_utils.train_valid_concat(X_train, X_valid, y_train, y_valid)
    model_ = model.xgboost(X_total, y_total)
    
    # inference with test data
    submission = inference(model_, 'submission', X_test)

    # save sample submission
    common_utils.submission_to_csv(submission, 'cluster,timefeature,categorical,xgboost(1000)')

    return prediction, mae

if __name__ == "__main__":
    prediction, mae = main()