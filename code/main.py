from utils.constant_utils import Directory
from utils import common_utils
import preprocessing
import preprocessing_fn
import features

import model
from inference import *

def main():
    print("Start the main.py successfully!")

    '''
    name : 실험자 이름입니다.
    title : result 폴더에 저장될 실험명을 지정합니다.
    '''
    name = 'lim'
    title = 'cluster,area_m2,drop_2024_age_contract_type,school,subway'

    df = common_utils.merge_data(Directory.train_data, Directory.test_data)

    # 클러스터 피처 apply
    for info_df_name in ['subway_info', 'school_info', 'park_info']:
        info_df = getattr(Directory, info_df_name)  
        df = features.clustering(df, info_df, feat_name=info_df_name, n_clusters=15)

    # 이상치 처리
    df = preprocessing_fn.handle_outliers(df)

    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)
    
    # 로그 변환
    #df = preprocessing_fn.log_transform(df, 'floor')

    # 중복 제거
    train_data_ = preprocessing_fn.handle_duplicates(train_data_)
    valid_data_ = preprocessing_fn.handle_duplicates(valid_data_)

    # built_year=2024 제거
    train_data_ = preprocessing_fn.remove_built_year_2024(train_data_)
    valid_data_ = preprocessing_fn.remove_built_year_2024(valid_data_)

    # 전처리 적용
    train_data_ = preprocessing.time_feature_preprocessing(train_data_)
    valid_data_ = preprocessing.time_feature_preprocessing(valid_data_)
    test_data_ = preprocessing.time_feature_preprocessing(test_data_)

    # 계약일 피처 제거
    # train_data_ = preprocessing_fn.drop_columns(train_data_, ['contract_day'])
    # valid_data_ = preprocessing_fn.drop_columns(valid_data_, ['contract_day'])
    # test_data_ = preprocessing_fn.drop_columns(test_data_, ['contract_day'])

    train_data_ = preprocessing_fn.drop_columns(train_data_, ['age'])
    valid_data_ = preprocessing_fn.drop_columns(valid_data_, ['age'])
    test_data_ = preprocessing_fn.drop_columns(test_data_, ['age'])
    train_data_ = preprocessing_fn.drop_columns(train_data_, ['contract_type'])
    valid_data_ = preprocessing_fn.drop_columns(valid_data_, ['contract_type'])
    test_data_ = preprocessing_fn.drop_columns(test_data_, ['contract_type'])

    # 새로운 피처 추가
    #train_data_, valid_data_, test_data_ = features.create_nearest_subway_distance(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_subway_within_radius(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_nearest_park_distance_and_area(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_school_within_radius(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.creat_area_m2_category(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.create_nearest_school_distance(train_data_, valid_data_, test_data_)
    train_data_, valid_data_, test_data_ = features.weighted_subway_distance(train_data_, valid_data_, test_data_)

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
