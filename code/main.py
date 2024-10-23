from utils.constant_utils import Directory
from utils import common_utils

import preprocessing
from features.clustering_features import *
from features.count_features import *
from features.distance_features import *
from features.other_features import *

from tqdm import tqdm

import model
from inference import *
import warnings
warnings.filterwarnings('ignore')


def main():
    print("Start the main.py successfully!")

    '''
    name : 실험자 이름입니다.
    title : result 폴더에 저장될 실험명을 지정합니다.
    '''
    name = 'jinnk0'
    title = 'cluster,timefeature,categorical,drop,gangnam,xgb1000'

    print("total data load ...")
    df = common_utils.merge_data(Directory.train_data, Directory.test_data)

    ### 클러스터 피처 apply
    print("clustering apply ...")
    for info_df_name in ['subway_info', 'school_info', 'park_info']:
        info_df = getattr(Directory, info_df_name)  
        df = clustering(df, info_df, feat_name=info_df_name, n_clusters=20)

    ### 이상치 처리
    print("start to cleaning outliers...")
    df = preprocessing.handle_age_outliers(df)

    ### 데이터 분할
    print("train, valid, test split for preprocessing & feature engineering ...")
    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)


    ### 데이터 전처리
    print("start to preprocessing...")
    # type 카테고리화
    train_data_ = preprocessing.numeric_to_categoric(train_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
    valid_data_ = preprocessing.numeric_to_categoric(valid_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
    test_data_ = preprocessing.numeric_to_categoric(test_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})

    # 중복 제거
    train_data_ = preprocessing.handle_duplicates(train_data_)
    valid_data_ = preprocessing.handle_duplicates(valid_data_)
    
    # 로그 변환
    #df = preprocessing_fn.log_transform(df, 'deposit')


    ### 피처 엔지니어링
    print("start to feature engineering...")
    # clustering_feature
    print("create clustering features")
    train_data, valid_data, test_data = create_clustering_target(train_data_, valid_data_, test_data_)

    # distance_features
    print("create distance features")
    train_data, valid_data, test_data = distance_gangnam(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_nearest_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_nearest_school_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = weighted_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_nearest_park_distance_and_area(train_data, valid_data, test_data)

    # other_features
    print("create other features")
    train_data, valid_data, test_data = create_temporal_feature(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_sin_cos_season(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_floor_area_interaction(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_sum_park_area_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = shift_interest_rate_function(train_data, valid_data, test_data)
    train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'age')
    train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'floor')
    train_data, valid_data, test_data = categorization(train_data, valid_data, test_data, category = 'area_m2')

    
    # count_features
    print("create count features")
    train_data, valid_data, test_data = transaction_count_function(train_data, valid_data, test_data)
    # 위의 함수를 바로 실행하기 위한 구조 : data/transaction_data에 train/valid/test_transaction_{month}.txt 구조의 파일이 있어야함
    train_data, valid_data, test_data = create_subway_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_school_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_place_within_radius(train_data, valid_data, test_data)
    
    
    
    ### feature drop(제거하고 싶은 feature는 drop_columns로 제거됨. contract_day에 원하는 column을 추가)
    train_data_ = preprocessing.drop_columns(train_data, ['contract_day'])
    valid_data_ = preprocessing.drop_columns(valid_data, ['contract_day'])
    test_data_ = preprocessing.drop_columns(test_data, ['contract_day'])

#-----------------------------------------------------------------------------------------------------------------------------------------------#
# 여기 아래서부터 자유롭게 시도(drop columns를 제외하면 대략 52개 column정도 있음)   
    
    ### 정규화
    print("standardization...")
    train_data_, valid_data_, test_data_ = preprocessing.standardization(train_data_, valid_data_, test_data_, scaling_type = 'standard')

    # feature selection
    train_data_scaled, valid_data_scaled, test_data_scaled = preprocessing.feature_selection(train_data_, valid_data_, test_data_)

    # feature split
    X_train, y_train, X_valid, y_valid, X_test = common_utils.split_feature_target(train_data_scaled, valid_data_scaled, test_data_scaled)
    
    # train model
    print("Training the model...")
    model_ = model.xgboost(X_train, y_train)

    prediction, mae = inference(model_, 'validation', X_valid, y_valid)

    # record MAE score as csv
    hyperparams = "learning_rate=0.3, n_estimators=1000, enable_categorical=True, random_state=Config.RANDOM_SEED"
    common_utils.mae_to_csv(name, title, hyperparams=hyperparams, mae = mae)

    # train with total dataset
    print("Training with total dataset...")
    X_total, y_total = common_utils.train_valid_concat(X_train, X_valid, y_train, y_valid)
    model_ = model.xgboost(X_total, y_total)

    # inference with test data
    submission = inference(model_, 'submission', X_test)

    # save sample submission
    common_utils.submission_to_csv(submission, 'test_all_columns+xgboost')

    print("Successfully executed main.py.")
    return prediction, mae

if __name__ == "__main__":
    prediction, mae = main()
    print(mae)
