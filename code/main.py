import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.constant_utils import Directory, Config
from utils import common_utils

from features import clustering_features

from models.SpatialWeightMatrix import SpatialWeightMatrix
from models.XGBoostWithSpatialWeight import XGBoostWithSpatialWeight
import models.SeedEnsemble as se

from Handler import feature_engineering as fe
from Handler import preprocessing as pre

from models.inference import *




def main():
    print("Start the main.py successfully!")

    '''
    name : 실험자 이름입니다.
    title : result 폴더에 저장될 실험명(feature + model + params)을 지정합니다.
    '''
    name = 'eun'
    title = 'final_seed(10)_ensemble_weight_matrix(k=10)_and_xgboost_test_and_optuna'


    ### data load
    print("total data load ...")
    df = common_utils.merge_data(Directory.train_data, Directory.test_data)



    ### 클러스터 피처 apply
    print("clustering apply ...")
    for info_df_name in ['subway_info', 'school_info', 'park_info']:
        info_df = getattr(Directory, info_df_name)  
        df = clustering_features.clustering(df, info_df, feat_name=info_df_name, n_clusters=20)



    ### 이상치 처리
    print("start to cleaning outliers...")
    df = pre.handle_age_outliers(df)



    ### 데이터 분할
    print("train, valid, test split for preprocessing & feature engineering ...")
    train_data_, valid_data_, test_data_ = common_utils.train_valid_test_split(df)



    ### 데이터 전처리
    print("start to preprocessing...")   
    # type 카테고리화
    train_data_ = pre.numeric_to_categoric(train_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
    valid_data_ = pre.numeric_to_categoric(valid_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})
    test_data_ = pre.numeric_to_categoric(test_data_, 'contract_type', {0:'new', 1:'renew', 2:'unknown'})

    # 중복 제거
    train_data_ = pre.handle_duplicates(train_data_)
    valid_data_ = pre.handle_duplicates(valid_data_)
    
    # 로그 변환
    # train_data_ = pre.log_transform(train_data_, 'deposit')
    # valid_data_ = pre.log_transform(valid_data_, 'deposit')



    ### 피처 엔지니어링
    print("start to feature engineering...")
    # valid를 위한 feature engineering(deposit features에서 test argument = 'valid'를 받음)
    # test를 위할 땐 추가로 total df에 대해 진행해야함(type = 'test')
    train_data_, valid_data_, test_data_ = fe.feature_engineering(train_data_, valid_data_, test_data_)
    
    # feature selection(필요시)
    # train_data_, valid_data_, test_data_ = preprocessing.feature_selection(train_data_, valid_data_, test_data_)



    ### feature drop
    '''
    제거하고 싶은 feature는 drop_columns로 제거 가능
    
    ex)
    train_data_ = preprocessing.drop_columns(train_data, ['contract_day'])
    valid_data_ = preprocessing.drop_columns(valid_data, ['contract_day'])
    test_data_ = preprocessing.drop_columns(test_data, ['contract_day'])
    
    '''



    ### 최종 dataset 구축(top_20_features)
    selected_columns = Config.TOP_20_FEATURES
     
    train_data_ = train_data_[selected_columns]
    valid_data_ = valid_data_[selected_columns]
    test_data_ = test_data_[selected_columns]

    
    
    ### 정규화
    print("standardization...")
    train_data_scaled, valid_data_scaled, test_data_scaled = pre.standardization(train_data_, valid_data_, test_data_, scaling_type = 'standard')


    print(train_data_scaled.shape, valid_data_scaled.shape, test_data_scaled.shape)
    print(train_data_scaled.head(3))
    
    
    
    ### training
    print("Training the model...")

    # 가중치 행렬 생성
    spatial_weight_matrix = SpatialWeightMatrix()
    spatial_weight_matrix.generate_weight_matrices(train_data_scaled, train_data_scaled, dataset_type='train')
    spatial_weight_matrix.generate_weight_matrices(valid_data_scaled, train_data_scaled, dataset_type='valid')

    # seed ensemble
    seed_ensemble = se(model_class=XGBoostWithSpatialWeight, spatial_weight_matrix=spatial_weight_matrix)
    seed_ensemble.train(train_data_scaled, dataset_type='train')
    
    final_test_preds = seed_ensemble.evaluate(valid_data_scaled, train_data_scaled)
    mae = mean_absolute_error(valid_data_scaled['deposit'], final_test_preds)
    print(f"total MAE on validation data: {mae}")

    # train with total dataset
    print("Training with total dataset...")
    total_train_data = pd.concat([train_data_scaled, valid_data_scaled])
    spatial_weight_matrix.generate_weight_matrices(total_train_data, total_train_data, dataset_type='train_total')
    spatial_weight_matrix.generate_weight_matrices(test_data_scaled, total_train_data, dataset_type='test')

    seed_ensemble = se(model_class=XGBoostWithSpatialWeight, spatial_weight_matrix=spatial_weight_matrix)
    seed_ensemble.train(total_train_data, dataset_type='train_total')
    
    
    
    ### inference
    final_test_preds = seed_ensemble.inference(test_data_scaled, total_train_data)

    sample_submission = Directory.sample_submission
    sample_submission['deposit'] = final_test_preds



    ### save submission
    common_utils.submission_to_csv(sample_submission, title)

    print("Successfully executed main.py.")

if __name__ == "__main__":
    main()