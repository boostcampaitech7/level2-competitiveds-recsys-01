from utils.constant_utils import Directory
from utils import common_utils

import preprocessing
from features.clustering_features import *
from features.count_features import *
from features.distance_features import *
from features.other_features import *
from models.SpatialWeightMatrix import SpatialWeightMatrix
from models.XGBoostWithSpatialWeight import XGBoostWithSpatialWeight
from sklearn.metrics import mean_absolute_error
from models.SeedEnsemble import SeedEnsemble
import warnings
warnings.filterwarnings('ignore')



def main():
    print("Start the main.py successfully!")

    '''
    name : 실험자 이름입니다.
    title : result 폴더에 저장될 실험명을 지정합니다.
    '''
    name = 'eun'
    title = 'final_seed(10)_ensemble_weight_matrix(k=10)_and_xgboost_test_and_optuna'

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

    ### 피처 엔지니어링
    print("start to feature engineering...")

    # clustering_feature
    print("create clustering features")
    train_data, valid_data, test_data = create_clustering_target(train_data_, valid_data_, test_data_)

    # distance_features
    print("create distance features")
    train_data, valid_data, test_data = distance_gangnam(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_nearest_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = weighted_subway_distance(train_data, valid_data, test_data)

    # other_features
    print("create other features")
    train_data, valid_data, test_data = create_floor_area_interaction(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_sum_park_area_within_radius(train_data, valid_data, test_data)

    
    # count_features
    print("create count features")
    train_data, valid_data, test_data = create_subway_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_school_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)
    
    ### feature select(feature importance 상위 20개)
    selected_columns = [
        'distance_km',
        'floor_area_interaction',
        'high_schools_within_radius',
        'subways_within_radius',
        'built_year',
        'subway_info',
        'longitude',
        'nearest_subway_distance_x',
        'area_m2',
        'middle_schools_within_radius',
        'schools_within_radius',
        'nearest_subway_distance_y',
        'cluster',
        'contract_type',
        'distance_to_centroid',
        'distance_category',
        'contract_year_month',
        'latitude',
        'nearest_park_area_sum',
        'elementary_schools_within_radius',
        'deposit'
    ]
    train_data_ = train_data[selected_columns]
    valid_data_ = valid_data[selected_columns]
    test_data_ = test_data[selected_columns]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
# 여기 아래서부터 자유롭게 시도(drop columns를 제외하면 대략 52개 column정도 있음)   

    ### 정규화
    print("standardization...")
    train_data, valid_data, test_data = preprocessing.standardization(train_data_, valid_data_, test_data_, scaling_type = 'standard')

    # train model
    print("Training the model...")

    # 가중치 행렬 생성
    spatial_weight_matrix = SpatialWeightMatrix()
    spatial_weight_matrix.generate_weight_matrices(train_data, train_data, dataset_type='train')
    spatial_weight_matrix.generate_weight_matrices(valid_data, train_data, dataset_type='valid')

    seed_ensemble = SeedEnsemble(model_class=XGBoostWithSpatialWeight, spatial_weight_matrix=spatial_weight_matrix)
    seed_ensemble.train(train_data, dataset_type='train')
    
    final_test_preds = seed_ensemble.evaluate(valid_data, train_data)
    mae = mean_absolute_error(valid_data['deposit'], final_test_preds)
    print(f"total MAE on validation data: {mae}")

    # train with total dataset
    print("Training with total dataset...")
    total_train_data = pd.concat([train_data, valid_data])
    spatial_weight_matrix.generate_weight_matrices(total_train_data, total_train_data, dataset_type='train_total')
    spatial_weight_matrix.generate_weight_matrices(test_data, total_train_data, dataset_type='test')

    seed_ensemble = SeedEnsemble(model_class=XGBoostWithSpatialWeight, spatial_weight_matrix=spatial_weight_matrix)
    seed_ensemble.train(total_train_data, dataset_type='train_total')
    final_test_preds = seed_ensemble.inference(test_data, total_train_data)

    sample_submission = Directory.sample_submission
    sample_submission['deposit'] = final_test_preds

    # save sample submission
    common_utils.submission_to_csv(sample_submission, title)

    print("Successfully executed main.py.")

if __name__ == "__main__":
    main()