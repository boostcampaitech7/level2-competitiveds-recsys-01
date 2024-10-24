import pandas as pd
import numpy as np
import os
import sys

from utils.constant_utils import Config, Directory
from utils.common_utils import *

from sklearn.neighbors import KDTree, BallTree
from sklearn.cluster import KMeans

from geopy.distance import great_circle

main_directory = os.path.abspath('..')
sys.path.append(main_directory)

from features import clustering_features, count_features, deposit_features, distance_features, other_features

def feature_engineering(train_data_ : pd.DataFrame , valid_data_ : pd.DataFrame , test_data_ : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    feature engineering을 가동시키는 함수입니다.
    
    features
        ㄴclustering features
        ㄴdistance features
        ㄴother features
        ㄴcount features
        ㄴdeposit features
    
    각각의 해당 feature들을 생성한 최종 dataset이 출력됩니다. 
    
    
    '''
    ### clustering_features
    print("create clustering features")
    train_data, valid_data, test_data = clustering_features.create_clustering_target(train_data_, valid_data_, test_data_)
    train_data, valid_data, test_data = clustering_features.create_cluster_deposit_median(train_data, valid_data, test_data)


    ### distance_features
    print("create distance features")
    train_data, valid_data, test_data = distance_features.create_nearest_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = distance_features.create_nearest_park_distance_and_area(train_data, valid_data, test_data)
    train_data, valid_data, test_data = distance_features.create_nearest_school_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = distance_features.weighted_subway_distance(train_data, valid_data, test_data)


    # other_features
    print("create other features")
    train_data, valid_data, test_data = other_features.create_temporal_feature(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.create_sin_cos_season(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.create_floor_area_interaction(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.distance_gangnam(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.create_sum_park_area_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.shift_interest_rate_function(train_data, valid_data, test_data)
    train_data, valid_data, test_data = other_features.categorization(train_data, valid_data, test_data, category = 'age')
    train_data, valid_data, test_data = other_features.categorization(train_data, valid_data, test_data, category = 'floor')
    train_data, valid_data, test_data = other_features.categorization(train_data, valid_data, test_data, category = 'area_m2')

    
    # count_features
    print("create count features")
    train_data, valid_data, test_data = count_features.transaction_count_function(train_data, valid_data, test_data)
    # 위의 함수를 바로 실행하기 위한 구조 : data/transaction_data에 train/valid/test_transaction_{month}.txt 구조의 파일이 있어야함
    train_data, valid_data, test_data = count_features.create_subway_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = count_features.create_school_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = count_features.create_school_counts_within_radius_by_school_level(train_data, valid_data, test_data)
    train_data, valid_data, test_data = count_features.create_place_within_radius(train_data, valid_data, test_data)
    
    
    # deposit_features
    print("create deposit features")
    train_data, valid_data = deposit_features.add_recent_rent_in_building(train_data, valid_data, type = 'valid')
    train_data, valid_data = deposit_features.add_avg_rent_in_past_year(train_data, valid_data, type = 'valid')
    train_data, valid_data = deposit_features.add_rent_growth_rate(train_data, valid_data, type = 'valid')

    
    return train_data, valid_data, test_data