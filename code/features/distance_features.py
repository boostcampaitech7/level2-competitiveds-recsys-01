from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KDTree, BallTree

from sklearn.cluster import KMeans

from geopy.distance import great_circle


### 거리

# 가장 가까운 지하철까지의 거리 함수
def create_nearest_subway_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subwayInfo = Directory.subway_info

    # KD-트리 생성
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    # 거리 계산 함수 정의
    def add_nearest_subway_distance(data):
        # 각 집의 좌표 가져오기
        house_coords = data[['latitude', 'longitude']].values
        # 가장 가까운 지하철 역까지의 거리 계산
        distances, indices = tree.query(house_coords, k=1)  # k=1: 가장 가까운 역
        # 거리를 데이터프레임에 추가 (미터 단위로 변환)
        data['nearest_subway_distance'] = distances.flatten()
        return data

    # 각 데이터셋에 대해 거리 추가
    train_data = add_nearest_subway_distance(train_data)
    valid_data = add_nearest_subway_distance(valid_data)
    test_data = add_nearest_subway_distance(test_data)

    return train_data, valid_data, test_data




# 가장 가까운 공원까지의 거리 & 가장 가까운 공원 면적 변수
def create_nearest_park_distance_and_area(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # 수도권 공원의 좌표로 KDTree 생성
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def add_nearest_park_features(data):
        # 각 집의 좌표로 가장 가까운 공원 찾기
        house_coords = data[['latitude', 'longitude']].values
        distances, indices = park_tree.query(house_coords, k=1)  # 가장 가까운 공원 찾기

        # 가장 가까운 공원까지의 거리 및 해당 공원의 면적 추가
        nearest_park_distances = distances.flatten()
        nearest_park_areas = seoul_area_parks.iloc[indices.flatten()]['area'].values  # 면적 정보를 가져옴

        data['nearest_park_distance'] = nearest_park_distances
        data['nearest_park_area'] = nearest_park_areas
        return data

    # train, valid, test 데이터에 가장 가까운 공원 거리 및 면적 추가
    train_data = add_nearest_park_features(train_data)
    valid_data = add_nearest_park_features(valid_data)
    test_data = add_nearest_park_features(test_data)

    return train_data, valid_data, test_data



# level별 가장 가까운 학교까지 거리
def create_nearest_school_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]

    elementary_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'elementary']
    middle_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'middle']
    high_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'high']

    # 각 학교 유형에 대해 BallTree 생성
    elementary_tree = BallTree(np.radians(elementary_schools[['latitude', 'longitude']]), metric='haversine')
    middle_tree = BallTree(np.radians(middle_schools[['latitude', 'longitude']]), metric='haversine')
    high_tree = BallTree(np.radians(high_schools[['latitude', 'longitude']]), metric='haversine')

    # 거리 계산 함수 정의
    def add_nearest_school_distance(data):
        unique_coords = data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        house_coords = np.radians(unique_coords.values)

        # 가장 가까운 학교까지의 거리 계산 (미터 단위로 변환)
        unique_coords['nearest_elementary_distance'] = elementary_tree.query(house_coords, k=1)[0].flatten() * 6371000
        unique_coords['nearest_middle_distance'] = middle_tree.query(house_coords, k=1)[0].flatten() * 6371000
        unique_coords['nearest_high_distance'] = high_tree.query(house_coords, k=1)[0].flatten() * 6371000

        data = data.merge(unique_coords, on=['latitude', 'longitude'], how='left')

        return data

    # 훈련 데이터에 거리 추가
    train_data = add_nearest_school_distance(train_data)
    valid_data = add_nearest_school_distance(valid_data)
    test_data = add_nearest_school_distance(test_data)

    return train_data, valid_data, test_data

# 환승역 가중치 거리 계산 함수
def weighted_subway_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subwayInfo = Directory.subway_info

    # 환승역 가중치 부여
    duplicate_stations = subwayInfo.groupby(['latitude', 'longitude']).size().reset_index(name='counts')
    transfer_stations = duplicate_stations[duplicate_stations['counts'] > 1]

    subwayInfo = subwayInfo.merge(transfer_stations[['latitude', 'longitude', 'counts']], 
                                  on=['latitude', 'longitude'], 
                                  how='left')
    subwayInfo['weight'] = subwayInfo['counts'].fillna(1)  # 환승역은 가중치 > 1, 나머지는 1

    subway_tree = BallTree(np.radians(subwayInfo[['latitude', 'longitude']]), metric='haversine')

    # 거리 계산 함수 정의
    def add_weighted_subway_distance(data):
        unique_coords = data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        house_coords = np.radians(unique_coords.values)

        distances, indices = subway_tree.query(house_coords, k=1)
        unique_coords['nearest_subway_distance'] = distances.flatten() * 6371000 

        weights = subwayInfo.iloc[indices.flatten()]['weight'].values

        # 거리를 가중치로 나누기
        unique_coords['nearest_subway_distance'] /= weights  

        data = data.merge(unique_coords, on=['latitude', 'longitude'], how='left')

        return data

    # 각 데이터셋에 대해 거리 추가
    train_data = add_weighted_subway_distance(train_data)
    valid_data = add_weighted_subway_distance(valid_data)
    test_data = add_weighted_subway_distance(test_data)

    return train_data, valid_data, test_data