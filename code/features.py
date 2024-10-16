from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

def clustering(total_df, info_df, feat_name, n_clusters=20):
    info = info_df[['longitude', 'latitude']].values
    
    kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=Config.RANDOM_SEED)
    kmeans.fit(info)
    
    clusters = kmeans.predict(total_df[['longitude', 'latitude']].values)
    total_df[feat_name] = pd.DataFrame(clusters, dtype='category')
    return total_df

def create_cluster_density(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 클러스터별 밀도 계산 (클러스터별 포인트 수)
    cluster_density = train_data.groupby('cluster').size().reset_index(name='density')

    train_data = train_data.merge(cluster_density, on='cluster', how='left')
    valid_data = valid_data.merge(cluster_density, on='cluster', how='left')
    test_data = test_data.merge(cluster_density, on='cluster', how='left')

    return train_data, valid_data, test_data

def create_cluster_distance_to_centroid(data: pd.DataFrame, centroids) -> pd.DataFrame:
    # 포함되는 군집의 centroid와의 거리 계산
    lat_centroids = np.array([centroids[cluster, 0] for cluster in data['cluster']])
    lon_centroids = np.array([centroids[cluster, 1] for cluster in data['cluster']])
    lat_diff = data['latitude'].values - lat_centroids
    lon_diff = data['longitude'].values - lon_centroids
    data['distance_to_centroid'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    return data

def create_clustering_target(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # K-means 클러스터링
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_SEED)
    train_data['cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
    valid_data['cluster'] = kmeans.predict(valid_data[['latitude', 'longitude']])
    test_data['cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])
    
    train_data['cluster'] = train_data['cluster'].astype('category')
    valid_data['cluster'] = valid_data['cluster'].astype('category')
    test_data['cluster'] = test_data['cluster'].astype('category')

    # 군집 밀도 변수 추가
    train_data, valid_data, test_data = create_cluster_density(train_data, valid_data, test_data)

    centroids = kmeans.cluster_centers_

    # 군집 centroid까지의 거리 변수 추가
    train_data = create_cluster_distance_to_centroid(train_data, centroids)
    valid_data = create_cluster_distance_to_centroid(valid_data, centroids)
    test_data = create_cluster_distance_to_centroid(test_data, centroids)

    return train_data, valid_data, test_data

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

def create_subway_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo에는 지하철 역의 위도와 경도가 포함되어 있다고 가정
    subwayInfo = Directory.subway_info
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    def count_subways_within_radius(data, radius):
        counts = []  # 초기화
        for i in range(0, len(data), 10000):
            batch = data.iloc[i:i+10000]
            house_coords = batch[['latitude', 'longitude']].values
            # KDTree를 사용하여 주어진 반경 내 지하철역 찾기
            indices = tree.query_radius(house_coords, r=radius)  # 반경에 대한 인덱스
            # 각 집의 주변 지하철역 개수 세기
            counts.extend(len(idx) for idx in indices)

        # counts가 데이터프레임 크기보다 작을 경우 0으로 채우기
        if len(counts) < len(data):
            counts.extend([0] * (len(data) - len(counts)))
        
        # 데이터프레임에 결과 추가
        data['subways_within_radius'] = counts
        return data

    # 각 데이터셋에 대해 거리 추가
    radius = 0.01  # 약 1km
    train_data = count_subways_within_radius(train_data, radius)
    valid_data = count_subways_within_radius(valid_data, radius)
    test_data = count_subways_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_nearest_park_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

        data['nearest_park_distance'] = nearest_park_distances
        return data

    # train, valid, test 데이터에 가장 가까운 공원 거리 및 면적 추가
    train_data = add_nearest_park_features(train_data)
    valid_data = add_nearest_park_features(valid_data)
    test_data = add_nearest_park_features(test_data)

    return train_data, valid_data, test_data

def create_school_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]
    school_coords = seoul_area_school[['latitude', 'longitude']].values
    tree = KDTree(school_coords, leaf_size=10)

    def count_schools_within_radius(data, radius):
        counts = []  # 학교 개수를 저장할 리스트 초기화
        for i in range(0, len(data), 10000):  # 10,000개씩 배치로 처리
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = tree.query_radius(house_coords, r=radius)  # 반경 내의 인덱스 찾기
            counts.extend(len(idx) for idx in indices)  # 각 배치의 학교 개수 추가
        data['schools_within_radius'] = counts  # 데이터에 추가
        return data
    
    radius = 0.02 # 약 2km
    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_sum_park_area_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    # 수도권 공원의 좌표 필터링
    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                  (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # 수도권 공원의 좌표로 KDTree 생성
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def sum_park_area_within_radius(data, radius=0.02):
        area_sums = []  # 공원 면적 합을 저장할 리스트 초기화
        for i in range(0, len(data), 10000):  # 10,000개씩 배치로 처리
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = park_tree.query_radius(house_coords, r=radius)  # 반경 내의 인덱스 찾기
            
            # 각 집에 대해 반경 2km 이내의 공원 면적의 합을 계산
            for idx in indices:
                if idx.size > 0:  # 2km 이내에 공원이 있을 경우
                    areas_sum = seoul_area_parks.iloc[idx]['area'].sum()
                else:
                    areas_sum = 0  # 공원이 없는 경우 면적 0
                area_sums.append(areas_sum)  # 면적 합 추가

        # 결과 추가
        data['nearest_park_area_sum'] = area_sums
        return data

    # train, valid, test 데이터에 반경 2km 이내의 공원 면적 합 추가
    train_data = sum_park_area_within_radius(train_data)
    valid_data = sum_park_area_within_radius(valid_data)
    test_data = sum_park_area_within_radius(test_data)

    return train_data, valid_data, test_data

def create_school_counts_within_radius_by_school_level(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                     (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]
    
    # 초, 중, 고등학교의 좌표를 분리
    elementary_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'elementary']
    middle_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'middle']
    high_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'high']

    # 각 학교 유형의 좌표로 KDTree 생성
    elementary_coords = elementary_schools[['latitude', 'longitude']].values
    middle_coords = middle_schools[['latitude', 'longitude']].values
    high_coords = high_schools[['latitude', 'longitude']].values

    tree_elementary = KDTree(elementary_coords, leaf_size=10)
    tree_middle = KDTree(middle_coords, leaf_size=10)
    tree_high = KDTree(high_coords, leaf_size=10)

    def count_schools_within_radius(data, radius):
        counts_elementary = []  # 초등학교 개수를 저장할 리스트 초기화
        counts_middle = []      # 중학교 개수를 저장할 리스트 초기화
        counts_high = []        # 고등학교 개수를 저장할 리스트 초기화

        for i in range(0, len(data), 10000):  # 10,000개씩 배치로 처리
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            
            # 각 학교 유형의 개수 세기
            indices_elementary = tree_elementary.query_radius(house_coords, r=radius)
            indices_middle = tree_middle.query_radius(house_coords, r=radius)
            indices_high = tree_high.query_radius(house_coords, r=radius)
            
            counts_elementary.extend(len(idx) for idx in indices_elementary)  # 각 배치의 초등학교 개수 추가
            counts_middle.extend(len(idx) for idx in indices_middle)        # 각 배치의 중학교 개수 추가
            counts_high.extend(len(idx) for idx in indices_high)            # 각 배치의 고등학교 개수 추가

        # 데이터에 추가
        data['elementary_schools_within_radius'] = counts_elementary
        data['middle_schools_within_radius'] = counts_middle
        data['high_schools_within_radius'] = counts_high
        
        return data

    radius = 0.02  # 약 2km
    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data