from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from geopy.distance import great_circle



### 클러스터링

# clustering 함수
def clustering(total_df, info_df, feat_name, n_clusters=20):
    info = info_df[['longitude', 'latitude']].values
    
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=Config.RANDOM_SEED)
    kmeans.fit(info)
    
    clusters = kmeans.predict(total_df[['longitude', 'latitude']].values)
    total_df[feat_name] = pd.DataFrame(clusters, dtype='category')
    return total_df



## create_clustering_target 관련 추가 함수(해당 함수 내에 쓰이는 함수들)

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






# target 위치로 clustering한 뒤 해당 군집의 밀도/cluster 중심까지 거리 함수
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
