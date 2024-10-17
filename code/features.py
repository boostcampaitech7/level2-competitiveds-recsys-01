from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree, BallTree
from geopy.distance import great_circle
from sklearn.preprocessing import StandardScaler


### 금리 shift 함수
def shift_interest_rate_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, month : int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)
    
    total_data = pd.concat([train_data[['date','interest_rate']],valid_data[['date','interest_rate']],test_data[['date','interest_rate']]], axis=0)
    
    # 원래의 인덱스 저장
    total_data['original_index'] = total_data.index
    
    # 데이터 정렬 (date 기준)
    df_sorted = df.sort_values('date').reset_index(drop=True)

    # 과거 금리 정보 구하기
    df_sorted['date_minus_1year'] = df_sorted['date'] - pd.DateOffset(years=1)
    df_sorted['date_minus_6months'] = df_sorted['date'] - pd.DateOffset(months=6)
    df_sorted['date_minus_3months'] = df_sorted['date'] - pd.DateOffset(months=3)

    df_sorted = pd.merge_asof(
        df_sorted, 
        df_sorted[['date', 'interest_rate']], 
        left_on='date_minus_1year', 
        right_on='date', 
        direction='backward', 
        suffixes=('', '_1year')
    )

    df_sorted = pd.merge_asof(
        df_sorted, 
        df_sorted[['date', 'interest_rate']], 
        left_on='date_minus_6months', 
        right_on='date', 
        direction='backward', 
        suffixes=('', '_6months')
    )

    df_sorted = pd.merge_asof(
        df_sorted, 
        df_sorted[['date', 'interest_rate']], 
        left_on='date_minus_3months', 
        right_on='date', 
        direction='backward', 
        suffixes=('', '_3months')
    )
    
    # 필요 없는 중간 날짜 컬럼(drop)
    df_sorted = df_sorted.drop(columns=['date_minus_1year', 'date_1year', 'date_minus_6months', 'date_6months', 'date_minus_3months', 'date_3months'])

    df_sorted['interest_rate_3months'] = df_sorted['interest_rate_3months'].fillna(df_sorted['interest_rate'])
    df_sorted['interest_rate_6months'] = df_sorted['interest_rate_6months'].fillna(df_sorted['interest_rate'])
    df_sorted['interest_rate_1year'] = df_sorted['interest_rate_1year'].fillna(df_sorted['interest_rate'])

    df_final = df_sorted.sort_values('original_index').drop(columns=['original_index']).reset_index(drop=True)

    train_data_ = df_final.iloc[:train_data_length,:]
    valid_data_ = df_final.iloc[train_data_length:train_data_length+valid_data_length,:]
    test_data_ = df_final.iloc[train_data_length+valid_data_length:,:]
        
    return train_data_, valid_data_, test_data_




### n 개월 동일한 아파트 거래량 함수
def transaction_count_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, month : int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)
    
    train_data_tot = pd.concat([train_data, valid_data], axis=0)
    total_data = pd.concat([train_data_tot, test_data], axis=0)
    
    total_data['transaction_count_last_3_months'] = 0
    
    # 위도, 경도, 건축 연도로 그룹화
    grouped = total_data.groupby(['latitude', 'longitude', 'built_year'])
    
    # 각 그룹에 대해 거래량 계산
    for (lat, lon, built_year), group in tqdm(grouped, desc="Calculating previous 3 months transaction counts by location and year"):
    # 그룹 내 거래일 정렬
        group = group.sort_values(by='date')
    
        # 거래량을 저장할 리스트 초기화
        transaction_counts = []

        for idx, row in group.iterrows():
            # 현재 거래일로부터 month 이전 날짜 계산
            end_date = row['date']
            start_date = end_date - pd.DateOffset(months=month)

            # 동일한 아파트에서의 거래량 계산
            transaction_count = group[
                (group['date'] < end_date) &  # 현재 거래일 이전
                (group['date'] >= start_date)  # month 이전
                ].shape[0]

            # 거래량 리스트에 추가
            transaction_counts.append(transaction_count)

        # 배치 결과를 데이터프레임에 저장
        total_data.loc[group.index, 'transaction_count_last_3_months'] = transaction_counts

    train_data_ = total_data.iloc[:train_data_length,:]
    valid_data_ = total_data.iloc[train_data_length:train_data_length+valid_data_length,:]
    test_data_ = total_data.iloc[train_data_length+valid_data_length:,:]
        
    return train_data_, valid_data_, test_data_



### 클러스터링

def clustering(total_df, info_df, feat_name, n_clusters=20):
    info = info_df[['longitude', 'latitude']].values
    
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=Config.RANDOM_SEED)
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

# 반경 내 지하철 개수 함수
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
    radius = 0.01  # ?�� 1km
    train_data = count_subways_within_radius(train_data, radius)
    valid_data = count_subways_within_radius(valid_data, radius)
    test_data = count_subways_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_nearest_park_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
# 가장 가까운 공원 거리 및 면적 함수
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
        nearest_park_areas = seoul_area_parks.iloc[indices.flatten()]['area'].values  # 면적 정보를 가져옴

        data['nearest_park_distance'] = nearest_park_distances
        data['nearest_park_area'] = nearest_park_areas
        return data

    # train, valid, test 데이터에 가장 가까운 공원 거리 및 면적 추가
    train_data = add_nearest_park_features(train_data)
    valid_data = add_nearest_park_features(valid_data)
    test_data = add_nearest_park_features(test_data)

    return train_data, valid_data, test_data

# 반경 내 학교 개수 함수
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
    radius = 0.01 # 약 1km
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

    
def distance_gangnam(df):
    gangnam = (37.498095, 127.028361548)

    def calculate_distance(df):
        point = (df['latitude'], df['longitude'])
        distance_km = great_circle(gangnam, point).kilometers
        return distance_km
    
    df['distance_km'] = df.apply(calculate_distance, axis=1)
    df['gangnam_5km'] = (df['distance_km'] <= 5).astype(int)
    df['gangnam_10km'] = (df['distance_km'] <= 10).astype(int)
    df['gangnam_remote'] = (df['distance_km'] > 10).astype(int)
    df.drop(columns=['distance_km'], inplace=True)

    return df


def create_temporal_feature(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame)-> pd.DataFrame:
    def combination_temporal_feature(df):
        df_preprocessed = df.copy()
        
        df_preprocessed['year'] = df_preprocessed['contract_year_month'].astype(str).str[:4].astype(int)
        df_preprocessed['month'] = df_preprocessed['contract_year_month'].astype(str).str[4:].astype(int)
        df_preprocessed['date'] = pd.to_datetime(df_preprocessed['year'].astype(str) + df_preprocessed['month'].astype(str).str.zfill(2) + df_preprocessed['contract_day'].astype(str).str.zfill(2))
        # 기본 특성 생성 (모든 데이터셋에 동일하게 적용 가능)
        df_preprocessed['day_of_week'] = df_preprocessed['date'].dt.dayofweek
        #df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df_preprocessed['quarter'] = df_preprocessed['date'].dt.quarter
        df_preprocessed['is_month_end'] = (df_preprocessed['date'].dt.is_month_end).astype(int)
        df_preprocessed['season'] = df_preprocessed['month'].map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 
                                        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
                                        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})
        return df_preprocessed
    train_data = combination_temporal_feature(train_data)
    valid_data = combination_temporal_feature(valid_data)
    test_data = combination_temporal_feature(test_data)

    return train_data, valid_data, test_data



def create_sin_cos_season(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame)-> pd.DataFrame:
    def combination_sin_cos_season(df):
        df_preprocessed = df.copy()
        # Cyclical encoding for seasons
        season_dict = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
        df_preprocessed['season_numeric'] = df_preprocessed['season'].map(season_dict)
        df_preprocessed['season_sin'] = np.sin(2 * np.pi * df_preprocessed['season_numeric'] / 4)
        df_preprocessed['season_cos'] = np.cos(2 * np.pi * df_preprocessed['season_numeric'] / 4)
        df_preprocessed = df_preprocessed.drop(['season_numeric'], axis=1)
        return df_preprocessed
    
    train_data = combination_sin_cos_season(train_data)
    valid_data = combination_sin_cos_season(valid_data)
    test_data = combination_sin_cos_season(test_data)

    return train_data, valid_data, test_data


def create_floor_area_interaction(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    def floor_weighted(df):
        df_preprocessed = df.copy()
        df_preprocessed['floor_weighted'] = df_preprocessed['floor'].apply(lambda x: x if x >= 0 else x * -0.5)
        df_preprocessed['floor_area_interaction'] = (
            df_preprocessed['floor_weighted'] * df_preprocessed['area_m2']
        )
        df_preprocessed.drop(['floor_weighted'], axis = 1, inplace = True)
        return df_preprocessed
    
    train_data = floor_weighted(train_data)
    valid_data = floor_weighted(valid_data)
    test_data = floor_weighted(test_data)

    return train_data, valid_data, test_data


def assign_info_cluster(train_data, school_info, park_info, subway_info):
    min_latitude = min(train_data['latitude'])
    max_latitude = max(train_data['latitude'])

    min_longitude = min(train_data['longitude'])
    max_longitude = max(train_data['longitude'])

    school_info_filtered = school_info[(school_info['latitude'] >= min_latitude) & (school_info['latitude'] <= max_latitude) & (school_info['longitude'] >= min_longitude) & (school_info['longitude'] <= max_longitude)]
    park_info_filtered = park_info[(park_info['latitude'] >= min_latitude) & (park_info['latitude'] <= max_latitude) & (park_info['longitude'] >= min_longitude) & (park_info['longitude'] <= max_longitude)]
    subway_info_filtered = subway_info[(subway_info['latitude'] >= min_latitude) & (subway_info['latitude'] <= max_latitude) & (subway_info['longitude'] >= min_longitude) & (subway_info['longitude'] <= max_longitude)]

    # train_data로 클러스터 형성
    X_train = train_data[['latitude', 'longitude']].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=25, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)

    # 다른 데이터셋에 클러스터 할당
    def assign_cluster(data):
        X = data[['latitude', 'longitude']].values
        X_scaled = scaler.transform(X)
        return kmeans.predict(X_scaled)

    train_data['cluster'] = kmeans.labels_
    school_info_filtered['cluster'] = assign_cluster(school_info_filtered)
    subway_info_filtered['cluster'] = assign_cluster(subway_info_filtered)
    park_info_filtered['cluster'] = assign_cluster(park_info_filtered)

    return train_data, school_info_filtered, park_info_filtered, subway_info_filtered

def cluster_count(park_info_filtered, school_info_filtered, subway_info_filtered, df):
    lst = [park_info_filtered, school_info_filtered, subway_info_filtered]
    for data in lst:
        tmp = pd.DataFrame(data['cluster'].value_counts())
        tmp['cluster'] = tmp.index
        tmp.index.name = None
        df = df.merge(tmp, on = 'cluster', how = 'left')
    df.rename(columns = {'count_x' : 'park_cluster_count', 'count_y' : 'school_cluster_count', 'count' : 'subway_cluster_count'}, inplace = True)
    return df


def treat_categorical_cols(df):
    df_new = df.copy()
    base_year = df_new['built_year'].min()
    df_new['new_built_year'] = df_new['built_year'] - base_year
    
    #df_new['year'] = (df_new['year'] // 10) * 10

    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)

    # 3. quarter 처리: 순환 인코딩
    df_new['quarter_sin'] = np.sin(2 * np.pi * df_new['quarter'] / 4)
    df_new['quarter_cos'] = np.cos(2 * np.pi * df_new['quarter'] / 4)

    tmp = pd.get_dummies(df_new['contract_type'], prefix='contract_type').astype(int)
    df_new = pd.concat([df_new, tmp], axis = 1)

    #df_new = df_new.drop(['year', 'built_year', 'month', 'quarter', 'contract_type'], axis = 1)
    return df_new


# 반경 내 공공시설(학교, 지하철, 공원) 개수 함수
def create_place_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo에는 지하철 역의 위도와 경도가 포함되어 있다고 가정
    subway_data = Directory.subway_info
    park_data = Directory.park_info
    school_data = Directory.school_info

    
    # seoul park 측정
    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                    (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]
    # seoul school 측정
    seoul_area_school = school_data[(school_data['latitude'] >= 37.0) & (school_data['latitude'] <= 38.0) &
                                    (school_data['longitude'] >= 126.0) & (school_data['longitude'] <= 128.0)]
    # seoul subway 측정
    seoul_area_subway = subway_data[(subway_data['latitude'] >= 37.0) & (subway_data['latitude'] <= 38.0) &
                                (subway_data['longitude'] >= 126.0) & (subway_data['longitude'] <= 128.0)]
    
    
    
    subway_coords = seoul_area_subway[['latitude', 'longitude']].values
    subway_tree = KDTree(subway_coords, leaf_size=10)
    school_coords = seoul_area_school[['latitude', 'longitude']].values
    school_tree = KDTree(subway_coords, leaf_size=10)
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(subway_coords, leaf_size=10)


    # count 함수
    def count_within_radius(data, radius, tree):
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
        
            return counts
    
    # 각 데이터셋에 대해 거리 추가
    radius = 0.01  # 약 1km

    # 각 데이터셋에 대해 count 계산
    train_subway_counts = count_within_radius(train_data, radius, subway_tree)
    train_school_counts = count_within_radius(train_data, radius, school_tree)
    train_park_counts = count_within_radius(train_data, radius, park_tree)

    valid_subway_counts = count_within_radius(valid_data, radius, subway_tree)
    valid_school_counts = count_within_radius(valid_data, radius, school_tree)
    valid_park_counts = count_within_radius(valid_data, radius, park_tree)

    test_subway_counts = count_within_radius(test_data, radius, subway_tree)
    test_school_counts = count_within_radius(test_data, radius, school_tree)
    test_park_counts = count_within_radius(test_data, radius, park_tree)

    # 각 데이터셋의 공공시설 총 카운트 계산
    train_counts = [subway + school + park for subway, school, park in zip(train_subway_counts, train_school_counts, train_park_counts)]
    valid_counts = [subway + school + park for subway, school, park in zip(valid_subway_counts, valid_school_counts, valid_park_counts)]
    test_counts = [subway + school + park for subway, school, park in zip(test_subway_counts, test_school_counts, test_park_counts)]

    # 각 데이터셋에 카운트를 추가
    train_data['public_facility_count'] = train_counts
    valid_data['public_facility_count'] = valid_counts
    test_data['public_facility_count'] = test_counts
    
    return train_data, valid_data, test_data




### 범주화

def categorization(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, category: str = None, drop: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if category == 'age':
        train_data['nage_category'] = train_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        valid_data['nage_category'] = valid_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        test_data['nage_category'] = test_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        
        if drop:
            train_data.drop(columns=['age'], inplace=True)
            valid_data.drop(columns=['age'], inplace=True)
            test_data.drop(columns=['age'], inplace=True)

    elif category == 'floor':
        train_data['floor_category'] = train_data['floor'].apply(lambda x: 'Other' if x <= 0 or x >= 30 else ('25~30' if 25 <= x <= 30 else x)).astype('category')
        valid_data['floor_category'] = valid_data['floor'].apply(lambda x: 'Other' if x <= 0 or x >= 30 else ('25~30' if 25 <= x <= 30 else x)).astype('category')
        test_data['floor_category'] = test_data['floor'].apply(lambda x: 'Other' if x <= 0 or x >= 30 else ('25~30' if 25 <= x <= 30 else x)).astype('category')

        if drop:
            train_data.drop(columns=['floor'], inplace=True)
            valid_data.drop(columns=['floor'], inplace=True)
            test_data.drop(columns=['floor'], inplace=True)
            
    elif category == 'area_m2':
        train_data['area_category'] = train_data['area_m2'].apply(lambda x: '60 under' if x < 60 else ('60~85' if 60 <= x <= 85 else '85 over')).astype('category')
        valid_data['area_category'] =valid_data['area_m2'].apply(lambda x: '60 under' if x < 60 else ('60~85' if 60 <= x <= 85 else '85 over')).astype('category')
        test_data['area_category'] = test_data['area_m2'].apply(lambda x: '60 under' if x < 60 else ('60~85' if 60 <= x <= 85 else '85 over')).astype('category')

        if drop:
            train_data.drop(columns=['area_m2'], inplace=True)
            valid_data.drop(columns=['area_m2'], inplace=True)
            test_data.drop(columns=['area_m2'], inplace=True)
            
    return train_data, valid_data, test_data

def creat_area_m2_category(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def categorize_area(x):
        range_start = (x // 50) * 50
        range_end = range_start + 49
        return f"{range_start} - {range_end}"

    for dataset in [train_data, valid_data, test_data]:
        area_dummies = pd.get_dummies(dataset['area_m2'].apply(categorize_area), prefix='area',drop_first=True)
        dataset = pd.concat([dataset, area_dummies], axis=1)

    return train_data, valid_data, test_data

#level별 가장 가까운 학교까지 거리
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
