from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from geopy.distance import great_circle


# 금리 shift 함수
def shift_interest_rate_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, years: list[int] = [1], months: list[int] = [3, 6]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)
    
    train_df = train_data.copy()
    valid_df = valid_data.copy()
    test_df = test_data.copy()

    total_data = pd.concat([train_data[['date', 'interest_rate']], 
                            valid_data[['date', 'interest_rate']], 
                            test_data[['date', 'interest_rate']]], axis=0)

    total_data['original_index'] = total_data.index

    # 날짜 기준으로 정렬
    df_sorted = total_data.sort_values('date').reset_index(drop=True)

    # 과거 shifting for문
    for year in years:
        df_sorted[f'date_minus_{year}year'] = df_sorted['date'] - pd.DateOffset(years=year)
        df_sorted = pd.merge_asof(
            df_sorted, 
            df_sorted[['date', 'interest_rate']], 
            left_on=f'date_minus_{year}year', 
            right_on='date', 
            direction='backward', 
            suffixes=('', f'_{year}year')
        )
        df_sorted[f'interest_rate_{year}year'] = df_sorted[f'interest_rate_{year}year'].fillna(df_sorted['interest_rate'])
    
    for month in months:
        df_sorted[f'date_minus_{month}months'] = df_sorted['date'] - pd.DateOffset(months=month)
        df_sorted = pd.merge_asof(
            df_sorted, 
            df_sorted[['date', 'interest_rate']], 
            left_on=f'date_minus_{month}months', 
            right_on='date', 
            direction='backward', 
            suffixes=('', f'_{month}months')
        )
        df_sorted[f'interest_rate_{month}months'] = df_sorted[f'interest_rate_{month}months'].fillna(df_sorted['interest_rate'])

    drop_columns = [col for col in df_sorted.columns if 'date_minus' in col or 'date_' in col and 'interest_rate' not in col]
    df_sorted = df_sorted.drop(columns=drop_columns)

    # 원래 순서로 정렬 후 데이터 분할
    df_final = df_sorted.sort_values('original_index').drop(columns=['original_index']).reset_index(drop=True)

    # train, valid, test split
    train_data_ = df_final.iloc[:train_data_length, :]
    valid_data_ = df_final.iloc[train_data_length:train_data_length + valid_data_length, :]
    test_data_ = df_final.iloc[train_data_length + valid_data_length:, :]
    
    train_data_.index = train_df.index
    valid_data_.index = valid_df.index
    test_data_.index = test_df.index

    train_data_shift = pd.concat([train_df, train_data_.iloc[:, 2:]], axis=1)
    valid_data_shift = pd.concat([valid_df, valid_data_.iloc[:, 2:]], axis=1)
    test_data_shift = pd.concat([test_df, test_data_.iloc[:, 2:]], axis=1)

    return train_data_shift, valid_data_shift, test_data_shift




# 반경 이내 공원 면적의 합 변수
def create_sum_park_area_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, radius : float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    # 수도권 공원의 좌표 필터링
    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                  (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # 수도권 공원의 좌표로 KDTree 생성
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def sum_park_area_within_radius(data, radius: float = 0.02):
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



# year, month, date 조작 변수
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


# 계절 mapping 후 sine, cosine 적용 함수
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

# 층수와 면적의 관계 변수
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




####### categorical columns

# 강남까지의 거리 범주화
def distance_gangnam(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    gangnam = (37.498095, 127.028361548)

    def calculate_distance(df):
        point = (df['latitude'], df['longitude'])
        distance_km = great_circle(gangnam, point).kilometers
        return distance_km
    
    for df in [train_data, valid_data, test_data]:
        df['distance_km'] = df.apply(calculate_distance, axis=1)
        df['distance_category'] = df['distance_km'].apply(lambda x: '5km 이내' if x <= 5 else ('5~10km 이내' if 5 < x <= 10 else '10km 초과')).astype('category')
        # df.drop(columns=['distance_km'], inplace=True)
    return train_data, valid_data, test_data


# floor, age, area 범주 함수
def categorization(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, category: str = None, drop: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if category == 'age':
        train_data['age_category'] = train_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        valid_data['age_category'] = valid_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        test_data['age_category'] = test_data['age'].apply(lambda x: 'Other' if x >= 50 else x).astype('category')
        
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


# area category 함수
def creat_area_m2_category(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def categorize_area(x):
        range_start = (x // 50) * 50
        range_end = range_start + 49
        return f"{range_start} - {range_end}"

    for dataset in [train_data, valid_data, test_data]:
        area_dummies = pd.get_dummies(dataset['area_m2'].apply(categorize_area), prefix='area',drop_first=True)
        dataset = pd.concat([dataset, area_dummies], axis=1)

    return train_data, valid_data, test_data







########### 기타

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

def apt_recent_transaction(df):
    # key : apt_idx, value : 최근 거래 3회의 평균값
    search_complete = {}

    tmp_df = df[df['type']=='train'].sort_values(by=['apt_idx', 'contract_year_month'])
    def search_recent_transaction(row):
        apt_idx = row['apt_idx']
        if apt_idx in search_complete.keys():
            return search_complete[apt_idx]
        recent_transaction = tmp_df[tmp_df['apt_idx']==apt_idx]['deposit'][-5:]
        recent_transaction = recent_transaction.mean()
        search_complete[apt_idx] = recent_transaction
        return recent_transaction

    df['recent_transaction'] = df.apply(search_recent_transaction, axis=1)
    df['recent_transaction'].fillna(df['recent_transaction'].median())
    return df