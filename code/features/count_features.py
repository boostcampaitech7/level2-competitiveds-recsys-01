from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

#from geopy.distance import great_circle



# n 개월 동일한 아파트 거래량 함수
def transaction_count_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 파일 경로 설정
    transaction_folder = os.path.join(Directory.root_path, 'level2-competitiveds-recsys-01/data/transaction_data')
    if not os.path.exists(transaction_folder):
        os.makedirs(transaction_folder)

    # 파일 이름에 개월 수를 포함
    train_file = f'{transaction_folder}/train_transaction_{months}.txt'
    valid_file = f'{transaction_folder}/valid_transaction_{months}.txt'
    test_file = f'{transaction_folder}/test_transaction_{months}.txt'

    # 이미 저장된 파일이 있으면 파일을 불러오기
    if os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file):
        print(f"Loading pre-calculated transaction data for {months} months.")
        train_transactions = pd.read_csv(train_file, header=None).squeeze().tolist()
        valid_transactions = pd.read_csv(valid_file, header=None).squeeze().tolist()
        test_transactions = pd.read_csv(test_file, header=None).squeeze().tolist()

        # 기존 데이터에 추가
        train_data[f'transaction_count_last_{months}_months'] = train_transactions
        valid_data[f'transaction_count_last_{months}_months'] = valid_transactions
        test_data[f'transaction_count_last_{months}_months'] = test_transactions

        return train_data, valid_data, test_data

    print(f"Calculating transaction counts for the last {months} months...")

    # 데이터 길이 계산
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)

    # 전체 데이터를 하나로 합치기
    train_data_tot = pd.concat([train_data, valid_data], axis=0)
    total_data = pd.concat([train_data_tot, test_data], axis=0)
    
    # 거래량 계산을 위한 열 초기화
    total_data[f'transaction_count_last_{months}_months'] = 0

    # 위도, 경도, 건축 연도로 그룹화
    grouped = total_data.groupby(['latitude', 'longitude', 'built_year'])

    # 각 그룹에 대해 거래량 계산
    for (lat, lon, built_year), group in tqdm(grouped, desc=f"Calculating previous {months} months transaction counts by location and year"):
        # 그룹 내 거래일 정렬
        group = group.sort_values(by='date')
    
        # 거래량을 저장할 리스트 초기화
        transaction_counts = []

        for idx, row in group.iterrows():
            # 현재 거래일로부터 months 이전 날짜 계산
            end_date = row['date']
            start_date = end_date - pd.DateOffset(months=months)

            # 동일한 아파트에서의 거래량 계산
            transaction_count = group[
                (group['date'] < end_date) &  # 현재 거래일 이전
                (group['date'] >= start_date)  # months 이전
                ].shape[0]

            # 거래량 리스트에 추가
            transaction_counts.append(transaction_count)

        # 배치 결과를 데이터프레임에 저장
        total_data.loc[group.index, f'transaction_count_last_{months}_months'] = transaction_counts

    # 결과 저장 (리스트 형태로)
    train_transactions = total_data[f'transaction_count_last_{months}_months'][:train_data_length].tolist()
    valid_transactions = total_data[f'transaction_count_last_{months}_months'][train_data_length:train_data_length + valid_data_length].tolist()
    test_transactions = total_data[f'transaction_count_last_{months}_months'][train_data_length + valid_data_length:].tolist()

    # 데이터를 txt 파일로 저장
    with open(train_file, 'w') as f:
        for item in train_transactions:
            f.write("%s\n" % item)

    with open(valid_file, 'w') as f:
        for item in valid_transactions:
            f.write("%s\n" % item)

    with open(test_file, 'w') as f:
        for item in test_transactions:
            f.write("%s\n" % item)

    # 기존 데이터에 추가
    train_data[f'transaction_count_last_{months}_months'] = train_transactions
    valid_data[f'transaction_count_last_{months}_months'] = valid_transactions
    test_data[f'transaction_count_last_{months}_months'] = test_transactions

    return train_data, valid_data, test_data



# 반경 내 지하철 개수 함수
def create_subway_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, radius : float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo에는 지하철 역의 위도와 경도가 포함되어 있다고 가정
    subwayInfo = Directory.subway_info
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    def count_subways_within_radius(data, radius: float = 0.01):
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
    train_data = count_subways_within_radius(train_data, radius)
    valid_data = count_subways_within_radius(valid_data, radius)
    test_data = count_subways_within_radius(test_data, radius)

    return train_data, valid_data, test_data



# 반경 내 학교 개수 함수
def create_school_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, radius : float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]
    school_coords = seoul_area_school[['latitude', 'longitude']].values
    tree = KDTree(school_coords, leaf_size=10)

    def count_schools_within_radius(data, radius: float = 0.02):
        counts = []  # 학교 개수를 저장할 리스트 초기화
        for i in range(0, len(data), 10000):  # 10,000개씩 배치로 처리
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = tree.query_radius(house_coords, r=radius)  # 반경 내의 인덱스 찾기
            counts.extend(len(idx) for idx in indices)  # 각 배치의 학교 개수 추가
        data['schools_within_radius'] = counts  # 데이터에 추가
        return data

    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data




# 반경 이내 초,중.고 각각의 개수
def create_school_counts_within_radius_by_school_level(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, radius : float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    def count_schools_within_radius(data, radius: float = 0.02):
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

    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data



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