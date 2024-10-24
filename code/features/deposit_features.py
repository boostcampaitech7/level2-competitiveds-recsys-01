from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
from utils.common_utils import *
import pandas as pd
import numpy as np
import os


# 동일한 아파트(위도, 경도, 건축연도, 면적)의 최근 전세가
def add_recent_rent_in_building(train_data: pd.DataFrame, test_data: pd.DataFrame, type: str = 'valid') -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if type == 'valid':
        df_train_new = train_data.copy()
        df_valid_new = test_data.copy()

        df_train_sorted = df_train_new.sort_values(by=['latitude', 'longitude', 'built_year', 'area_m2', 'date'])

        df_train_sorted['recent_rent_in_building'] = df_train_sorted.groupby(['latitude', 'longitude', 'built_year', 'area_m2'])['deposit'].shift(1)
        df_train_sorted['recent_rent_in_building'].fillna(df_train_sorted['deposit'], inplace=True)

        # 마지막 계약가 계산
        recent_rent_index = df_train_sorted.groupby(['latitude', 'longitude', 'built_year', 'area_m2'])['recent_rent_in_building'].nth(-1).index
        recent_rent_df = df_train_sorted.loc[recent_rent_index]

        df_valid_new = df_valid_new.merge(recent_rent_df[['latitude', 'longitude', 'built_year', 'area_m2', 'recent_rent_in_building']], 
                                           on=['latitude', 'longitude', 'built_year', 'area_m2'], how='left')

        df_valid_new['recent_rent_in_building'].fillna(recent_rent_df['recent_rent_in_building'].median(), inplace=True)

        return df_train_sorted.sort_index(), df_valid_new
        
    elif type == 'test':
        df_total_new = train_data.copy()
        df_test_new = test_data.copy()

        df_total_new_sorted = df_total_new.sort_values(by=['latitude', 'longitude', 'built_year', 'area_m2', 'date'])

        df_total_new_sorted['recent_rent_in_building'] = df_total_new_sorted.groupby(['latitude', 'longitude', 'built_year', 'area_m2'])['deposit'].shift(1)
        df_total_new_sorted['recent_rent_in_building'].fillna(df_total_new_sorted['deposit'], inplace=True)

        # 마지막 계약가 계산
        recent_rent_df_map = df_total_new_sorted[['latitude', 'longitude', 'built_year', 'area_m2', 'recent_rent_in_building']].reset_index()
        recent_rent_df_map_index = recent_rent_df_map.groupby(['latitude', 'longitude', 'built_year', 'area_m2'])['recent_rent_in_building'].nth(-1).index
        recent_rent_df_map_scaled = recent_rent_df_map.loc[recent_rent_df_map_index]

        df_test_new = df_test_new.merge(recent_rent_df_map_scaled[['latitude', 'longitude', 'built_year', 'area_m2', 'recent_rent_in_building']], 
                                         on=['latitude', 'longitude', 'built_year', 'area_m2'], how='left')

        df_test_new['recent_rent_in_building'].fillna(recent_rent_df_map['recent_rent_in_building'].median(), inplace=True)

        return df_total_new_sorted.sort_index(), df_test_new



# 동일한 지역(위도, 경도, 건축연도, 면적)의 과거 평균 전세가(중앙값으로 계산)
def add_avg_rent_in_past_year(train_data: pd.DataFrame, test_data: pd.DataFrame, type: str = 'valid') -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if type == 'valid':
        df_train_new = train_data.copy()
        df_valid_new = test_data.copy()
        
        
        df_train_new['original_index'] = df_train_new.index
        df_train_new = df_train_new.sort_values(by=['latitude', 'longitude','area_m2','built_year','date'], ascending=[True, True, True, True, True])
        
        df_train_new['avg_rent_in_past_year'] = df_train_new.groupby(['latitude', 'longitude','built_year', 'area_m2'])['deposit'].transform(lambda x : x.shift(1).expanding().median().round(1))
        df_train_new['avg_rent_in_past_year'].fillna(df_train_new['deposit'], inplace=True)

        df_train_new = df_train_new.sort_values(by='original_index').drop(columns='original_index')
        
        df_train_new_map = df_train_new.groupby(['latitude', 'longitude','built_year', 'area_m2'])['deposit'].median().reset_index()
        df_train_new_map.rename(columns={'deposit': 'deposit_past_year_median'}, inplace=True)
        
        df_valid_new = df_valid_new.merge(df_train_new_map, on=['latitude', 'longitude', 'built_year', 'area_m2'], how='left')
        df_valid_new.rename(columns = {'deposit_past_year_median': 'avg_rent_in_past_year'}, inplace=True)
    
        df_valid_new['avg_rent_in_past_year'].fillna(df_train_new_map['deposit_past_year_median'].median(), inplace=True)

        return df_train_new, df_valid_new
        
        
    elif type == 'test':
        df_total_new = train_data.copy()
        df_test_new = test_data.copy()
    
        
        df_total_new['original_index'] = df_total_new.index
        df_total_new = df_total_new.sort_values(by=['latitude', 'longitude','area_m2','built_year','date'], ascending=[True, True, True, True, True])
        
        df_total_new['avg_rent_in_past_year'] = df_total_new.groupby(['latitude', 'longitude','built_year', 'area_m2'])['deposit'].transform(lambda x : x.shift(1).expanding().median().round(1))
        df_total_new['avg_rent_in_past_year'].fillna(df_total_new['deposit'], inplace=True)

        df_total_new = df_total_new.sort_values(by='original_index').drop(columns='original_index')
        
        df_total_new_map = df_total_new.groupby(['latitude', 'longitude','built_year', 'area_m2'])['deposit'].median().reset_index()
        df_total_new_map.rename(columns={'deposit': 'deposit_past_year_median'}, inplace=True)
        
        df_test_new = df_test_new.merge(df_total_new_map, on=['latitude', 'longitude', 'built_year', 'area_m2'], how='left')
        df_test_new.rename(columns = {'deposit_past_year_median': 'avg_rent_in_past_year'}, inplace=True)

        df_test_new['avg_rent_in_past_year'].fillna(df_total_new_map['deposit_past_year_median'].median(), inplace=True)

    
    return df_total_new, df_test_new



# 연도별 전세가 상승률
def add_rent_growth_rate(train_data: pd.DataFrame, test_data: pd.DataFrame, type: str = 'valid') -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if type == 'valid':
        df_train_new = train_data.copy()
        df_valid_new = test_data.copy()

        # 연도별 전세가 중앙값 계산
        median_deposit = df_train_new.groupby(['latitude', 'longitude', 'year'])['deposit'].median().reset_index()
        median_deposit['previous_year_deposit'] = median_deposit.groupby(['latitude', 'longitude'])['deposit'].shift(1)
        median_deposit['deposit_rate'] = ((median_deposit['deposit'] - median_deposit['previous_year_deposit']) / median_deposit['previous_year_deposit']).round(2)

        # 연도별 평균 상승률 계산
        mean_deposit_rate_per_year = median_deposit.groupby('year')['deposit_rate'].transform('mean').round(2)
        median_deposit['deposit_rate'] = median_deposit['deposit_rate'].fillna(mean_deposit_rate_per_year)

        # 전체 평균 상승률 계산
        median_deposit['deposit_rate'] = median_deposit['deposit_rate'].fillna(median_deposit['deposit_rate'].mean().round(2))

        # train 데이터에 전세가 상승률 추가
        df_train_new = df_train_new.merge(median_deposit[['latitude', 'longitude', 'year', 'deposit_rate']], 
                                          on=['latitude', 'longitude', 'year'], how='left')

        # valid 데이터에 최신 전세가 상승률 추가
        recent_data = median_deposit.loc[median_deposit.groupby(['latitude', 'longitude'])['year'].idxmax()]
        df_valid_new = df_valid_new.merge(recent_data[['latitude', 'longitude', 'deposit_rate']], 
                                          on=['latitude', 'longitude'], how='left')
        
        df_valid_new['deposit_rate'] = df_valid_new['deposit_rate'].fillna(median_deposit['deposit_rate'].mean().round(2))

        # 결과를 합쳐서 반환
        return df_train_new.sort_index(), df_valid_new.sort_index()

    elif type == 'test':
        df_total_new = train_data.copy()
        df_test_new = test_data.copy()

        # 연도별 전세가 중앙값 계산
        median_deposit = df_total_new.groupby(['latitude', 'longitude', 'year'])['deposit'].median().reset_index()
        median_deposit['previous_year_deposit'] = median_deposit.groupby(['latitude', 'longitude'])['deposit'].shift(1)
        median_deposit['deposit_rate'] = ((median_deposit['deposit'] - median_deposit['previous_year_deposit']) / median_deposit['previous_year_deposit']).round(2)

        # 연도별 평균 상승률 계산
        mean_deposit_rate_per_year = median_deposit.groupby('year')['deposit_rate'].transform('mean').round(2)
        median_deposit['deposit_rate'] = median_deposit['deposit_rate'].fillna(mean_deposit_rate_per_year)

        # 전체 평균 상승률 계산
        median_deposit['deposit_rate'] = median_deposit['deposit_rate'].fillna(median_deposit['deposit_rate'].mean().round(2))

        # train + valid 데이터에 전세가 상승률 추가
        df_total_new = df_total_new.merge(median_deposit[['latitude', 'longitude', 'year', 'deposit_rate']], 
                                          on=['latitude', 'longitude', 'year'], how='left')

        # test 데이터에 최신 전세가 상승률 추가
        recent_data = median_deposit.loc[median_deposit.groupby(['latitude', 'longitude'])['year'].idxmax()]
        df_test_new = df_test_new.merge(recent_data[['latitude', 'longitude', 'deposit_rate']], 
                                        on=['latitude', 'longitude'], how='left')

        df_test_new['deposit_rate'] = df_test_new['deposit_rate'].fillna(median_deposit['deposit_rate'].mean().round(2))

        # 결과를 합쳐서 반환
        return df_total_new.sort_index(), df_test_new.sort_index()
