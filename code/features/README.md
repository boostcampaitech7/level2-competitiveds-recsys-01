# Features Engineering Directory

해당 디렉토리(Features)는 우리 프로젝트에서 사용된 모든 피처 엔지니어링 스크립트를 포함하고 있습니다.  
각 스크립트는 특정한 피처 세트를 생성하는 역할을 하며 아래에 나열된 설명이 있습니다.

## Directory Structure

```bash
feature/
┣ clustering_features.py
┣ count_features.py
┣ deposit_features.py
┣ distance_features.py
┣ other_features.py
┣ README.md

```
## Feature Descriptions

**clustering_features.py**
| function   | feature name    | description       |
|------------|-------------------|-------------------------|
| clustering | subway_info | subway의 위경도 기준으로 k=20 clustering  |
| clustering | park_info | park의 위경도 기준으로 k=20 clustering     |
| clustering | school_info | 학교의 위경도 기준으로 k=20 clustering     |
|create_clustering_target| cluster | target을 기준으로 k=20 clustering |
|create_clustering_target| distance_to_centroid | 군집 내 centroid와의 거리 |
|create_cluster_deposit_median| cluster_median | target 기준의 cluster 중앙 전세값 |

**count_features.py**
| function   | feature name    | description       |
|------------|----------|-------------------------|
| transaction_count_function | transaction_count_last_{months}_months | n 개월 동일한 아파트 거래량(default = 3)  |
| create_subway_within_radius | subways_within_radius | 반경 내 지하철 개수(default = 0.01km)    |
| create_school_within_radius | schools_within_radius | 반경 내 학교 개수(default = 0.02km)   |
| create_school_counts_within_radius_by_school_level | elementary_schools_within_radius | 반경 이내 초등학교 개수    |
| create_school_counts_within_radius_by_school_level | middle_schools_within_radius | 반경 이내 중학교 개수  |
| create_school_counts_within_radius_by_school_level | high_schools_within_radius | 반경 이내 고등학교 개수 |
| create_place_within_radius | public_facility_count | 반경 이내 공공시설 개수(default = 0.01km  |

**distance_features.py**
| function   | feature name    | description       |
|------------|----------|-------------------------|
| create_nearest_subway_distance | nearest_subway_distance | 가장 가까운 지하철까지의 거리 |
| create_nearest_park_distance_and_area | nearest_park_distance |가장 가까운 공원까지의 거리 |
| create_nearest_park_distance_and_area | nearest_park_area |가장 가까운 공원 면적 |   
| create_nearest_school_distance | nearest_elementary_distance | 가장 가까운 초등학교까지 거리 |
| create_nearest_school_distance | nearest_middle_distance | 가장 가까운 중학교까지 거리 |
| create_nearest_school_distance | nearest_high_distance | 가장 가까운 고등학교까지 거리 |
| weighted_subway_distance | weight |환승역 가중치 거리 계산  |


**other_features.py**
| function   | feature name    | description       |
|------------|----------|-------------------------|
| create_temporal_feature | year,month,date,quarter,season ... | year, month, date 조작 변수 |
| create_sin_cos_season | season_sin & cos |계절 mapping 후 sine, cosine 적용 |
| create_floor_area_interaction | floor_area_interaction |층수와 면적의 관계 |   
| distance_gangnam | distance_km | 강남까지의 거리 |
| distance_gangnam | distance_category | 강남까지의 범주화 |
| create_sum_park_area_within_radius | nearest_park_area_sum | 반경 이내 공원 면적의 합 |
| shift_interest_rate_function | interest_rate_{year}year or {month}month | 계약 시점 기준 이전 금리(default = 3,6,12) |
| categorization | age_category | age 범주  |
| categorization | floor_category |floor 범주  |
| categorization | area_category | area 범주  |


**deposit_features.py**
| function   | feature name    | description       |
|------------|----------|-------------------------|
| add_recent_rent_in_building | recent_rent_in_building | 동일한 아파트(위도, 경도, 건축연도, 면적)의 최근 전세가 |
| add_avg_rent_in_past_year | avg_rent_in_past_year |동일한 지역(위도, 경도, 건축연도, 면적)의 과거 평균 전세가(중앙값으로 계산) |
| add_rent_growth_rate | deposit_rate | 연도별 최근 전세가 상승률 |   



## Usage

피처 엔지니어링 프로세스를 사용하려면, handler/feature_engineering.py 파일에서 feature_engineering 함수를 간단히 import하면 됩니다.  
이 함수는 개별 피처 생성 스크립트를 자동으로 실행하고 통합된 피처 세트를 반환합니다.


```python
# main.py
from handler.feature_engineering as fe

train_df, valid_df, test_df = fe.feature_engineering(train,valid,test)

```
