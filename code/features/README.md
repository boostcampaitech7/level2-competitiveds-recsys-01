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
| create_school_within_radius | schools_within_radius | 반경 내 학교 개수(default = 0.01km)   |
| create_school_counts_within_radius_by_school_level | school_info | 학교의 위경도 기준으로 clustering     |
| create_place_within_radius | school_info | 학교의 위경도 기준으로 clustering     |



## Code Working In Main

import handler.feature_engineering
