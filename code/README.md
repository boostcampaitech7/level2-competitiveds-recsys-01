# Feature Engineering Directory

This directory contains all the feature engineering scripts used in our project. Each script is responsible for creating a specific set of features, which are listed and explained below.

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
|------------|----------|-------------------------|
| clustering | subway_info | subway의 위경도 기준으로 clustering  |
| clustering | park_info | park의 위경도 기준으로 clustering     |
| clustering | school_info | 학교의 위경도 기준으로 clustering     |


Feature Name	Description
cluster_id	Assigns each user to a cluster based on transaction patterns
cluster_avg_spent	Average spending within each cluster
cluster_freq	Frequency of transactions per cluster
count_features.py
Feature Name	Description
txn_count	Total number of transactions by a user
txn_count_last_30	Number of transactions in the last 30 days
txn_count_large	Number of transactions above a certain threshold
deposit_features.py
Feature Name	Description
total_deposit	Total deposit amount over a given time period
avg_deposit	Average deposit amount per transaction
deposit_freq	Frequency of deposit transactions
distance_features.py
Feature Name	Description
avg_distance	Average geographical distance between transaction locations
max_distance	Maximum distance observed between transaction locations
distance_from_home	Distance of transactions from user's registered address




## Code Working In Main

import handler.feature_engineering
