from sklearn.cluster import KMeans
from utils.constant_utils import Config, Directory
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from geopy.distance import great_circle


### ê¸ˆë¦¬ shift ?•¨?ˆ˜
def shift_interest_rate_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, month : int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)
    
    total_data = pd.concat([train_data[['date','interest_rate']],valid_data[['date','interest_rate']],test_data[['date','interest_rate']]], axis=0)
    
    # ?›?˜?˜ ?¸?±?Š¤ ????¥
    total_data['original_index'] = total_data.index
    
    # ?°?´?„° ? •? ¬ (date ê¸°ì??)
    df_sorted = df.sort_values('date').reset_index(drop=True)

    # ê³¼ê±° ê¸ˆë¦¬ ? •ë³? êµ¬í•˜ê¸?
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
    
    # ?•„?š” ?—†?Š” ì¤‘ê°„ ?‚ ì§? ì»¬ëŸ¼(drop)
    df_sorted = df_sorted.drop(columns=['date_minus_1year', 'date_1year', 'date_minus_6months', 'date_6months', 'date_minus_3months', 'date_3months'])

    df_sorted['interest_rate_3months'] = df_sorted['interest_rate_3months'].fillna(df_sorted['interest_rate'])
    df_sorted['interest_rate_6months'] = df_sorted['interest_rate_6months'].fillna(df_sorted['interest_rate'])
    df_sorted['interest_rate_1year'] = df_sorted['interest_rate_1year'].fillna(df_sorted['interest_rate'])

    df_final = df_sorted.sort_values('original_index').drop(columns=['original_index']).reset_index(drop=True)

    train_data_ = df_final.iloc[:train_data_length,:]
    valid_data_ = df_final.iloc[train_data_length:train_data_length+valid_data_length,:]
    test_data_ = df_final.iloc[train_data_length+valid_data_length:,:]
        
    return train_data_, valid_data_, test_data_




### n ê°œì›” ?™?¼?•œ ?•„?ŒŒ?Š¸ ê±°ë˜?Ÿ‰ ?•¨?ˆ˜
def transaction_count_function(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, month : int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data_length = len(train_data)
    valid_data_length = len(valid_data)
    test_data_length = len(test_data)
    
    train_data_tot = pd.concat([train_data, valid_data], axis=0)
    total_data = pd.concat([train_data_tot, test_data], axis=0)
    
    total_data['transaction_count_last_3_months'] = 0
    
    # ?œ„?„, ê²½ë„, ê±´ì¶• ?—°?„ë¡? ê·¸ë£¹?™”
    grouped = total_data.groupby(['latitude', 'longitude', 'built_year'])
    
    # ê°? ê·¸ë£¹?— ????•´ ê±°ë˜?Ÿ‰ ê³„ì‚°
    for (lat, lon, built_year), group in tqdm(grouped, desc="Calculating previous 3 months transaction counts by location and year"):
    # ê·¸ë£¹ ?‚´ ê±°ë˜?¼ ? •? ¬
        group = group.sort_values(by='date')
    
        # ê±°ë˜?Ÿ‰?„ ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”
        transaction_counts = []

        for idx, row in group.iterrows():
            # ?˜„?¬ ê±°ë˜?¼ë¡œë???„° month ?´? „ ?‚ ì§? ê³„ì‚°
            end_date = row['date']
            start_date = end_date - pd.DateOffset(months=month)

            # ?™?¼?•œ ?•„?ŒŒ?Š¸?—?„œ?˜ ê±°ë˜?Ÿ‰ ê³„ì‚°
            transaction_count = group[
                (group['date'] < end_date) &  # ?˜„?¬ ê±°ë˜?¼ ?´? „
                (group['date'] >= start_date)  # month ?´? „ 
                ].shape[0]

            # ê±°ë˜?Ÿ‰ ë¦¬ìŠ¤?Š¸?— ì¶”ê??
            transaction_counts.append(transaction_count)

        # ë°°ì¹˜ ê²°ê³¼ë¥? ?°?´?„°?”„? ˆ?„?— ????¥
        total_data.loc[group.index, 'transaction_count_last_3_months'] = transaction_counts

    train_data_ = total_data.iloc[:train_data_length,:]
    valid_data_ = total_data.iloc[train_data_length:train_data_length+valid_data_length,:]
    test_data_ = total_data.iloc[train_data_length+valid_data_length:,:]
        
    return train_data_, valid_data_, test_data_



### ?´?Ÿ¬?Š¤?„°ë§?

def clustering(total_df, info_df, feat_name, n_clusters=20):
    info = info_df[['longitude', 'latitude']].values
    
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=Config.RANDOM_SEED)
    kmeans.fit(info)
    
    clusters = kmeans.predict(total_df[['longitude', 'latitude']].values)
    total_df[feat_name] = pd.DataFrame(clusters, dtype='category')
    return total_df

def create_cluster_density(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ?´?Ÿ¬?Š¤?„°ë³? ë°??„ ê³„ì‚° (?´?Ÿ¬?Š¤?„°ë³? ?¬?¸?Š¸ ?ˆ˜)
    cluster_density = train_data.groupby('cluster').size().reset_index(name='density')

    train_data = train_data.merge(cluster_density, on='cluster', how='left')
    valid_data = valid_data.merge(cluster_density, on='cluster', how='left')
    test_data = test_data.merge(cluster_density, on='cluster', how='left')

    return train_data, valid_data, test_data

def create_cluster_distance_to_centroid(data: pd.DataFrame, centroids) -> pd.DataFrame:
    # ?¬?•¨?˜?Š” êµ°ì§‘?˜ centroid????˜ ê±°ë¦¬ ê³„ì‚°
    lat_centroids = np.array([centroids[cluster, 0] for cluster in data['cluster']])
    lon_centroids = np.array([centroids[cluster, 1] for cluster in data['cluster']])
    lat_diff = data['latitude'].values - lat_centroids
    lon_diff = data['longitude'].values - lon_centroids
    data['distance_to_centroid'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    return data

def create_clustering_target(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # K-means ?´?Ÿ¬?Š¤?„°ë§?
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_SEED)
    train_data['cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
    valid_data['cluster'] = kmeans.predict(valid_data[['latitude', 'longitude']])
    test_data['cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])
    
    train_data['cluster'] = train_data['cluster'].astype('category')
    valid_data['cluster'] = valid_data['cluster'].astype('category')
    test_data['cluster'] = test_data['cluster'].astype('category')

    # êµ°ì§‘ ë°??„ ë³??ˆ˜ ì¶”ê??
    train_data, valid_data, test_data = create_cluster_density(train_data, valid_data, test_data)

    centroids = kmeans.cluster_centers_

    # êµ°ì§‘ centroidê¹Œì???˜ ê±°ë¦¬ ë³??ˆ˜ ì¶”ê??
    train_data = create_cluster_distance_to_centroid(train_data, centroids)
    valid_data = create_cluster_distance_to_centroid(valid_data, centroids)
    test_data = create_cluster_distance_to_centroid(test_data, centroids)

    return train_data, valid_data, test_data




### ê±°ë¦¬

# ê°??¥ ê°?ê¹Œìš´ ì§??•˜ì² ê¹Œì§??˜ ê±°ë¦¬ ?•¨?ˆ˜
def create_nearest_subway_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subwayInfo = Directory.subway_info

    # KD-?Š¸ë¦? ?ƒ?„±
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    # ê±°ë¦¬ ê³„ì‚° ?•¨?ˆ˜ ? •?˜
    def add_nearest_subway_distance(data):
        # ê°? ì§‘ì˜ ì¢Œí‘œ ê°?? ¸?˜¤ê¸?
        house_coords = data[['latitude', 'longitude']].values
        # ê°??¥ ê°?ê¹Œìš´ ì§??•˜ì²? ?—­ê¹Œì???˜ ê±°ë¦¬ ê³„ì‚°
        distances, indices = tree.query(house_coords, k=1)  # k=1: ê°??¥ ê°?ê¹Œìš´ ?—­
        # ê±°ë¦¬ë¥? ?°?´?„°?”„? ˆ?„?— ì¶”ê?? (ë¯¸í„° ?‹¨?œ„ë¡? ë³??™˜)
        data['nearest_subway_distance'] = distances.flatten()
        return data

    # ê°? ?°?´?„°?…‹?— ????•´ ê±°ë¦¬ ì¶”ê??
    train_data = add_nearest_subway_distance(train_data)
    valid_data = add_nearest_subway_distance(valid_data)
    test_data = add_nearest_subway_distance(test_data)

    return train_data, valid_data, test_data

# ë°˜ê²½ ?‚´ ì§??•˜ì²? ê°œìˆ˜ ?•¨?ˆ˜
def create_subway_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo?—?Š” ì§??•˜ì²? ?—­?˜ ?œ„?„??? ê²½ë„ê°? ?¬?•¨?˜?–´ ?ˆ?‹¤ê³? ê°?? •
    subwayInfo = Directory.subway_info
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    def count_subways_within_radius(data, radius):
        counts = []  # ì´ˆê¸°?™”
        for i in range(0, len(data), 10000):
            batch = data.iloc[i:i+10000]
            house_coords = batch[['latitude', 'longitude']].values
            # KDTreeë¥? ?‚¬?š©?•˜?—¬ ì£¼ì–´ì§? ë°˜ê²½ ?‚´ ì§??•˜ì² ì—­ ì°¾ê¸°
            indices = tree.query_radius(house_coords, r=radius)  # ë°˜ê²½?— ????•œ ?¸?±?Š¤
            # ê°? ì§‘ì˜ ì£¼ë?? ì§??•˜ì² ì—­ ê°œìˆ˜ ?„¸ê¸?
            counts.extend(len(idx) for idx in indices)

        # countsê°? ?°?´?„°?”„? ˆ?„ ?¬ê¸°ë³´?‹¤ ?‘?„ ê²½ìš° 0?œ¼ë¡? ì±„ìš°ê¸?
        if len(counts) < len(data):
            counts.extend([0] * (len(data) - len(counts)))
        
        # ?°?´?„°?”„? ˆ?„?— ê²°ê³¼ ì¶”ê??
        data['subways_within_radius'] = counts
        return data

    # ê°? ?°?´?„°?…‹?— ????•´ ê±°ë¦¬ ì¶”ê??
    radius = 0.01  # ?•½ 1km
    train_data = count_subways_within_radius(train_data, radius)
    valid_data = count_subways_within_radius(valid_data, radius)
    test_data = count_subways_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_nearest_park_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
# ê°??¥ ê°?ê¹Œìš´ ê³µì› ê±°ë¦¬ ë°? ë©´ì  ?•¨?ˆ˜
    park_data = Directory.park_info

    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # ?ˆ˜?„ê¶? ê³µì›?˜ ì¢Œí‘œë¡? KDTree ?ƒ?„±
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def add_nearest_park_features(data):
        # ê°? ì§‘ì˜ ì¢Œí‘œë¡? ê°??¥ ê°?ê¹Œìš´ ê³µì› ì°¾ê¸°
        house_coords = data[['latitude', 'longitude']].values
        distances, indices = park_tree.query(house_coords, k=1)  # ê°??¥ ê°?ê¹Œìš´ ê³µì› ì°¾ê¸°

        # ê°??¥ ê°?ê¹Œìš´ ê³µì›ê¹Œì???˜ ê±°ë¦¬ ë°? ?•´?‹¹ ê³µì›?˜ ë©´ì  ì¶”ê??
        nearest_park_distances = distances.flatten()

        data['nearest_park_distance'] = nearest_park_distances
        nearest_park_areas = seoul_area_parks.iloc[indices.flatten()]['area'].values  # ë©´ì  ? •ë³´ë?? ê°?? ¸?˜´

        data['nearest_park_distance'] = nearest_park_distances
        data['nearest_park_area'] = nearest_park_areas
        return data

    # train, valid, test ?°?´?„°?— ê°??¥ ê°?ê¹Œìš´ ê³µì› ê±°ë¦¬ ë°? ë©´ì  ì¶”ê??
    train_data = add_nearest_park_features(train_data)
    valid_data = add_nearest_park_features(valid_data)
    test_data = add_nearest_park_features(test_data)

    return train_data, valid_data, test_data

# ë°˜ê²½ ?‚´ ?•™êµ? ê°œìˆ˜ ?•¨?ˆ˜
def create_school_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]
    school_coords = seoul_area_school[['latitude', 'longitude']].values
    tree = KDTree(school_coords, leaf_size=10)

    def count_schools_within_radius(data, radius):
        counts = []  # ?•™êµ? ê°œìˆ˜ë¥? ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”
        for i in range(0, len(data), 10000):  # 10,000ê°œì”© ë°°ì¹˜ë¡? ì²˜ë¦¬
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = tree.query_radius(house_coords, r=radius)  # ë°˜ê²½ ?‚´?˜ ?¸?±?Š¤ ì°¾ê¸°
            counts.extend(len(idx) for idx in indices)  # ê°? ë°°ì¹˜?˜ ?•™êµ? ê°œìˆ˜ ì¶”ê??
        data['schools_within_radius'] = counts  # ?°?´?„°?— ì¶”ê??
        return data
    
    radius = 0.02 # ?•½ 2km
    radius = 0.01 # ?•½ 1km
    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_sum_park_area_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    # ?ˆ˜?„ê¶? ê³µì›?˜ ì¢Œí‘œ ?•„?„°ë§?
    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                  (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # ?ˆ˜?„ê¶? ê³µì›?˜ ì¢Œí‘œë¡? KDTree ?ƒ?„±
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def sum_park_area_within_radius(data, radius=0.02):
        area_sums = []  # ê³µì› ë©´ì  ?•©?„ ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”
        for i in range(0, len(data), 10000):  # 10,000ê°œì”© ë°°ì¹˜ë¡? ì²˜ë¦¬
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = park_tree.query_radius(house_coords, r=radius)  # ë°˜ê²½ ?‚´?˜ ?¸?±?Š¤ ì°¾ê¸°
            
            # ê°? ì§‘ì— ????•´ ë°˜ê²½ 2km ?´?‚´?˜ ê³µì› ë©´ì ?˜ ?•©?„ ê³„ì‚°
            for idx in indices:
                if idx.size > 0:  # 2km ?´?‚´?— ê³µì›?´ ?ˆ?„ ê²½ìš°
                    areas_sum = seoul_area_parks.iloc[idx]['area'].sum()
                else:
                    areas_sum = 0  # ê³µì›?´ ?—†?Š” ê²½ìš° ë©´ì  0
                area_sums.append(areas_sum)  # ë©´ì  ?•© ì¶”ê??

        # ê²°ê³¼ ì¶”ê??
        data['nearest_park_area_sum'] = area_sums
        return data

    # train, valid, test ?°?´?„°?— ë°˜ê²½ 2km ?´?‚´?˜ ê³µì› ë©´ì  ?•© ì¶”ê??
    train_data = sum_park_area_within_radius(train_data)
    valid_data = sum_park_area_within_radius(valid_data)
    test_data = sum_park_area_within_radius(test_data)

    return train_data, valid_data, test_data

def create_school_counts_within_radius_by_school_level(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                     (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]
    
    # ì´?, ì¤?, ê³ ë“±?•™êµì˜ ì¢Œí‘œë¥? ë¶„ë¦¬
    elementary_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'elementary']
    middle_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'middle']
    high_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'high']

    # ê°? ?•™êµ? ?œ ?˜•?˜ ì¢Œí‘œë¡? KDTree ?ƒ?„±
    elementary_coords = elementary_schools[['latitude', 'longitude']].values
    middle_coords = middle_schools[['latitude', 'longitude']].values
    high_coords = high_schools[['latitude', 'longitude']].values

    tree_elementary = KDTree(elementary_coords, leaf_size=10)
    tree_middle = KDTree(middle_coords, leaf_size=10)
    tree_high = KDTree(high_coords, leaf_size=10)

    def count_schools_within_radius(data, radius):
        counts_elementary = []  # ì´ˆë“±?•™êµ? ê°œìˆ˜ë¥? ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”
        counts_middle = []      # ì¤‘í•™êµ? ê°œìˆ˜ë¥? ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”
        counts_high = []        # ê³ ë“±?•™êµ? ê°œìˆ˜ë¥? ????¥?•  ë¦¬ìŠ¤?Š¸ ì´ˆê¸°?™”

        for i in range(0, len(data), 10000):  # 10,000ê°œì”© ë°°ì¹˜ë¡? ì²˜ë¦¬
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            
            # ê°? ?•™êµ? ?œ ?˜•?˜ ê°œìˆ˜ ?„¸ê¸?
            indices_elementary = tree_elementary.query_radius(house_coords, r=radius)
            indices_middle = tree_middle.query_radius(house_coords, r=radius)
            indices_high = tree_high.query_radius(house_coords, r=radius)
            
            counts_elementary.extend(len(idx) for idx in indices_elementary)  # ê°? ë°°ì¹˜?˜ ì´ˆë“±?•™êµ? ê°œìˆ˜ ì¶”ê??
            counts_middle.extend(len(idx) for idx in indices_middle)        # ê°? ë°°ì¹˜?˜ ì¤‘í•™êµ? ê°œìˆ˜ ì¶”ê??
            counts_high.extend(len(idx) for idx in indices_high)            # ê°? ë°°ì¹˜?˜ ê³ ë“±?•™êµ? ê°œìˆ˜ ì¶”ê??

        # ?°?´?„°?— ì¶”ê??
        data['elementary_schools_within_radius'] = counts_elementary
        data['middle_schools_within_radius'] = counts_middle
        data['high_schools_within_radius'] = counts_high
        
        return data

    radius = 0.02  # ?•½ 2km
    train_data = count_schools_within_radius(train_data, radius)
    valid_data = count_schools_within_radius(valid_data, radius)
    test_data = count_schools_within_radius(test_data, radius)

    return train_data, valid_data, test_data


def create_cluster_distance_to_centroid(data: pd.DataFrame, centroids) -> pd.DataFrame:
    # Æ÷ÇÔµÇ´Â ±ºÁıÀÇ centroid¿ÍÀÇ °Å¸® °è»ê
    lat_centroids = np.array([centroids[cluster, 0] for cluster in data['cluster']])
    lon_centroids = np.array([centroids[cluster, 1] for cluster in data['cluster']])
    lat_diff = data['latitude'].values - lat_centroids
    lon_diff = data['longitude'].values - lon_centroids
    data['distance_to_centroid'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    return data

def create_clustering_target(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # K-means Å¬·¯½ºÅÍ¸µ
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_SEED)
    train_data['cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
    valid_data['cluster'] = kmeans.predict(valid_data[['latitude', 'longitude']])
    test_data['cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])
    
    train_data['cluster'] = train_data['cluster'].astype('category')
    valid_data['cluster'] = valid_data['cluster'].astype('category')
    test_data['cluster'] = test_data['cluster'].astype('category')

    # ±ºÁı ¹Ğµµ º¯¼ö Ãß°¡
    train_data, valid_data, test_data = create_cluster_density(train_data, valid_data, test_data)

    centroids = kmeans.cluster_centers_

    # ±ºÁı centroid±îÁöÀÇ °Å¸® º¯¼ö Ãß°¡
    train_data = create_cluster_distance_to_centroid(train_data, centroids)
    valid_data = create_cluster_distance_to_centroid(valid_data, centroids)
    test_data = create_cluster_distance_to_centroid(test_data, centroids)

    return train_data, valid_data, test_data

def create_nearest_subway_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subwayInfo = Directory.subway_info

    # KD-Æ®¸® »ı¼º
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    # °Å¸® °è»ê ÇÔ¼ö Á¤ÀÇ
    def add_nearest_subway_distance(data):
        # °¢ ÁıÀÇ ÁÂÇ¥ °¡Á®¿À±â
        house_coords = data[['latitude', 'longitude']].values
        # °¡Àå °¡±î¿î ÁöÇÏÃ¶ ¿ª±îÁöÀÇ °Å¸® °è»ê
        distances, indices = tree.query(house_coords, k=1)  # k=1: °¡Àå °¡±î¿î ¿ª
        # °Å¸®¸¦ µ¥ÀÌÅÍÇÁ·¹ÀÓ¿¡ Ãß°¡ (¹ÌÅÍ ´ÜÀ§·Î º¯È¯)
        data['nearest_subway_distance'] = distances.flatten()
        return data

    # °¢ µ¥ÀÌÅÍ¼Â¿¡ ´ëÇØ °Å¸® Ãß°¡
    train_data = add_nearest_subway_distance(train_data)
    valid_data = add_nearest_subway_distance(valid_data)
    test_data = add_nearest_subway_distance(test_data)

    return train_data, valid_data, test_data

def create_subway_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo¿¡´Â ÁöÇÏÃ¶ ¿ªÀÇ À§µµ¿Í °æµµ°¡ Æ÷ÇÔµÇ¾î ÀÖ´Ù°í °¡Á¤
    subwayInfo = Directory.subway_info
    subway_coords = subwayInfo[['latitude', 'longitude']].values
    tree = KDTree(subway_coords, leaf_size=10)

    def count_subways_within_radius(data, radius):
        counts = []  # ÃÊ±âÈ­
        for i in range(0, len(data), 10000):
            batch = data.iloc[i:i+10000]
            house_coords = batch[['latitude', 'longitude']].values
            # KDTree¸¦ »ç¿ëÇÏ¿© ÁÖ¾îÁø ¹İ°æ ³» ÁöÇÏÃ¶¿ª Ã£±â
            indices = tree.query_radius(house_coords, r=radius)  # ¹İ°æ¿¡ ´ëÇÑ ÀÎµ¦½º
            # °¢ ÁıÀÇ ÁÖº¯ ÁöÇÏÃ¶¿ª °³¼ö ¼¼±â
            counts.extend(len(idx) for idx in indices)

        # counts°¡ µ¥ÀÌÅÍÇÁ·¹ÀÓ Å©±âº¸´Ù ÀÛÀ» °æ¿ì 0À¸·Î Ã¤¿ì±â
        if len(counts) < len(data):
            counts.extend([0] * (len(data) - len(counts)))
        
        # µ¥ÀÌÅÍÇÁ·¹ÀÓ¿¡ °á°ú Ãß°¡
        data['subways_within_radius'] = counts
        return data

    # °¢ µ¥ÀÌÅÍ¼Â¿¡ ´ëÇØ °Å¸® Ãß°¡
    radius = 0.01  # ¾à 1km
    train_data = count_subways_within_radius(train_data, radius)
    valid_data = count_subways_within_radius(valid_data, radius)
    test_data = count_subways_within_radius(test_data, radius)

    return train_data, valid_data, test_data

def create_nearest_park_distance_and_area(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # ¼öµµ±Ç °ø¿øÀÇ ÁÂÇ¥·Î KDTree »ı¼º
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def add_nearest_park_features(data):
        # °¢ ÁıÀÇ ÁÂÇ¥·Î °¡Àå °¡±î¿î °ø¿ø Ã£±â
        house_coords = data[['latitude', 'longitude']].values
        distances, indices = park_tree.query(house_coords, k=1)  # °¡Àå °¡±î¿î °ø¿ø Ã£±â

        # °¡Àå °¡±î¿î °ø¿ø±îÁöÀÇ °Å¸® ¹× ÇØ´ç °ø¿øÀÇ ¸éÀû Ãß°¡
        nearest_park_distances = distances.flatten()
        nearest_park_areas = seoul_area_parks.iloc[indices.flatten()]['area'].values  # ¸éÀû Á¤º¸¸¦ °¡Á®¿È

        data['nearest_park_distance'] = nearest_park_distances
        data['nearest_park_area'] = nearest_park_areas
        return data

    # train, valid, test µ¥ÀÌÅÍ¿¡ °¡Àå °¡±î¿î °ø¿ø °Å¸® ¹× ¸éÀû Ãß°¡
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
        counts = []  # ÇĞ±³ °³¼ö¸¦ ÀúÀåÇÒ ¸®½ºÆ® ÃÊ±âÈ­
        for i in range(0, len(data), 10000):  # 10,000°³¾¿ ¹èÄ¡·Î Ã³¸®
            batch = data.iloc[i:i + 10000]
            house_coords = batch[['latitude', 'longitude']].values
            indices = tree.query_radius(house_coords, r=radius)  # ¹İ°æ ³»ÀÇ ÀÎµ¦½º Ã£±â
            counts.extend(len(idx) for idx in indices)  # °¢ ¹èÄ¡ÀÇ ÇĞ±³ °³¼ö Ãß°¡
        data['schools_within_radius'] = counts  # µ¥ÀÌÅÍ¿¡ Ãß°¡
        return data
    
def distance_gangnam(df):
    gangnam = (37.498095, 127.028361548)

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
        # ê¸°ë³¸ ?Š¹?„± ?ƒ?„± (ëª¨ë“  ?°?´?„°?…‹?— ?™?¼?•˜ê²? ? ?š© ê°??Š¥)
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



def create_nearest_park_distance_and_area(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    park_data = Directory.park_info

    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]

    # ?ˆ˜?„ê¶? ê³µì›?˜ ì¢Œí‘œë¡? KDTree ?ƒ?„±
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(park_coords, leaf_size=10)

    def add_nearest_park_features(data):
        # ê°? ì§‘ì˜ ì¢Œí‘œë¡? ê°??¥ ê°?ê¹Œìš´ ê³µì› ì°¾ê¸°
        house_coords = data[['latitude', 'longitude']].values
        distances, indices = park_tree.query(house_coords, k=1)  # ê°??¥ ê°?ê¹Œìš´ ê³µì› ì°¾ê¸°

        # ê°??¥ ê°?ê¹Œìš´ ê³µì›ê¹Œì???˜ ê±°ë¦¬ ë°? ?•´?‹¹ ê³µì›?˜ ë©´ì  ì¶”ê??
        nearest_park_distances = distances.flatten()
        nearest_park_areas = seoul_area_parks.iloc[indices.flatten()]['area'].values  # ë©´ì  ? •ë³´ë?? ê°?? ¸?˜´

        data['nearest_park_distance'] = nearest_park_distances
        data['nearest_park_area'] = nearest_park_areas
        return data

    # train, valid, test ?°?´?„°?— ê°??¥ ê°?ê¹Œìš´ ê³µì› ê±°ë¦¬ ë°? ë©´ì  ì¶”ê??
    train_data = add_nearest_park_features(train_data)
    valid_data = add_nearest_park_features(valid_data)
    test_data = add_nearest_park_features(test_data)

    return train_data, valid_data, test_data



def assign_info_cluster(train_data, school_info, park_info, subway_info):
    min_latitude = min(train_data['latitude'])
    max_latitude = max(train_data['latitude'])

    min_longitude = min(train_data['longitude'])
    max_longitude = max(train_data['longitude'])

    school_info_filtered = school_info[(school_info['latitude'] >= min_latitude) & (school_info['latitude'] <= max_latitude) & (school_info['longitude'] >= min_longitude) & (school_info['longitude'] <= max_longitude)]
    park_info_filtered = park_info[(park_info['latitude'] >= min_latitude) & (park_info['latitude'] <= max_latitude) & (park_info['longitude'] >= min_longitude) & (park_info['longitude'] <= max_longitude)]
    subway_info_filtered = subway_info[(subway_info['latitude'] >= min_latitude) & (subway_info['latitude'] <= max_latitude) & (subway_info['longitude'] >= min_longitude) & (subway_info['longitude'] <= max_longitude)]

    # train_dataë¡? ?´?Ÿ¬?Š¤?„° ?˜•?„±
    X_train = train_data[['latitude', 'longitude']].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=25, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)

    # ?‹¤ë¥? ?°?´?„°?…‹?— ?´?Ÿ¬?Š¤?„° ?• ?‹¹
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

    # 3. quarter ì²˜ë¦¬: ?ˆœ?™˜ ?¸ì½”ë”©
    df_new['quarter_sin'] = np.sin(2 * np.pi * df_new['quarter'] / 4)
    df_new['quarter_cos'] = np.cos(2 * np.pi * df_new['quarter'] / 4)

    tmp = pd.get_dummies(df_new['contract_type'], prefix='contract_type').astype(int)
    df_new = pd.concat([df_new, tmp], axis = 1)

    #df_new = df_new.drop(['year', 'built_year', 'month', 'quarter', 'contract_type'], axis = 1)
    return df_new


# ë°˜ê²½ ?‚´ ê³µê³µ?‹œ?„¤(?•™êµ?, ì§??•˜ì²?, ê³µì›) ê°œìˆ˜ ?•¨?ˆ˜
def create_place_within_radius(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # subwayInfo?—?Š” ì§??•˜ì²? ?—­?˜ ?œ„?„??? ê²½ë„ê°? ?¬?•¨?˜?–´ ?ˆ?‹¤ê³? ê°?? •
    subway_data = Directory.subway_info
    park_data = Directory.park_info
    school_data = Directory.school_info

    
    # seoul park ì¸¡ì •
    seoul_area_parks = park_data[(park_data['latitude'] >= 37.0) & (park_data['latitude'] <= 38.0) &
                                    (park_data['longitude'] >= 126.0) & (park_data['longitude'] <= 128.0)]
    # seoul school ì¸¡ì •
    seoul_area_school = school_data[(school_data['latitude'] >= 37.0) & (school_data['latitude'] <= 38.0) &
                                    (school_data['longitude'] >= 126.0) & (school_data['longitude'] <= 128.0)]
    # seoul subway ì¸¡ì •
    seoul_area_subway = subway_data[(subway_data['latitude'] >= 37.0) & (subway_data['latitude'] <= 38.0) &
                                (subway_data['longitude'] >= 126.0) & (subway_data['longitude'] <= 128.0)]
    
    
    
    subway_coords = seoul_area_subway[['latitude', 'longitude']].values
    subway_tree = KDTree(subway_coords, leaf_size=10)
    school_coords = seoul_area_school[['latitude', 'longitude']].values
    school_tree = KDTree(subway_coords, leaf_size=10)
    park_coords = seoul_area_parks[['latitude', 'longitude']].values
    park_tree = KDTree(subway_coords, leaf_size=10)


    # count ?•¨?ˆ˜
    def count_within_radius(data, radius, tree):
            counts = []  # ì´ˆê¸°?™”
            for i in range(0, len(data), 10000):
                batch = data.iloc[i:i+10000]
                house_coords = batch[['latitude', 'longitude']].values
                # KDTreeë¥? ?‚¬?š©?•˜?—¬ ì£¼ì–´ì§? ë°˜ê²½ ?‚´ ì§??•˜ì² ì—­ ì°¾ê¸°
                indices = tree.query_radius(house_coords, r=radius)  # ë°˜ê²½?— ????•œ ?¸?±?Š¤
                # ê°? ì§‘ì˜ ì£¼ë?? ì§??•˜ì² ì—­ ê°œìˆ˜ ?„¸ê¸?
                counts.extend(len(idx) for idx in indices)

            # countsê°? ?°?´?„°?”„? ˆ?„ ?¬ê¸°ë³´?‹¤ ?‘?„ ê²½ìš° 0?œ¼ë¡? ì±„ìš°ê¸?
            if len(counts) < len(data):
                counts.extend([0] * (len(data) - len(counts)))
        
            return counts
    
    # ê°? ?°?´?„°?…‹?— ????•´ ê±°ë¦¬ ì¶”ê??
    radius = 0.01  # ?•½ 1km

    # ê°? ?°?´?„°?…‹?— ????•´ count ê³„ì‚°
    train_subway_counts = count_within_radius(train_data, radius, subway_tree)
    train_school_counts = count_within_radius(train_data, radius, school_tree)
    train_park_counts = count_within_radius(train_data, radius, park_tree)

    valid_subway_counts = count_within_radius(valid_data, radius, subway_tree)
    valid_school_counts = count_within_radius(valid_data, radius, school_tree)
    valid_park_counts = count_within_radius(valid_data, radius, park_tree)

    test_subway_counts = count_within_radius(test_data, radius, subway_tree)
    test_school_counts = count_within_radius(test_data, radius, school_tree)
    test_park_counts = count_within_radius(test_data, radius, park_tree)

    # ê°? ?°?´?„°?…‹?˜ ê³µê³µ?‹œ?„¤ ì´? ì¹´ìš´?Š¸ ê³„ì‚°
    train_counts = [subway + school + park for subway, school, park in zip(train_subway_counts, train_school_counts, train_park_counts)]
    valid_counts = [subway + school + park for subway, school, park in zip(valid_subway_counts, valid_school_counts, valid_park_counts)]
    test_counts = [subway + school + park for subway, school, park in zip(test_subway_counts, test_school_counts, test_park_counts)]

    # ê°? ?°?´?„°?…‹?— ì¹´ìš´?Š¸ë¥? ì¶”ê??
    train_data['public_facility_count'] = train_counts
    valid_data['public_facility_count'] = valid_counts
    test_data['public_facility_count'] = test_counts
    
    return train_data, valid_data, test_data




### ë²”ì£¼?™”

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

#levelº° °¡Àå °¡±î¿î ÇĞ±³±îÁö °Å¸®
def create_nearest_school_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    school_info = Directory.school_info
    seoul_area_school = school_info[(school_info['latitude'] >= 37.0) & (school_info['latitude'] <= 38.0) &
                                (school_info['longitude'] >= 126.0) & (school_info['longitude'] <= 128.0)]

    elementary_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'elementary']
    middle_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'middle']
    high_schools = seoul_area_school[seoul_area_school['schoolLevel'] == 'high']

    # °¢ ÇĞ±³ À¯Çü¿¡ ´ëÇØ BallTree »ı¼º
    elementary_tree = BallTree(np.radians(elementary_schools[['latitude', 'longitude']]), metric='haversine')
    middle_tree = BallTree(np.radians(middle_schools[['latitude', 'longitude']]), metric='haversine')
    high_tree = BallTree(np.radians(high_schools[['latitude', 'longitude']]), metric='haversine')

    # °Å¸® °è»ê ÇÔ¼ö Á¤ÀÇ
    def add_nearest_school_distance(data):
        unique_coords = data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        house_coords = np.radians(unique_coords.values)

        # °¡Àå °¡±î¿î ÇĞ±³±îÁöÀÇ °Å¸® °è»ê (¹ÌÅÍ ´ÜÀ§·Î º¯È¯)
        unique_coords['nearest_elementary_distance'] = elementary_tree.query(house_coords, k=1)[0].flatten() * 6371000
        unique_coords['nearest_middle_distance'] = middle_tree.query(house_coords, k=1)[0].flatten() * 6371000
        unique_coords['nearest_high_distance'] = high_tree.query(house_coords, k=1)[0].flatten() * 6371000

        data = data.merge(unique_coords, on=['latitude', 'longitude'], how='left')

        return data

    # ÈÆ·Ã µ¥ÀÌÅÍ¿¡ °Å¸® Ãß°¡
    train_data = add_nearest_school_distance(train_data)
    valid_data = add_nearest_school_distance(valid_data)
    test_data = add_nearest_school_distance(test_data)

    return train_data, valid_data, test_data

def weighted_subway_distance(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subwayInfo = Directory.subway_info

    # È¯½Â¿ª °¡ÁßÄ¡ ºÎ¿©
    duplicate_stations = subwayInfo.groupby(['latitude', 'longitude']).size().reset_index(name='counts')
    transfer_stations = duplicate_stations[duplicate_stations['counts'] > 1]

    subwayInfo = subwayInfo.merge(transfer_stations[['latitude', 'longitude', 'counts']], 
                                  on=['latitude', 'longitude'], 
                                  how='left')
    subwayInfo['weight'] = subwayInfo['counts'].fillna(1)  # È¯½Â¿ªÀº °¡ÁßÄ¡ > 1, ³ª¸ÓÁö´Â 1

    subway_tree = BallTree(np.radians(subwayInfo[['latitude', 'longitude']]), metric='haversine')

    # °Å¸® °è»ê ÇÔ¼ö Á¤ÀÇ
    def add_weighted_subway_distance(data):
        unique_coords = data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        house_coords = np.radians(unique_coords.values)

        distances, indices = subway_tree.query(house_coords, k=1)
        unique_coords['nearest_subway_distance'] = distances.flatten() * 6371000 

        weights = subwayInfo.iloc[indices.flatten()]['weight'].values

        # °Å¸®¸¦ °¡ÁßÄ¡·Î ³ª´©±â
        unique_coords['nearest_subway_distance'] /= weights  

        data = data.merge(unique_coords, on=['latitude', 'longitude'], how='left')

        return data

    # °¢ µ¥ÀÌÅÍ¼Â¿¡ ´ëÇØ °Å¸® Ãß°¡
    train_data = add_weighted_subway_distance(train_data)
    valid_data = add_weighted_subway_distance(valid_data)
    test_data = add_weighted_subway_distance(test_data)

    return train_data, valid_data, test_data
