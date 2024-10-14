from sklearn.cluster import KMeans
from utils.constant_utils import Config
import pandas as pd
from geopy.distance import great_circle

def clustering(total_df, info_df, feat_name, n_clusters=20):
    info = info_df[['longitude', 'latitude']].values
    
    kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, n_init=10, random_state=Config.RANDOM_SEED)
    kmeans.fit(info)
    
    clusters = kmeans.predict(total_df[['longitude', 'latitude']].values)
    total_df[feat_name] = pd.DataFrame(clusters, dtype='category')
    return total_df

def distance_gangnam(df):
    gangnam = (37.498095, 127.028361548)

    def calculate_distance(df):
        point = (df['latitude'], df['longitude'])
        distance_km = great_circle(gangnam, point).kilometers
        return distance_km
    
    df['distance_km'] = df.apply(calculate_distance, axis=1)
    df['gangnam_5km'] = (df['distance_km'] <= 5).astype(int)
    df['gangnam_10km'] = (df['distance_km'] <= 7).astype(int)
    df['gangnam_remote'] = (df['distance_km'] > 7).astype(int)
    df.drop(columns=['distance_km'], inplace=True)

    return df