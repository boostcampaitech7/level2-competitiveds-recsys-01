import features

def feat1(train_data, valid_data, test_data):
    train_data, valid_data, test_data = features.create_clustering_target(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_nearest_subway_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_subway_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_nearest_park_distance(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_school_within_radius(train_data, valid_data, test_data)
    train_data, valid_data, test_data = features.create_sum_park_area_within_radius(train_data, valid_data, test_data)
    return train_data, valid_data, test_data