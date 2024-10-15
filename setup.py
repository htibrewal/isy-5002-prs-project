from sklearn.preprocessing import LabelEncoder


base_folder = '/Users/harsh/Desktop/Parking CSV Data/2023'

def get_X_y_encoded(resultant_df, target='car_park_number'):
    X = resultant_df.drop(target, axis=1)
    y = resultant_df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded
