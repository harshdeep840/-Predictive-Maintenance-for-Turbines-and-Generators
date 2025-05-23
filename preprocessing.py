import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath="/content/turbine_sensor_data.csv"):
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values(by='timestamp')
    features = df[['temperature', 'vibration', 'pressure']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler, df
