import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(config):
    df = pd.read_csv(config["data"]["input_csv"])
    X = df.drop(columns=config["data"]["drop_columns"]).values
    y = df[config["data"]["target_columns"]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_seed"]
    )

    return X_train, X_val, y_train, y_val, scaler_X, scaler_y
