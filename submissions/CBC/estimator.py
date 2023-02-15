from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from catboost import CatBoost, CatBoostClassifier


def compute_rolling_std(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "std"])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def compute_rolling_mean(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the mean over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "mean"])
    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def clip_column(X_df, column, min, max):
    X_df[column] = X_df[column].clip(min, max)
    return X_df


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 250)
        X = clip_column(X, 'Np_nl', 0, 100)
        X = clip_column(X, 'Np', 0, 500)

        X = compute_rolling_std(X, "B", "2h")
        X = compute_rolling_mean(X, "B", "2h")
        X = compute_rolling_std(X, 'Beta', '2h')
        X = compute_rolling_mean(X, 'Beta', '2h')
        X = compute_rolling_std(X, 'RmsBob', '2h')
        X = compute_rolling_mean(X, 'RmsBob', '2h')
        X = compute_rolling_std(X, 'Vx', '2h')
        X = compute_rolling_mean(X, 'Vx', '2h')

        X = compute_rolling_std(X, "B", "1h")
        X = compute_rolling_mean(X, "B", "1h")
        X = compute_rolling_std(X, 'Beta', '1h')
        X = compute_rolling_mean(X, 'Beta', '1h')
        X = compute_rolling_std(X, 'RmsBob', '1h')
        X = compute_rolling_mean(X, 'RmsBob', '1h')
        X = compute_rolling_std(X, 'Vx', '1h')
        X = compute_rolling_mean(X, 'Vx', '1h')

        X = compute_rolling_std(X, "B", "6h")
        X = compute_rolling_mean(X, "B", "6h")
        X = compute_rolling_std(X, 'Beta', '6h')
        X = compute_rolling_mean(X, 'Beta', '6h')
        X = compute_rolling_std(X, 'RmsBob', '6h')
        X = compute_rolling_mean(X, 'RmsBob', '6h')
        X = compute_rolling_std(X, 'Vx', '6h')
        X = compute_rolling_mean(X, 'Vx', '6h')

        X = compute_rolling_std(X, "B", "12h")
        X = compute_rolling_mean(X, "B", "12h")
        X = compute_rolling_std(X, 'Beta', '12h')
        X = compute_rolling_mean(X, 'Beta', '12h')
        X = compute_rolling_std(X, 'RmsBob', '12h')
        X = compute_rolling_mean(X, 'RmsBob', '12h')
        X = compute_rolling_std(X, 'Vx', '12h')
        X = compute_rolling_mean(X, 'Vx', '12h')
        return X


def get_estimator():

    feature_extractor = FeatureExtractor()

    scaler = StandardScaler()

    classifier = CatBoostClassifier(iterations=200,
                                    depth=4,
                                    l2_leaf_reg=3,
                                    loss_function='Logloss',
                                    learning_rate=0.05,
                                    verbose=50,
                                    task_type="GPU")
    

    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe
