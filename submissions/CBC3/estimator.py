from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from catboost import CatBoost, CatBoostClassifier


def compute_rolling_std(X_df, feature, time_window, center=True):
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

def compute_rolling_mean(X_df, feature, time_window, center=True):
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

def compute_rolling_variables(X_df, feature, time_window, center=True):
    X_df = compute_rolling_mean(X_df, feature, time_window, center)
    X_df = compute_rolling_std(X_df, feature, time_window, center)
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

        for i in ["1h", "2h", "4h", "6h", "12h", "24h", "48h"]:
            for j in ["B", "Beta", "RmsBob", "Vx", "Vth"]:
                X = compute_rolling_variables(X, j, i)
                X = X.copy()

        return X


def get_estimator():

    feature_extractor = FeatureExtractor()

    classifier = CatBoostClassifier(iterations=300,
                                    depth=5,
                                    l2_leaf_reg=3,
                                    loss_function='Logloss',
                                    auto_class_weights='SqrtBalanced',
                                    learning_rate=0.05,
                                    logging_level='Silent',
                                    task_type="CPU")
    

    pipe = make_pipeline(
        feature_extractor,
        classifier)
    return pipe
