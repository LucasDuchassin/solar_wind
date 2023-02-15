from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from keras import layers, models, optimizers, losses, metrics
import pandas as pd
from catboost import CatBoost, CatBoostClassifier
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier


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

def create_model():
    model = models.Sequential()
    model.add(layers.Conv1D(16, 60))
    model.add(layers.Conv1D(16, 30))
    model.add(layers.Conv1D(16, 15, activation='ReLU'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(16, 10))
    model.add(layers.Conv1D(16, 5))
    model.add(layers.Conv1D(16, 3, activation='ReLU'))
    model.add(layers.Flatten())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics.BinaryAccuracy())
    return model

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 250)
        X = clip_column(X, 'Np_nl', 0, 100)
        X = clip_column(X, 'Np', 0, 500)

        for i in ["1h", "2h", "4h", "6h", "12h", "24h", "48h", "72h"]:
            for j in ["B", "Beta", "RmsBob", "Vx", "Vth"]:
                X = compute_rolling_variables(X, j, i)

        return X

def get_estimator():

    feature_extractor = FeatureExtractor()

    scaler = StandardScaler()


    #classifier = LogisticRegression(max_iter=2000)
    classifier = KerasClassifier(create_model, epochs=10, batch_size=32)

    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe
