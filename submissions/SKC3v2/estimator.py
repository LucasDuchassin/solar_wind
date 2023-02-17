from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
import pandas as pd


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
    name = "_".join([feature, time_window, "std", str(center)])
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
    name = "_".join([feature, time_window, "mean", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def compute_rolling_variables(X_df, feature, time_window, center=True):
    X_df = compute_rolling_mean(X_df, feature, time_window, True)
    X_df = compute_rolling_std(X_df, feature, time_window, True)
    X_df = compute_rolling_mean(X_df, feature, time_window, False)
    X_df = compute_rolling_std(X_df, feature, time_window, False)
    return X_df

def clip_column(X_df, column, min, max):
    X_df[column] = X_df[column].clip(min, max)
    return X_df

def smoothingroll(y, factor):
    y2 = y.copy()
    i=factor
    factor = factor
    while i < (y.shape[0]+1-factor):
        y2[:,0][i] = y[:,0][range(i-factor,i+factor)].mean()
        y2[:,1][i] = y[:,1][range(i-factor,i+factor)].mean()
        i+=1
    return y2


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 250)
        X = clip_column(X, 'Np_nl', 0, 100)
        X = clip_column(X, 'Np', 0, 500)

        Cols = ["B", "Beta", "RmsBob", "Vx", "Range F 9"]

        X = X.drop(columns=[col for col in X if col not in Cols])

        for i in Cols:
            for j in ["1h", "2h", "6h", "12h", "24h"]:
                X = compute_rolling_variables(X, i, j)
            X = X.copy()

        return X

class CustomClf(BaseEstimator):
    def __init__(self):
        self._estimator_type = "classifier"
        self.estimator = VotingClassifier(
            [
            ('LR', LogisticRegression(max_iter=5000,
                                    class_weight={0:1, 1:1.8},
                                    n_jobs=1,
                                    tol=0.005,
                                    penalty='l2',
                                    C=1,
                                    solver='liblinear')),
            ('LR2', LogisticRegression(max_iter=5000))
            ],
            voting='soft',
            n_jobs=-1
        )
    

    def fit(self, X, y):
        return self.estimator.fit(X, y=y)
    
    def predict_proba(self, X):
        y = self.estimator.predict_proba(X)
        y = smoothingroll(y, 24)
        return y
    
    def classes_(self):
        return self.estimator.classes_
    
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    

def get_estimator():

    feature_extractor = FeatureExtractor()

    scaler = StandardScaler()


    classifier = CustomClf()

    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe
