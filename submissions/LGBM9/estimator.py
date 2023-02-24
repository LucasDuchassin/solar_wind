from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np
 
 
def compute_rolling_std(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "std", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_mean(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "mean", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_median(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "median", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).median()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def compute_rolling_min(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "median", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).min()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def compute_rolling_max(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "median", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).max()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def compute_rolling_variables(X_df, feature, time_window):
    X_df = compute_rolling_mean(X_df, feature, time_window, True)
    X_df = compute_rolling_std(X_df, feature, time_window, True)
    X_df = compute_rolling_median(X_df, feature, time_window, True)
    X_df = compute_rolling_min(X_df, feature, time_window, True)
    X_df = compute_rolling_max(X_df, feature, time_window, True)
    X_df = compute_rolling_mean(X_df, feature, time_window, False)
    X_df = compute_rolling_std(X_df, feature, time_window, False)
    X_df = compute_rolling_median(X_df, feature, time_window, False)
    X_df = compute_rolling_min(X_df, feature, time_window, False)
    X_df = compute_rolling_max(X_df, feature, time_window, False)
    return X_df
 
def compute_rolling_diff(X, feat, periods):
    name = "_".join([feat, str(periods)])
    X[name] = X[feat].pct_change(periods=periods)
    X[name] = X[name].ffill().bfill()
    X[name] = X[name].astype(X[feat].dtype)
    return X
 
def clip_column(X_df, column, min, max):
    X_df[column] = X_df[column].clip(min, max)
    return X_df
 
def smoothing2(y, factor):
    i=0
    factor = factor
    while i < (y.shape[0]+1-factor):
        y[:,0][range(i,i+factor)] = y[:,0][range(i,i+factor)].mean()
        y[:,1][range(i,i+factor)] = y[:,1][range(i,i+factor)].mean()
        i+=factor
    return y
 
    
def smoothingroll(y, factor):
    y2 = y.copy()
    i=factor
    factor = factor
    while i < (y.shape[0]+1-factor):
        y2[:,0][i] = (y[:,0][range(i-factor,i+factor)].mean())
        y2[:,1][i] = (y[:,1][range(i-factor,i+factor)].mean())
        i+=1
    return y2
 
def smoothingroll2(y, factor):
    kernel = 10*[1] + 10*[3] + 15*[6] + 10*[4] + 10*[2]
    kernel = (kernel/np.sum(kernel))
    y[:,0] = np.convolve(y[:,0], kernel, 'same')
    y[:,1] = np.convolve(y[:,1], kernel, 'same')
    return y
        
def smoothingroll3(y, factor, quantile):
    y = pd.DataFrame(y)
    y[1] = y[1].rolling(factor, min_periods=0, center=True).quantile(quantile)
    y[0] = 1-y[1].ffill().bfill()
    y = y.to_numpy()
    return y
 
class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self
 
    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 100)
        X = clip_column(X, "B", 0, 100)
        X = clip_column(X, "RmsBob", 0, 2)
        X = clip_column(X, "Vth", 0, 350)
        X = clip_column(X, "V", 0, 2000)
        X = clip_column(X, "Vx", -1000, 1000)
        X = clip_column(X, "Range F 13", 0, 10**9)
        X = clip_column(X, "Pdyn", 0, 5e-13)

        Cols = ["B", "Beta", "RmsBob", "V", "Vth", "Range F 13", "Pdyn"]

        X = X.drop(columns=[col for col in X if col not in Cols])

        Cols = ["B", "Beta", "RmsBob", "V", "Pdyn"]

        for col in Cols:
            X = compute_rolling_variables(X, col, "2h")
            X = compute_rolling_variables(X, col, "4h")
            X = compute_rolling_variables(X, col, "8h")
            X = compute_rolling_variables(X, col, "16h")
            X = compute_rolling_variables(X, col, "32h")
            X = X.copy()

        return X
    
class CustomClf(BaseEstimator):
    def __init__(self):
        self._estimator_type = "classifier"
        self.estimator = LGBMClassifier(objective='binary',
                                num_leaves=15,
                                min_split_gain=0.,
                                max_depth=20,
                                learning_rate=0.02,
                                n_estimators=400,
                                class_weight={0:1, 1:1.8},
                                reg_lambda=0.5,
                                )
        
 
    def fit(self, X, y):
        return self.estimator.fit(X, y=y)
    
    def predict(self, X, y):
        y = self.estimator.predict(X)
        #y = smoothingp(y, 6)
        return y
    
    def predict_proba(self, X):
        y = self.estimator.predict_proba(X)
        y = smoothingroll2(y, 30)
        return y
    
    def classes_(self):
        return self.estimator.classes_
    
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    
 
def get_estimator():
 
    feature_extractor = FeatureExtractor()
    classifier = CustomClf()
    
 
    pipe = make_pipeline(
        feature_extractor,
        classifier)
    return pipe