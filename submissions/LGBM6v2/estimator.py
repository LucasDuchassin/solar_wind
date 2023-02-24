from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
import numpy as np
 
 
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
 
def compute_rolling_median(X_df, feature, time_window, center=True):
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
    name = "_".join([feature, time_window, "median", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).median()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_variables(X_df, feature, time_window, center=True):
    X_df = compute_rolling_mean(X_df, feature, time_window, True)
    X_df = compute_rolling_std(X_df, feature, time_window, True)
    X_df = compute_rolling_mean(X_df, feature, time_window, False)
    X_df = compute_rolling_std(X_df, feature, time_window, False)
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
 
def compute_rolling_diff(X, feat, periods):
    name = "_".join([feat, str(periods)])
    X[name] = X[feat].pct_change(periods=periods)
    X[name] = X[name].ffill().bfill()
    X[name] = X[name].astype(X[feat].dtype)
    return X
 
def smoothing(y, factor):
    i=0
    factor = factor
    while i < (y.shape[0]+1-factor):
        if y[:,0][range(i,i+factor)].mean() > 0.5:
            y[:,0][range(i,i+factor)] = 0.99 + 10**-15
            y[:,1][range(i,i+factor)] = 0.01 - 10**-15
        else:
            y[:,0][range(i,i+factor)] = 0.01 - 10**-15
            y[:,1][range(i,i+factor)] = 0.99 + 10**-15
        i+=factor
    return y
 
def smoothing2(y, factor):
    i=0
    factor = factor
    while i < (y.shape[0]+1-factor):
        y[:,0][range(i,i+factor)] = y[:,0][range(i,i+factor)].mean()
        y[:,1][range(i,i+factor)] = y[:,1][range(i,i+factor)].mean()
        i+=factor
    return y
 
def smoothingp(y, factor):
    i=0
    factor = 18
    while i < (y.shape[0]+1-factor):
        y[range(i, i+factor)] = 0
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
    kernel = np.ones(factor)/factor
    kernel = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
 
    kernel = 10*[1] + 10*[3] + 10*[5] + 10*[3] + 10*[1]
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
        X = clip_column(X, 'Beta', 0, 500)
        X = clip_column(X, "B", 0, 1000)
        X = clip_column(X, "RmsBob", 0, 20)
        X = clip_column(X, "Vth", 0, 200)
        X = clip_column(X, "Vx", 0, 500)
        X = clip_column(X, "Np", 0, 100)
        X = clip_column(X, "Range F 13", 0, 10**9)
        X = clip_column(X, "Pdyn", 0, 5e-13)
        Cols = ["B", "Beta", "RmsBob", "V", "Vth", "Range F 13", "Pdyn"]

        X = X.drop(columns=[col for col in X if col not in Cols])

        X = compute_rolling_median(X, "Pdyn", "6h", True)
        X = compute_rolling_median(X, "Pdyn", "12h", True)
        X = compute_rolling_median(X, "Pdyn", "24h", True)
 
        X = compute_rolling_std(X, "B", "24h", True)
        X = compute_rolling_std(X, "B", "24h", False)
 
        X = compute_rolling_mean(X, "RmsBob", "24h", True)
        X = compute_rolling_mean(X, "RmsBob", "24h", False)
 
        X = compute_rolling_mean(X, "RmsBob", "12h", True)
 
        X = compute_rolling_mean(X, "B", "24h", True)
 
        X = compute_rolling_mean(X, "Beta", "2h", True)
 
        X = compute_rolling_mean(X, "Beta", "1h", True)
        X = compute_rolling_mean(X, "B", "1h", True)
        X = compute_rolling_mean(X, "RmsBob", "1h", True)
 
        X = compute_rolling_mean(X, "RmsBob", "6h", True)
 
        X = compute_rolling_mean(X, "Beta", "12h", True)
        X = compute_rolling_mean(X, "Beta", "12h", False)
 
        X = compute_rolling_mean(X, "Beta", "6h", True)
        X = compute_rolling_mean(X, "Beta", "6h", False)
 
        X = compute_rolling_mean(X, "Beta", "24h", True)
        X = compute_rolling_mean(X, "Beta", "24h", False)
 
        X = compute_rolling_std(X, 'B', "48h", False)
        X = compute_rolling_std(X, 'B', "48h", True)
 
        X = compute_rolling_mean(X, "RmsBob", "48h", True)
 
        X = compute_rolling_std(X, 'B', "48h", False)
 
        X = compute_rolling_std(X, 'B', "72h", False)
 
        X = compute_rolling_mean(X, "RmsBob", "18h", True)
 
        X = compute_rolling_mean(X, "RmsBob", "12h", True)
        X = compute_rolling_mean(X, "RmsBob", "12h", False)
 
        X = compute_rolling_mean(X, "Beta", "3h", True)
        X = compute_rolling_mean(X, "Beta", "4h", True)
        X = compute_rolling_mean(X, "Beta", "5h", True)
 
        X = compute_rolling_median(X, "RmsBob", "24h", True)
        X = compute_rolling_median(X, "RmsBob", "24h", False)
        X = compute_rolling_median(X, "B", "24h", True)
        X = compute_rolling_median(X, "B", "24h", False)
        X = compute_rolling_median(X, "Beta", "24h", True)
        X = compute_rolling_median(X, "Beta", "24h", False)
 
        X = compute_rolling_median(X, "RmsBob", "12h", True)
        X = compute_rolling_median(X, "RmsBob", "12h", False)
        X = compute_rolling_median(X, "B", "12h", True)
        X = compute_rolling_median(X, "B", "12h", False)
        X = compute_rolling_median(X, "Beta", "12h", True)
        X = compute_rolling_median(X, "Beta", "12h", False)
 
        X = compute_rolling_diff(X, 'B_12h_median_True', 48)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 48)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 48)
        X = compute_rolling_diff(X, 'B_12h_median_True', 96)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 96)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 96)
        X = compute_rolling_diff(X, 'B_12h_median_True', 192)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 192)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 192)
 
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -24)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 48)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -48)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 96)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -96)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 192)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -192)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -12)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', -6)
 
        X = compute_rolling_diff(X, 'B_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -24)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 48)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -48)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 96)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -96)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 192)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -192)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -12)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'B_1h_mean_True', -6)
 
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -24)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 48)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -48)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 96)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -96)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 192)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -192)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -12)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', -6)
 
        X = X.copy()
 
        X = compute_rolling_mean(X, "V", "1h", True)
        X = compute_rolling_mean(X, "V", "6h", True)
        X = compute_rolling_mean(X, "V", "12h", True)
        X = compute_rolling_mean(X, "V", "24h", True)
        X = compute_rolling_std(X, "V", "1h", True)
        X = compute_rolling_std(X, "V", "6h", True)
        X = compute_rolling_std(X, "V", "12h", True)
        X = compute_rolling_std(X, "V", "24h", True)
 
        X = compute_rolling_diff(X, 'V_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -24)
        X = compute_rolling_diff(X, 'V_1h_mean_True', 48)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -48)
        X = compute_rolling_diff(X, 'V_1h_mean_True', 96)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -96)
        X = compute_rolling_diff(X, 'V_1h_mean_True', 192)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -192)
        X = compute_rolling_diff(X, 'V_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -12)
        X = compute_rolling_diff(X, 'V_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'V_1h_mean_True', -6)
        
        X = compute_rolling_diff(X, 'B_24h_std_True', 48)
        X = compute_rolling_diff(X, 'B_24h_std_True', -48)
        X = compute_rolling_diff(X, 'B_24h_std_True', 96)
        X = compute_rolling_diff(X, 'B_24h_std_True', -96)
        return X
    
class CustomClf(BaseEstimator):
    def __init__(self):
        self._estimator_type = "classifier"
        self.estimator = VotingClassifier(
            [
            ('LR', LogisticRegression(max_iter=5000,
                                    class_weight={0:1, 1:1.5})),
            ('LGBM', LGBMClassifier(objective='binary',
                                num_leaves=20,
                                min_split_gain=0.,
                                max_depth=15,
                                learning_rate=0.02,
                                n_estimators=400,
                                class_weight={0:1, 1:1.8},
                                reg_lambda=0.5,
                                ))
            ],
            voting='soft',
            n_jobs=-1,
            weights=[1,3]
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
    scaler = StandardScaler()
    
 
    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe