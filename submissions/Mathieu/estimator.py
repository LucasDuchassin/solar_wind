from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
 
 
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
 
 
class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self
 
    def transform(self, X):
        Cols = ["B", "Beta", "RmsBob", "Vth", "Pdyn", "V"]
        X = X.drop(columns=[col for col in X if col not in Cols])
        for i in Cols:
            for j in ["2h", "6h", "12h", "24h"]:
                X = compute_rolling_mean(X, i, j, True)
                X = compute_rolling_mean(X, i, j, False)
                X = compute_rolling_std(X, i, j, True)
                X = compute_rolling_std(X, i, j, False)
            X = X.copy()
        return X
    
 
def get_estimator():
 
    feature_extractor = FeatureExtractor()
    classifier = LGBMClassifier(objective='binary',
                                learning_rate=0.02,
                                n_estimators=300,
                                class_weight={0:1, 1:2})
    
 
    pipe = make_pipeline(
        feature_extractor,
        classifier)
    return pipe