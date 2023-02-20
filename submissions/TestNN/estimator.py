from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras import layers, models, optimizers, losses, metrics, regularizers
import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K



def clip_column(X_df, column, min, max):
    X_df[column] = X_df[column].clip(min, max)
    return X_df

def data_tensor3(X, size):
    Ten=[]
    for i in range (size, X.shape[0]+1-size):
        Ten.append(tf.convert_to_tensor(X.iloc[i-size:i+size]))
        i+=size
    return tf.stack(Ten, axis=0)

def create_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
        except RuntimeError as e:
            print(e)
    K.clear_session()
    model = models.Sequential([
        layers.Conv1D(32, 3, padding='same', activation='relu', input_shape=(2*10, 4), kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.Conv1D(32, 6, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.MaxPool1D(),
        layers.Dropout(0.2),
        layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.Conv1D(64, 6, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.MaxPool1D(),
        layers.Dropout(0.2),
        layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics.BinaryAccuracy())
    return model

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 200)
        X = clip_column(X, 'B', 0, 200)
        X = clip_column(X, 'RmsBob', 0, 3)
        X = clip_column(X, 'V', 0, 2000)
        Cols = ['B', 'Beta', 'RmsBob', 'V']
        X = X.drop(columns=[col for col in X if col not in Cols])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)
        X = data_tensor3(X, 10)

        return X
    
    

def get_estimator():

    feature_extractor = FeatureExtractor()

    scaler = StandardScaler()


    #classifier = LogisticRegression(max_iter=2000)
    classifier = KerasClassifier(create_model, epochs=10, batch_size=1024)

    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe
