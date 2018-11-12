import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard

from preprocess import Preprocess


def baseline_model():
  model = Sequential()
  model.add(Dense(units=1, input_dim=1, activation='relu', kernel_initializer='normal'))
  #model.add(BatchNormalization())
  model.add(Dense(units=1, kernel_initializer='normal'))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

def rmsle(y_pred, y_true):
  assert len(y_pred) == len(y_true)
  return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true), 2)))


trips = (
  Preprocess(pd.read_csv('./data/train.csv'))
    .clean()
    .add_routes(pd.read_csv('./data/routes_1.csv'), pd.read_csv('./data/routes_2.csv'))
    .add_epoch()
    .transform()
)
X = trips[['Total_distance']].values
y = trips['trip_duration'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)
tensorboard = TensorBoard(log_dir='./data', histogram_freq=0, write_graph=True, write_images=True)
estimator = KerasRegressor(build_fn=baseline_model)
model = estimator.fit(X_train_scaled, y_train, verbose=1, callbacks=[tensorboard])

rmsle(estimator.predict(scaler.transform(X_test)), y_test)
