import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard

from preprocess import Preprocess


def baseline_model():
  model = Sequential()
  model.add(Dense(units=1, input_dim=1, activation='relu', kernel_initializer='normal'))
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
trips = trips.loc[:1000]
X = trips[['Total_distance']]
y = trips['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

pipeline = make_pipeline(
  StandardScaler(),
  KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
)
tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
#seed = 7
#np.random.seed(seed)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X_train, y_train, cv=kfold, error_score='raise')
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
pipeline.set_params(kerasregressor__callbacks=[tensorboard])
model = pipeline.fit(X_train, y_train)
rmsle(model.predict(X_test), y_test)
