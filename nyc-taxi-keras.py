from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard
from keras.optimizers import Nadam
from keras.constraints import maxnorm

from preprocess import Preprocess
from gridsearch import GridSearchCVKeras

def baseline_model(optimizer=Nadam, learn_rate=0.01, kernel_initializer='normal', activation='relu', dropout_rate=0.5, weight_constraint=5, units=10):
  model = Sequential()
  model.add(Dense(units=units, input_dim=8, activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=maxnorm(weight_constraint)))
  model.add(Dropout(dropout_rate))
  model.add(Dense(units=1, kernel_initializer=kernel_initializer))
  model.compile(loss='mean_squared_error', optimizer=optimizer(lr=learn_rate))
  return model

def rmsle(y_true, y_pred):
  assert len(y_pred) == len(y_true)
  return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true), 2)))

trips = (
  Preprocess(pd.read_csv('./data/train.csv'))
    .clean()
    .add_routes(pd.read_csv('./data/routes_1.csv'), pd.read_csv('./data/routes_2.csv'))
    .add_epoch()
    .transform()
)
X = trips[[
  'pickup_datetime_epoch', 
  'dropoff_datetime_epoch', 
  'passenger_count', 
  'pickup_longitude', 
  'pickup_latitude', 
  'Total_distance',
  'Total_time',
  'Number_of_steps']].values
y = trips['trip_duration'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)

estimator = KerasRegressor(build_fn=baseline_model)
tensorboard = TensorBoard(log_dir='./graph/', histogram_freq=0, write_graph=True, write_images=True)
model = estimator.fit(X_train_scaled, y_train, epochs=5, batch_size=10, verbose=1, callbacks=[tensorboard])
print(rmsle(y_test, estimator.predict(scaler.transform(X_test))))

# gsk = GridSearchCVKeras(KerasRegressor, baseline_model, make_scorer(rmsle, greater_is_better=False))
# gsk_result = gsk.fit(X_scaled, y)

#{
# 'verbose': 1, 
# 'epochs': 25, 
# 'batch_size': 10, 
# 'optimizer': <class 'keras.optimizers.Nadam'>, 
# 'learn_rate': 0.01, 
# 'kernel_initializer': 'normal', 
# 'activation': 'relu', 
# 'dropout_rate': 0.5, 
# 'weight_constraint': 5, 
# 'units': 10
# }



# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train, y_train)



# param_grid = {
#   'units': [1, 5, 10, 15, 20, 25, 30],
#   'activation': []
# }

# GridSearchCV(
#   estimator=KerasRegressor(build_fn=baseline_model(activation, 'normal', 10)),
#   param_grid=param_grid,
#   scoring=
# )

# for activation in ['relu', 'softmax', 'sigmoid']:
#   tensorboard = TensorBoard(log_dir='./graph/' + strftime('%Y%m%d-%H%M%S', gmtime()), histogram_freq=0, write_graph=True, write_images=True)
#   estimator = KerasRegressor(build_fn=baseline_model(activation, 'normal', 10))
#   model = estimator.fit(X_train_scaled, y_train, epochs=10, verbose=1, callbacks=[tensorboard])

#   print('Activation=', activation, ':',  rmsle(y_test, estimator.predict(scaler.transform(X_test))))
