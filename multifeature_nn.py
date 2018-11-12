import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline, Pipeline
from preprocess import Preprocess
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# create a function that returns a Keras model, taking as parameters things you
# want to verify using cross-valdiation and model selection
adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

def create_model():
  # create model
  model = Sequential()
  model.add(Dense(3, kernel_initializer='normal', activation='relu'))
  model.add(Dense(10, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', epochs=5, optimizer=adam)
  return model

# wrap the model using the function you created
clf = KerasRegressor(build_fn=create_model,verbose=1)

# Set up pipeline with Keras model
pipeline = Pipeline([
  ('standardscaler', StandardScaler()),
  ('kerasregressor', clf)
]
)

trips = (
  Preprocess(pd.read_csv('./data/train.csv'))
    .clean()
    .add_routes(pd.read_csv('./data/routes_1.csv'), pd.read_csv('./data/routes_2.csv'))
    .add_epoch()
    .transform()
)
X = np.array(trips[['Total_distance','pickup_latitude','pickup_longitude']])
y = np.array(trips['trip_duration'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

# set up Tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

#Regression fit and predict using sklearn
pipeline.set_params(kerasregressor__batch_size = 32, kerasregressor__callbacks=[tbCallBack])
model = pipeline.fit(X_train, y_train)
model.score(X_test, y_test)
y_pred = model.predict(X_test)

def rmsle(y_pred, y_true):
  assert len(y_pred) == len(y_true)
  return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true), 2)))

print(rmsle(y_pred, y_test))

plt.figure()
plt.scatter(y_pred, y_test, s=1)
#plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()

