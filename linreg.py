import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from preprocess import Preprocess

pipeline = make_pipeline(
  StandardScaler(),
  LinearRegression()
)

trips = (
  Preprocess(pd.read_csv('./data/train.csv'))
    .clean()
    .add_routes(pd.read_csv('./data/routes_1.csv'), pd.read_csv('./data/routes_2.csv'))
    .add_epoch()
    .transform()
)
X = trips[['Total_distance']]
y = trips['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

model = pipeline.fit(X_train, y_train)
model.score(X_test, y_test)
y_pred = model.predict(X_test)
mean_squared_log_error(y_test, y_pred)

def rmsle(y_pred, y_true):
  assert len(y_pred) == len(y_true)
  return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true), 2)))

rmsle(y_pred, y_test)

plt.figure(figsize=[15, 8])
plt.scatter(X_test, y_test, s=1)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()

