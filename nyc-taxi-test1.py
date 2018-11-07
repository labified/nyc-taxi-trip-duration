import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

import Transformers

train = pd.read_csv('./data/train_100k.csv')

train = Transformers.clean(train)

#Create data
X = train[['passenger_count', 
           'pickup_longitude',
           'pickup_latitude',
           'dropoff_longitude',
           'dropoff_latitude',
           'pickup_datetime_epoch']]

y = train['trip_duration']

#Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X, y)
np.shape(X_scaled)
X_scaled[:5]

#Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.33, random_state=42)

#Train model
model = LinearRegression().fit(X_train, y_train)
model.get_params()

#Evaluate model
model.score(X_test, y_test)

y_pred = model.predict(X_test)

plt.subplot(2, 1, 1)
plt.scatter(np.arange(0, len(y_pred)), y_pred)
plt.subplot(2, 1, 2)
plt.scatter(np.arange(0, len(y_test)), y_test)
plt.show()

mean_squared_log_error(y_test, y_pred)