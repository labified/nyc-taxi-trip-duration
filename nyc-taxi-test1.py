import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from preprocess import Preprocess

train = pd.read_csv('./data/train_with_routes.csv')

plt.figure(figsize=[15, 8])
plt.scatter(train['pickup_longitude'], train['pickup_latitude'], s=1)
plt.show()