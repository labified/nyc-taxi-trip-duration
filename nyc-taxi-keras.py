import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('./data/train.csv')

#Fix data types
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'],infer_datetime_format=True)
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'],infer_datetime_format=True)
train['store_and_fwd_flag'] = train['store_and_fwd_flag'].astype('category')

train.info()
train.head()

# Remove invalid coordinates
#y: latitude min -> max: 40.495397 - 40.916060      south -> north
#x: longitude min -> max: -74.256217 - -73.699297   west -> east
LAT_MIN = 40.495397
LAT_MAX =  40.916060
LONG_MIN = -74.256217
LONG_MAX = -73.699297

plt.scatter(train['pickup_longitude'], train['pickup_latitude'])
plt.show()

plt.scatter(train['dropoff_longitude'], train['dropoff_latitude'])
plt.show()

#1458644

train = train[(train['pickup_longitude'] >= LONG_MIN) & (train['pickup_longitude'] <= LONG_MAX)]
#1458437

train = train[(train['pickup_latitude'] >= LAT_MIN) & (train['pickup_latitude'] <= LAT_MAX)]
#1458338

train = train[(train['dropoff_longitude'] >= LONG_MIN) & (train['dropoff_longitude'] <= LONG_MAX)]
#1457710

train = train[(train['dropoff_latitude'] >= LAT_MIN) & (train['dropoff_latitude'] <= LAT_MAX)]
#1457301


#Remove invalid trip durations
train = train[(train['dropoff_datetime'] - train['pickup_datetime']) <= pd.Timedelta(2, 'h')]
#1455060

