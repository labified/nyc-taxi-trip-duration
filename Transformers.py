import pandas as pd

def clean(X):
  #Fix data types
  X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'],infer_datetime_format=True)
  X['dropoff_datetime'] = pd.to_datetime(X['dropoff_datetime'],infer_datetime_format=True)
  X['store_and_fwd_flag'] = X['store_and_fwd_flag'].astype('category')
  
  # Remove invalid coordinates
  #y: latitude min -> max: 40.495397 - 40.916060      south -> north
  #x: longitude min -> max: -74.256217 - -73.699297   west -> east
  LAT_MIN = 40.495397
  LAT_MAX =  40.916060
  LONG_MIN = -74.256217
  LONG_MAX = -73.699297
  X = X[(X['pickup_longitude'] >= LONG_MIN) & (X['pickup_longitude'] <= LONG_MAX)]
  X = X[(X['pickup_latitude'] >= LAT_MIN) & (X['pickup_latitude'] <= LAT_MAX)]
  X = X[(X['dropoff_longitude'] >= LONG_MIN) & (X['dropoff_longitude'] <= LONG_MAX)]
  X = X[(X['dropoff_latitude'] >= LAT_MIN) & (X['dropoff_latitude'] <= LAT_MAX)]
  
  #Remove invalid trip durations
  X = X[(X['dropoff_datetime'] - X['pickup_datetime']) <= pd.Timedelta(2, 'h')]
  X = X[(X['dropoff_datetime'] - X['pickup_datetime']) >= pd.Timedelta(10, 's')]

  #Use epoch for DateTimes
  X['pickup_datetime_epoch'] = X['pickup_datetime'].map(lambda dt: int(dt.value / 1000000000))
  X['dropoff_datetime_epoch'] = X['dropoff_datetime'].map(lambda dt: int(dt.value / 1000000000))

  return X
