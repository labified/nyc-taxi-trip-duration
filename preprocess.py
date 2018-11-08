import pandas as pd

class Preprocess():
  def __init__(self, X):
    self.X = X

  def clean(self):
    X = self.X.copy()
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
    X = X[((topd(X['dropoff_datetime']) - topd(X['pickup_datetime'])) <= pd.Timedelta(2, 'h')) &
          ((topd(X['dropoff_datetime']) - topd(X['pickup_datetime'])) >= pd.Timedelta(10, 's'))]
    
    self.X = X
    return self

  def add_routes(self, *routes):
    self.X = pd.merge(self.X, pd.concat(routes).rename({'ID': 'id'}, axis='columns'), on='id')
    return self

  def add_epoch(self):
    X = self.X.copy()
    #Use epoch for DateTimes
    X.loc[:, 'pickup_datetime_epoch'] = topd(X['pickup_datetime']).map(lambda dt: int(dt.value / 1000000000))
    X.loc[:, 'dropoff_datetime_epoch'] = topd(X['dropoff_datetime']).map(lambda dt: int(dt.value / 1000000000))

    self.X = X
    return self

  def transform(self):
    return self.X


def topd(x):
  return pd.to_datetime(x, infer_datetime_format=True)