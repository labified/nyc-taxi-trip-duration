import numpy as np

from sklearn.model_selection import GridSearchCV

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

class GridSearchCVKeras():
  def __init__(self, create_estimator, create_model, scoring):
    self.create_estimator = create_estimator
    self.create_model = create_model
    self.scoring = scoring
    self.sk_params = {'verbose': 1}

  def fit(self, X, y):
    #bs_e = self.batchsize_epochs(X, y)
    #self.sk_params = {**self.sk_params, **bs_e}

    #optimizer = self.optimization(X, y)
    #self.sk_params = {**self.sk_params, **optimizer}
    
    # lr = self.learning_rate(X, y)
    # self.sk_params = {**self.sk_params, **lr}

    # k_init = self.kernel_initializer(X, y)
    # self.sk_params = {**self.sk_params, **k_init}

    self.sk_params = {**self.sk_params, 'epochs': 25, 'batch_size': 10, 'optimizer': Nadam, 'learn_rate': 0.01, 'kernel_initializer': 'normal'}

    a = self.activation(X, y)
    self.sk_params = {**self.sk_params, **a}

    d = self.dropout(X, y)
    self.sk_params = {**self.sk_params, **d}

    u = self.units(X, y)
    self.sk_params = {**self.sk_params, **u}

    return self.sk_params

  def batchsize_epochs(self, X, y):
    #10, 25
    param_grid = {
      'batch_size': [10, 20, 40, 60, 80, 100],
      'epochs': [25]
    }
    return self.run_gridsearch(param_grid, X, y)

  def optimization(self, X, y):
    #Nadam
    param_grid = {
      'optimizer': [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
    }
    return self.run_gridsearch(param_grid, X, y)

  def learning_rate(self, X, y):
    #0.01
    param_grid = {
      'learn_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    }
    return self.run_gridsearch(param_grid, X, y)

  def kernel_initializer(self, X, y):
    #normal
    param_grid = {
      'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    }
    return self.run_gridsearch(param_grid, X, y)

  def activation(self, X, y):
    #relu
    param_grid = {
      'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    }
    return self.run_gridsearch(param_grid, X, y)

  def dropout(self, X, y):
    #0.5, 5
    param_grid = {
      'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      'weight_constraint': [1, 2, 3, 4, 5]
    }
    return self.run_gridsearch(param_grid, X, y)

  def units(self, X, y):
    #10
    param_grid = {
      'units': [1, 5, 10, 15, 20, 25, 30]
    }
    return self.run_gridsearch(param_grid, X, y)

  def run_gridsearch(self, param_grid, X, y):
    print('Searching for:', param_grid)
    estimator = self.create_estimator(build_fn=self.create_model, **self.sk_params)
    np.random.seed(7)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=self.scoring, cv=3, n_jobs=1)
    print('Using params:', grid.get_params())
    grid_result = grid.fit(X, y)
    print('Best params:', grid_result.best_params_)
    return grid_result.best_params_
