from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
from sklearn.base import TransformerMixin, BaseEstimator



class Conv2DDenseLayer(layers.Conv2DLayer):

  def __init__(self, incoming, num_units,
               W=init.GlorotUniform(),
               b=init.Constant(0.),
               nonlinearity=nonlinearities.rectify,
               **kwargs):
    num_filters = num_units
    filter_size = kwargs.get("filter_size", incoming.output_shape[2:])
    if "filter_size" in kwargs:
        del kwargs["filter_size"]
    super(Conv2DDenseLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)


# source : https://gist.github.com/duschendestroyer/5170087
class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = linearize(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = linearize(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

