from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
from sklearn.base import TransformerMixin, BaseEstimator
import theano.tensor as T
import theano


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


def inv_conv_output_length(input_length, filter_size, stride, pad=0):
    if input_length is None:
        return None
    if pad == 'full':
        output_length = (input_length + 1) * stride + filter_size
    elif pad == 'valid':
        output_length = (input_length - 1) * stride + filter_size
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = (input_length + 2 * pad - 1) * stride + filter_size
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))
    return output_length


class Deconv2DLayer(layers.Conv2DLayer):
    def __init__(self, incoming, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(inv_conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    #def get_W_shape(self):
    #    shape = super(Deconv2DLayer, self).get_W_shape()
    #    return (shape[1], shape[0]) + shape[2:]

    def convolve(self, input, **kwargs):
        shape = self.get_output_shape_for(input.shape)
        fake_output = T.alloc(0., *shape)
        border_mode = 'half' if self.pad == 'same' else self.pad
        
        w_shape = self.get_W_shape()
        w_shape = (w_shape[1], w_shape[0]) + w_shape[2:]
        shape = self.get_output_shape_for(self.input_layer.output_shape)
        W = self.W.transpose((1, 0, 2, 3))

        conved = self.convolution(fake_output, W,
                                  shape, w_shape,
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return theano.grad(None, wrt=fake_output, known_grads={conved: input})


class Depool2DLayer(Deconv2DLayer):

    def __init__(self, incoming, pool_size=(2, 2),
                 pool_stride=(2, 2), **kwargs):
        super(Depool2DLayer, self).__init__(
                incoming,
                num_filters=incoming.output_shape[1],
                filter_size=pool_size,
                stride=pool_stride,
                **kwargs)

if __name__ == "__main__":
    import numpy as np
    l_in = layers.InputLayer((None, 1, 4, 4))
    l_deconv = Deconv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=2)
    print(l_deconv.output_shape)
    l_deconv = Deconv2DLayer(
            l_deconv,
            num_filters=64,
            filter_size=(5, 5),
            stride=2)
    print(l_deconv.output_shape)
    x = T.tensor4()
    f = theano.function([x], layers.get_output(l_deconv, x))
    A = np.random.uniform(size=(5, 1, 4, 4)).astype(np.float32)
    print(f(A).shape)
