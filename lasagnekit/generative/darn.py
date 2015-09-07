import matplotlib as mpl
mpl.use('Agg')
import theano
import theano.tensor as T
from lasagne import init, nonlinearities
import numpy as np
from lasagne import layers
from theano.tensor.shared_randomstreams import RandomStreams

from theano.sandbox import rng_mrg
class Layer(layers.Layer):

    def __init__(self, incoming, num_units,
                 weights=init.Uniform(), 
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid,
                 **kwargs):

        super(Layer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.incoming = incoming
        self.num_units = num_units
        #if weights is None:
        #    weights = [init.Uniform()] * num_units
        self.weights = self.add_param(weights, (num_inputs + self.num_units, num_inputs + self.num_units), name="W")
        #for i in range(num_units):
        #    self.weights.append(self.add_param(weights[i], (num_inputs + i,), name="W{0}".format(i)))
        #self.weights = tuple(self.weights)
        self.b = self.add_param(b, (num_units,), name="b") 
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
    
    def get_output_for(self, input, **kwargs):
        mode = kwargs.get("mode", "evaluate")
        on = kwargs.get("on")
        rng = kwargs.get("rng", RandomStreams())
        if mode == "evaluate":
            assert on is not None

        if input.ndim > 2:
            input = input.flatten(2)

        if mode == "evaluate":
            return self.evaluate(input, on)
        elif mode == "sample":
            return self.sample(input, rng)

    def evaluate(self, input, on):
        def fn_evaluate(i, W_i, b_i, x, h):
            h_x = T.concatenate((h[:, 0:i], x), axis=1)
            output = self.nonlinearity(T.dot(h_x, W_i[0:i + x.shape[1]]) + b_i)
            return (output ** h[:, i]) * (1 - output) ** (1 - h[:, i])

        sequences = [
                T.arange(self.num_units),
                self.weights,
                self.b,
        ]
        outputs_info = [
        ]
        non_sequences = [
                input,
                on
        ]

        result, updates = theano.scan(fn_evaluate, sequences=sequences,
                                      outputs_info=outputs_info,
                                      non_sequences=non_sequences,
                                      strict=True)
        return result.T, updates
    
    def sample(self, input, rng, nb_samples=1):
        def fn_sample(i, W_i, b_i, unif, R):
            I = i + T.prod(input.shape[1:])
            output = T.dot(R[:, 0:I], W_i[0:I]) + b_i
            h_cur = self.nonlinearity(output)
            h_cur = 1. * (unif < h_cur)
            #R +=
            #R = T.inc_subtensor(R[:, input.shape[1] + i], h_cur)
            return T.set_subtensor(R[:, input.shape[1] + i], h_cur)

        unif_samples = rng.uniform(size=(input.shape[0], self.num_units)).T
        sequences = [
                T.arange(self.num_units),
                self.weights,
                self.b,
                unif_samples
        ]
        #h0 = T.alloc(np.cast[theano.config.floatX](0.), input.shape[0], input.shape[1])
        #h0 += input
        R = T.alloc(np.cast[theano.config.floatX](0.), input.shape[0], input.shape[1] + self.num_units)
        R = T.set_subtensor(R[:, 0:input.shape[1]], input)
        outputs_info = [
            R
        ]
        non_sequences = [
        ]
        result, updates = theano.scan(fn_sample, sequences=sequences,
                                       outputs_info=outputs_info,
                                       non_sequences=non_sequences,
                                       strict=True)
        return result[-1][:, input.shape[1]:], updates
 


from lasagnekit.generative.capsule import Capsule
import matplotlib.pyplot as plt
from lasagnekit.misc.plot_weights import grid_plot
from lasagnekit.easy import get_2d_square_image_view, BatchOptimizer, LightweightModel
from collections import OrderedDict
from lasagne import updates
def test():
    state = 10
    rng = rng_mrg.MRG_RandomStreams(seed=state)
    x_dim = 64
    h_dim = 100
    
    x_in = layers.InputLayer((None, x_dim))
    l_out = Layer(x_in, num_units=h_dim)
    model_encoder = LightweightModel([x_in], [l_out])

    h_in = layers.InputLayer((None, h_dim))
    l_out = Layer(h_in, num_units=x_dim)
    model_decoder = LightweightModel([h_in], [l_out])

    h_in = layers.InputLayer((None, 1))
    l_out = Layer(h_in, num_units=h_dim)
    model_prior = LightweightModel([h_in], [l_out])

    def loss_function(model, tensors):
        X = tensors["X"]
        (h, u), = model_encoder.get_output(X, mode="sample", rng=rng)
        #print(X.ndim, h.ndim)
        (q_h_given_x, _), = model_encoder.get_output(X, mode="evaluate", on=h)
        (p_x_given_h, _),  = model_decoder.get_output(h, mode="evaluate", on=X)
        ones = T.alloc(np.cast[theano.config.floatX](1.), *h.shape)
        zeros = T.alloc(np.cast[theano.config.floatX](0.), *h.shape)
        (p_h, _), = model_prior.get_output(zeros[:, 0:1], mode="evaluate", on=ones)
        L = -((T.log(p_x_given_h).sum(axis=1) + T.log(p_h).sum(axis=1) - T.log(q_h_given_x).sum(axis=1)))
        return (L.mean()), u

    input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
    )
    
    functions = dict(
    )

    # sample function
    nb = T.iscalar()
    sample_input = T.alloc(np.cast[theano.config.floatX](0.), nb, 1) 
    from theano.updates import OrderedUpdates
    u = OrderedUpdates()
    (s_h, u_h), = model_prior.get_output(sample_input, mode="sample", rng=rng)
    ones = T.alloc(np.cast[theano.config.floatX](1.), s_h.shape[0], x_dim)
    (s_x, u_x), = model_decoder.get_output(s_h, mode="evaluate", on=ones)
    sample = theano.function([nb], s_x, updates=u_x)
    batch_optimizer = BatchOptimizer(verbose=1, max_nb_epochs=100,
            batch_size=256,
            optimization_procedure=(updates.rmsprop, {"learning_rate": 0.0001})
    )
    class Container(object):
        def __init__(self, models):
            self.models = models
        def get_all_params(self):
            return [p for m in self.models for p in m.get_all_params()]
    models = [model_encoder, model_decoder]
    models = Container(models)
    darn = Capsule(input_variables, 
                   models,
                   loss_function,
                   functions=functions,
                   batch_optimizer=batch_optimizer)

    from sklearn.datasets import load_digits
    def build_digits():
        digits = load_digits()
        X = digits.data
        X = X.astype(np.float32) / 16.
        return X, digits.images.shape[1:]

    X, imshape = build_digits()
    darn.fit(X=X)

    s=(sample(20))
    s = get_2d_square_image_view(s)
    grid_plot(s, imshow_options={"cmap": "gray"})
    plt.savefig("out.png")

def simple_test():
    x_in = layers.InputLayer((None, 10))
    a = Layer(x_in, 15)
    
    h_batch = T.matrix()
    X_batch = T.matrix()

    o = a.get_output_for(X_batch, mode="evaluate", on=h_batch)
    s = a.get_output_for(X_batch, mode="sample")

    f = theano.function([X_batch, h_batch], o)
    sample = theano.function([X_batch], s)

    X = np.random.normal(size=(100, 10))
    h = np.random.normal(size=(100, 15))
    X = X.astype(np.float32)
    h = h.astype(np.float32)
    
    print(sample(X).shape)
    print(f(X , h).shape)

if __name__ == "__main__":
    test()
