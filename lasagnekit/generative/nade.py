import theano
import theano.tensor as T
from lasagne import init
import numpy as np
from lasagne import layers
from theano.tensor.shared_randomstreams import RandomStreams

class NadeLayer(layers.Layer):

    def __init__(self, incoming,
                 num_units,
                 W=init.GlorotUniform(),
                 V=init.GlorotUniform(),
                 b=init.Constant(0.),
                 c=init.Constant(0.),
                 **kwargs):
        super(NadeLayer, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        self.V = self.add_param(V, (num_inputs, num_units), name="V")
        self.b = self.add_param(b, (num_inputs,), name="b")
        self.c = self.add_param(c, (num_units,), name="c")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        sampler = kwargs.get("sampler", False)
        rng = kwargs.get("rng", RandomStreams())

        if input.ndim > 2:
            # if the input has more than three dimensions, flatten it into a
            # batch of feature vectors of 3dimensions.
            input = input.flatten(2)


        def fn(x_cur, W_i, V_i, b_i, p_v_cur, h_cur):
            A = x_cur
            h_cur_new = h_cur + V_i.dimshuffle('x', 0) * A.dimshuffle(0, 'x')
            p_v_i_1 = T.nnet.sigmoid(b_i + T.dot(T.nnet.sigmoid(h_cur), W_i.dimshuffle(0, 'x')))[:, 0]
            p_v_i = p_v_i_1 ** A  * (1 - p_v_i_1) ** (1 - A)
            return p_v_i, h_cur_new

        def fn_sampler(W_i, V_i, b_i, p_v_cur, s_v_cur, h_cur):
            A = s_v_cur
            p_v_i_1 = T.nnet.sigmoid(b_i + T.dot(T.nnet.sigmoid(h_cur), W_i.dimshuffle(0, 'x')))[:, 0]
            s_v_i = 1. * (rng.uniform(size=(input.shape[0],)) < p_v_i_1)
            h_cur_new = h_cur + V_i.dimshuffle('x', 0) * s_v_i.dimshuffle(0, 'x')
            p_v_i = p_v_i_1
            return p_v_i, s_v_i, h_cur_new

        if sampler is True:
            sequences = []
        else:
            sequences = [input.T]
        sequences.extend([
            self.W,
            self.V,
            self.b
        ])
        h1 = (
            T.alloc(np.cast[theano.config.floatX](0.), input.shape[0], self.num_units) +
            self.c.dimshuffle('x', 0)
        )
        h1 = T.unbroadcast(h1, 1)
        v1 = (
            T.alloc(np.cast[theano.config.floatX](1.), input.shape[0])
        )
        v1 = T.unbroadcast(v1, 0)
        outputs_info = [
            v1,
        ]
        if sampler is True:
            s1 = (
                T.alloc(np.cast[theano.config.floatX](1.), input.shape[0])
            )
            outputs_info.append(s1)
            s1 = T.unbroadcast(s1, 0)
        outputs_info.append(h1)
        if sampler is True:
            func = fn_sampler
        else:
            func = fn
        result, _ = theano.scan(func, sequences=sequences,
                                outputs_info=outputs_info,
                                strict=True)
        if sampler is True:
            return result[0].T
        else:
            return result[0].T

if __name__ == "__main__":

    import matplotlib as mpl
    mpl.use('Agg')

    from lasagnekit.easy import BatchOptimizer, LightweightModel, get_batch_slice, get_nb_batches
    from lasagne import updates
    from sklearn.datasets import load_digits
    from sklearn.utils import shuffle
    from lasagnekit.datasets.mnist import MNIST
    from lasagne.misc.plot_weights import grid_plot
    from lasagnekit.easy import get_2d_square_image_view
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    from lasagnekit.generative.capsule import Capsule

    data = MNIST()
    data.load()
    X = data.X
    X[X > 0.5] = 1.
    X[X <= 0.5] = 0.

    X = shuffle(X)

    order = range(X.shape[1])
    #np.random.shuffle(order)
    X = X[:, order]

    nb_examples = X.shape[0]
    x_dim = X.shape[1]
    h_dim = 500

    x_in = layers.InputLayer((None, x_dim))
    l_out = NadeLayer(x_in, num_units=h_dim)
    model = LightweightModel([x_in], [l_out])

    def loss_function(model, tensors):
        o, = model.get_output(tensors.get("X"))
        return -T.log(o).sum(axis=1).mean()

    input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
    )

    functions = dict(
           sample=dict(
               get_output=lambda model, X: model.get_output(X, sampler=True)[0],
               params=["X"]
           ),
           log_likelihood=dict(
               get_output=lambda model, X: T.log(model.get_output(X)[0]).sum(axis=1),
               params=["X"]
           ),
    )

    batch_optimizer = BatchOptimizer(verbose=1, max_nb_epochs=2,
            batch_size=256,
            optimization_procedure=(updates.rmsprop, {"learning_rate": 0.001})
    )
    nade = Capsule(input_variables, model,
                   loss_function,
                   functions=functions,
                   batch_optimizer=batch_optimizer)
    nade.fit(X=X)
    T = np.ones((100, x_dim)).astype(np.float32)
    T = T[:, order]
    s = get_2d_square_image_view(nade.sample(T))
    grid_plot(s, imshow_options={"cmap": "gray"})
    plt.savefig("out.png")
