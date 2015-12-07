from lasagne import layers
from lasagne import init, nonlinearities, utils, updates
import theano
import theano.tensor as T

import numpy as np

from lasagnekit.generative.neural_net import NeuralNet
from lasagnekit.easy import BatchOptimizer, LightweightModel

class SimpleRNN(layers.Layer):

    def __init__(self, incoming, num_units,
                Wh=init.Orthogonal(),
                Wx=init.Orthogonal(),
                b=init.Constant(0.),
                nonlinearity=nonlinearities.rectify,
                **kwargs):

        super(SimpleRNN, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[2:]))

        self.Wx = utils.create_param(Wx, (num_inputs, num_units), name="Wx")
        self.Wh = utils.create_param(Wh, (num_units, num_units), name="Wh")
        self.b = utils.create_param(b, (num_units,), name="b") if b is not None else None

    def get_params(self):
        return [self.Wx, self.Wh] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):

        if input.ndim > 3:
            # if the input has more than three dimensions, flatten it into a
            # batch of feature vectors of 3dimensions.
            input = input.flatten(3)


        def fn(inp, y_prev_t, Wh):
            y_t = T.dot(y_prev_t, Wh) + inp
            y_t = self.nonlinearity(y_t)
            return y_t
        x = T.dot(input.dimshuffle(1, 0, 2), self.Wx) + self.b
        sequences = [
                x
        ]
        y0 = T.alloc(np.cast[theano.config.floatX](0.),
                     input.shape[0], self.num_units)
        y0 = T.unbroadcast(y0, 1)
        outputs_info = [
                y0
        ]
        non_sequences = [
                self.Wh
        ]
        result, _ = theano.scan(fn, sequences=sequences,
                                outputs_info=outputs_info,
                                non_sequences=non_sequences)
        return result.dimshuffle(1, 0, 2)
        """
        y = []
        y_t = T.alloc(0., input.shape[0], self.num_units)
        for t in range(20):
            y_t = T.dot(y_t, self.Wh) + T.dot(input[:, t, :], self.Wx)
            if self.bx is not None:
                y_t = y_t + self.bx.dimshuffle('x', 0)
            if self.bh is not None:
                y_t = y_t + self.bh.dimshuffle('x', 0)
            y_t = self.nonlinearity(y_t)
            y.append(T.shape_padleft(y_t, 1))
        Y = T.concatenate(y, axis=0).dimshuffle(1, 0, 2)
        return Y
        """


class LSTM(layers.Layer):

    def __init__(self, incoming, num_units,
                Wxi=init.Orthogonal(),
                Wxf=init.Orthogonal(),
                Wxo=init.Orthogonal(),
                Wxg=init.Orthogonal(),
                Whi=init.Orthogonal(),
                Whf=init.Orthogonal(),
                Who=init.Orthogonal(),
                Whg=init.Orthogonal(),
                bi=init.Constant(0.),
                bf=init.Constant(0.),
                bo=init.Constant(0.),
                bg=init.Constant(0.),
                nonlinearity=nonlinearities.tanh,
                **kwargs):

        super(LSTM, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[2:]))

        self.Wxi = utils.create_param(Wxi, (num_inputs, num_units), name="Wxi")
        self.Wxf = utils.create_param(Wxf, (num_inputs, num_units), name="Wxf")
        self.Wxo = utils.create_param(Wxo, (num_inputs, num_units), name="Wxo")
        self.Wxg = utils.create_param(Wxg, (num_inputs, num_units), name="Wxg")
        self.Whi = utils.create_param(Wxi, (num_units, num_units), name="Whi")
        self.Whf = utils.create_param(Wxf, (num_units, num_units), name="Whf")
        self.Who = utils.create_param(Wxo, (num_units, num_units), name="Who")
        self.Whg = utils.create_param(Wxg, (num_units, num_units), name="Whg")
        self.bi = utils.create_param(bi, (num_units,), name="bi") if bi is not None else None
        self.bf = utils.create_param(bf, (num_units,), name="bf") if bf is not None else None
        self.bo = utils.create_param(bo, (num_units,), name="bo") if bo is not None else None
        self.bg = utils.create_param(bg, (num_units,), name="bg") if bg is not None else None


    def get_params(self):
        return [self.Wxi, self.Wxf, self.Wxo, self.Wxg, self.Whi, self.Whf, self.Who, self.Whg] + self.get_bias_params()

    def get_bias_params(self):
        return [self.bi, self.bf, self.bo, self.bg]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):

        if input.ndim > 3:
            # if the input has more than three dimensions, flatten it into a
            # batch of feature vectors of 3dimensions.
            input = input.flatten(3)


        def fn(ix_t, fx_t, ox_t, gx_t, h_prev_t, c_prev_t, Whi, Whf, Who, Whg):
            i_t = T.nnet.sigmoid(ix_t + T.dot(h_prev_t, Whi))
            f_t = T.nnet.sigmoid(fx_t + T.dot(h_prev_t, Whf))
            o_t = T.nnet.sigmoid(ox_t + T.dot(h_prev_t, Who))
            g_t = self.nonlinearity(gx_t + T.dot(h_prev_t, Whg))
            c_t = f_t * c_prev_t + i_t * g_t
            h_t = o_t * self.nonlinearity(c_t)
            return h_t, c_t

        input_ = input.dimshuffle(1, 0, 2)

        ix = T.dot(input_, self.Wxi) + self.bi
        fx = T.dot(input_, self.Wxf) + self.bf
        ox = T.dot(input_, self.Wxo) + self.bo
        gx = T.dot(input_, self.Wxg) + self.bg
        sequences = [
            ix,
            fx,
            ox,
            gx,
        ]
        h0 = T.alloc(np.cast[theano.config.floatX](0.),
                     input.shape[0], self.num_units)
        h0 = T.unbroadcast(h0, 1)

        c0 = T.alloc(np.cast[theano.config.floatX](0.),
                     input.shape[0], self.num_units)
        c0 = T.unbroadcast(c0, 1)
        outputs_info = [
                h0, c0
        ]
        result, _ = theano.scan(fn, sequences=sequences,
                                outputs_info=outputs_info,
                                non_sequences=[self.Whi, self.Whf, self.Who, self.Whg],
                                strict=True)
        return result[0].dimshuffle(1, 0, 2)

if __name__ == "__main__":
    from sklearn.utils import shuffle

    data = np.load("/home/gridcl/mehdicherti/work/data/paul_graham/paul_graham.npz")
    X = data['X']
    X = X[:, :, :]
    X = X.astype(theano.config.floatX)
    X = shuffle(X)

    X_batch = T.tensor3()
    
    x_dim = X.shape[2]
    t_dim = X.shape[1] - 1

    l_in = layers.InputLayer(shape=(None, t_dim, x_dim))
    h = SimpleRNN(l_in, num_units=800, nonlinearity=nonlinearities.tanh)
    h = SimpleRNN(h, num_units=800, nonlinearity=nonlinearities.tanh)
    h = SimpleRNN(h, num_units=X.shape[2], nonlinearity=nonlinearities.softmax)
    l_out = h
    model = LightweightModel([l_in], [l_out])

    def loss_function(pred, real):
        return ((pred - real) ** 2).sum(axis=(1, 2))

    gen = theano.function([X_batch], l_out.get_output(X_batch))
    # instantiate the model
    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):

            super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            
            if epoch % 10 == 0:
                s = gen(X[0:10, 0:-1, :])
                print(s.shape)
                mapping = data['inverse_mapping'].tolist()
                i = 0
                for a in s:
                    print("pred:")
                    print("".join(map(lambda k:mapping[k], a.argmax(axis=1))))
                    print("real:")
                    print("".join(map(lambda k:mapping[k], X[i, 1:, :].argmax(axis=1))))
                    print("---")
                    i += 1
             
    nnet = NeuralNet(model, loss_function=loss_function,
                     X_type=T.tensor3,
                     y_type=T.tensor3,
                     batch_optimizer=MyBatchOptimizer(verbose=1, max_nb_epochs=1000, batch_size=128,
                                                      optimization_procedure=(updates.rmsprop, {"learning_rate":0.0001})))

    nnet.fit(X[:, 0:-1, :], X[:, 1:, :])
