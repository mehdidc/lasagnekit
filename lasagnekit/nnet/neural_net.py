import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from collections import OrderedDict
from .. import easy

def cross_entropy(y_pred, y, **params):
    return T.nnet.categorical_crossentropy(y_pred, y)


class NeuralNet(object):

    def __init__(self, nnet_x_to_y,
                 batch_optimizer=None, rng=None,
                 loss_function=cross_entropy,
                 loss_params=None,
                 noise_function=None,
                 X_type=T.matrix,
                 y_type=T.ivector):
        self.nnet_x_to_y = nnet_x_to_y

        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        self.batch_optimizer.model = self
        if rng is None:
            rng = RandomStreams(seed=10001)
        self.rng = rng

        if loss_params is None:
            loss_params = dict()
        self.loss_params = loss_params
        self.loss_function = loss_function
        self.noise_function = noise_function

        self.X_type = X_type
        self.y_type = y_type

        self.predict_function = None
        self.get_loss = None

        self.iter_update_batch = None

    def get_state(self):
        return [param.get_value() for param in self.all_params]
    
    def set_state(self, state):
        for cur_param, state_param in zip(self.all_params, state):
            cur_param.set_value(state_param, borrow=True)

    def prepare(self, X, y, optional=None):
        
        if optional is not None:
            optional_tensors = {}
            for k, v in optional.items():
                tensor_type = {
                    1: T.vector,
                    2: T.matrix,
                    3: T.tensor3,
                    4: T.tensor4
                }
                type_ = tensor_type[len(v.shape)]
                optional_tensors[k] = type_('%s_batch' % (k,))

        X_batch = self.X_type('X_batch')
        y_batch = self.y_type('y_batch')

        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           self.batch_optimizer.batch_size)

        if self.noise_function is not None:
            X_batch_noisified = self.noise_function(X_batch)
            X_batch_for_encoder = X_batch_noisified
        else:
            X_batch_for_encoder = X_batch

        y_with_noise, = self.nnet_x_to_y.get_output(X_batch_for_encoder)
        y_without_noise, = self.nnet_x_to_y.get_output(X_batch)

        y_hat_without_noise_pred = T.argmax(y_without_noise, axis=1)
        self.predict_function = theano.function([X_batch],
                                                T.argmax(self.nnet_x_to_y.get_output(X_batch, determnistic=True)[0], axis=1))
        self.predict_proba = theano.function([X_batch],
                                             self.nnet_x_to_y.get_output(X_batch, determnisitic=True)[0])

        self.all_params = (self.nnet_x_to_y.get_all_params())

        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure

        optional_params = OrderedDict()
        optional_params["X_batch"] = X_batch
        optional_params["y_batch"] = y_batch
        optional_params.update(self.loss_params)

        if optional is not None:
            optional_params.update(optional_tensors)
        loss = self.loss_function(y_with_noise, y_batch,
                                  **optional_params).mean()

        L = [X_batch, y_batch]
        if optional is not None:
            L.extend(optional_tensors.values())
        print(optional_tensors.keys())
        self.get_loss = theano.function(L, loss)

        updates = opti_function(loss, self.all_params, **opti_kwargs)

        if self.batch_optimizer.whole_dataset_in_device is True:
            X = theano.shared(X, borrow=True)
            y = theano.shared(y, borrow=True)
            givens = {
                X_batch: X[batch_slice],
                y_batch: y[batch_slice],
            }
            if optional is not None:
                for k in optional.keys():
                    v = optional[k]
                    v = theano.shared(v, borrow=True)
                    givens[optional_tensors[k]] = v[batch_slice]

            iter_update_batch = theano.function(
                [batch_index], loss,
                updates=updates,
                givens=givens
            )
        else:
            L = [X_batch, y_batch]
            if optional is not None:
                L.extend(optional_tensors.values())
            iter_update = theano.function(
                L,
                loss,
                updates=updates
            )

            def iter_update_batch(batch_index):
                sl = slice(batch_index * self.batch_optimizer.batch_size,
                           (batch_index+1) * self.batch_optimizer.batch_size)

                params = [X[sl], y[sl]]
                if optional is not None:
                    for k, v in optional.items():
                        params.append(v[sl])
                return iter_update(*params)
        self.iter_update_batch = iter_update_batch
    
    def fit(self, X, y, optional=None):
        if self.iter_update_batch is None:
            self.prepare(X, y, optional)
        nb_batches = easy.get_nb_batches(X.shape[0],
                                         self.batch_optimizer.batch_size)
        self.batch_optimizer.optimize(nb_batches, self.iter_update_batch)
        return self

    def predict(self, X):
        assert self.predict_function is not None
        return self.predict_function(X)
