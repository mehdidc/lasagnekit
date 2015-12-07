import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano

from .. import easy

from lasagne import objectives
import numpy as np

class Autoencoder(object):

    def __init__(self,  nnet_x_to_z, nnet_z_to_x,
                        batch_optimizer=None, rng=None,
                        noise_function=None,
                        loss_function=None,
                        loss_function_y=None,
                        loss_function_z=None,
                        nnet_x_to_y=None,
                        X_type=None,
                        walkback=1):
        self.nnet_x_to_z = nnet_x_to_z
        self.nnet_z_to_x = nnet_z_to_x
        self.nnet_x_to_y = nnet_x_to_y

        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        self.batch_optimizer.model = self
        if rng is None:
            rng = RandomStreams(seed=10001)
        self.rng = rng

        self.encode_function = None  # only available after fit
        self.decode_function = None  # only available after fit
        self.predict_function = None # only available after fit
        self.iter_update_batch = None
        self.iter_update = None

        self.get_loss = None

        if loss_function is None:
            loss_function = lambda x, x_hat : objectives.squared_error(x, x_hat).sum(axis=1)
        self.loss_function = loss_function
        self.loss_function_y = loss_function_y
        self.loss_function_z = loss_function_z
        self.noise_function = noise_function
        self.walkback = walkback

        if X_type is None:
            X_type = T.matrix
        self.X_type = X_type

    def fit(self, X, y=None):
        assert ( (y is None and self.nnet_x_to_y is None) or
                 (y is not None and self.nnet_x_to_y is not None))

        nb_batches = easy.get_nb_batches(len(X), self.batch_optimizer.batch_size)

        if self.iter_update_batch is not None:
            if y is not None:
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_optimizer.batch_size,
                            (batch_index+1) * self.batch_optimizer.batch_size)
                    return self.iter_update(X[sl], y[sl])
            else:
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_optimizer.batch_size,
                            (batch_index+1) * self.batch_optimizer.batch_size)
                    return self.iter_update(X[sl])

            self.batch_optimizer.optimize(nb_batches, iter_update_batch)
            return

        X_batch = self.X_type('X_batch')
        if y is not None:
            y_batch = T.ivector('y_batch')
            self.y_batch = y_batch
        self.X_batch = X_batch

        Z_batch = T.matrix('Z_batch')
        self.Z_batch = Z_batch
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           self.batch_optimizer.batch_size)

        if self.noise_function is not None:
            X_batch_noisified = self.noise_function(X_batch)
            X_batch_for_encoder = X_batch_noisified
        else:
            X_batch_for_encoder = X_batch

        z, = self.nnet_x_to_z.get_output(X_batch_for_encoder)
        z_without_noise, = self.nnet_x_to_z.get_output(X_batch)

        if y is not None:
            y_hat, = self.nnet_x_to_y.get_output(X_batch_for_encoder)
            y_hat_without_noise, = self.nnet_x_to_y.get_output(X_batch)

            y_hat_without_noise_pred = T.argmax(y_hat_without_noise, axis=1)
            self.predict_function = theano.function([X_batch], y_hat_without_noise_pred)

        x_hat, = self.nnet_z_to_x.get_output(z)
        x_hat_without_noise, = self.nnet_z_to_x.get_output(z_without_noise)

        self.encode_function = theano.function([X_batch], [z_without_noise])
        self.recover_function = theano.function([X_batch], x_hat_without_noise)

        x_hat_from_z, = self.nnet_z_to_x.get_output(Z_batch)
        self.decode_function = theano.function([Z_batch], [x_hat_from_z])

        all_params = (self.nnet_x_to_z.get_all_params() +
                      self.nnet_z_to_x.get_all_params())
        if y is not None:
            all_params.extend(self.nnet_x_to_y.get_all_params())
        all_params = list(set(all_params))

        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure

        if self.walkback == 1:
            loss_reconstruction = self.loss_function(X_batch, x_hat).mean()

            if y is not None:
                loss_accuracy = self.loss_function_y(y_batch, y_hat).mean()
            else:
                loss_accuracy = 0.

        elif self.walkback > 1:
            # TODO
            pass

        if self.loss_function_z is not None:
            loss_representation = self.loss_function_z(self)
        else:
            loss_representation = 0.

        loss = loss_reconstruction +  loss_accuracy + loss_representation

        if y is None:
            self.get_loss = theano.function([X_batch], loss)
        else:
            self.get_loss = theano.function([X_batch , y_batch], loss)
            self.get_supervised_error = theano.function([X_batch, y_batch], loss_accuracy)

        self.get_reconstruction_error = theano.function([X_batch], loss_reconstruction)

        updates = opti_function(loss, all_params, **opti_kwargs)

        if self.batch_optimizer.whole_dataset_in_device == True:
            X = theano.shared(X, borrow=True)
            if y is not None:
                y = theano.shared(y, borrow=True)

                iter_update_batch = theano.function(
                    [batch_index], loss,
                    updates=updates,
                    givens={
                        X_batch: X[batch_slice],
                        y_batch: y[batch_slice],
                    },
                )

            else:
                iter_update_batch = theano.function(
                    [batch_index], loss,
                    updates=updates,
                    givens={
                        X_batch: X[batch_slice],
                    },
                )
        else:
            if y is not None:
                iter_update = theano.function(
                        [X_batch, y_batch],
                        loss,
                        updates=updates
                )
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_optimizer.batch_size,
                              (batch_index+1) * self.batch_optimizer.batch_size)
                    return iter_update(X[sl], y[sl])
            else:
                iter_update = theano.function(
                        [X_batch],
                        loss,
                        updates=updates
                )
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_optimizer.batch_size,
                              (batch_index+1) * self.batch_optimizer.batch_size)
                    return iter_update(X[sl])
        self.iter_update_batch = iter_update_batch
        self.iter_update = iter_update
        self.batch_optimizer.optimize(nb_batches, iter_update_batch)
        return self

    def encode(self, X):
        assert self.encode_function is not None
        return self.encode_function(X)

    def decode(self, Z):
        assert self.decode_function is not None
        return self.decode_function(Z)

    def sample(self, nb=10, nb_iterations=10, sampling_function=lambda X:X):
        X = np.random.uniform(size=(nb, self.nnet_x_to_z.input_layers[0].shape[1])).astype(theano.config.floatX)
        for i in xrange(nb_iterations):
            X = sampling_function(self.recover_function(X))
        return X

    def predict(self, X):
        assert self.predict_function is not None
        return self.predict_function(X)


def greedy_learn(models, X, y=None):
    input = X
    for m in models:
        print(input.shape)
        m.fit(input, y)
        input, = m.encode(input)
    return models

def greedy_encode(models, X):
    x = X
    for m in models:
        x, = m.encode(x)
    return x


def greedy_learn_with_validation(models, splits, X, y=None,
                                 report_rec_error=True,
                                 report_accuracy=False,
                                 report_supervised_error=True,
                                 stat_train=None,
                                 stat_valid=None):
    train, valid = splits
    input_train, input_valid = X[train], X[valid]
    if y is not None:
        y_train, y_valid = y[train], y[valid]
    else:
        y_train, y_valid = None, None
    for m in models:
        if report_rec_error is True:
            m.batch_optimizer.add_stat(
                    {"rec_error_train": lambda:m.get_reconstruction_error(input_train),
                     "rec_error_valid": lambda:m.get_reconstruction_error(input_valid),
                    }
            )
            if stat_train is not None:
                m.batch_optimizer.add_stat(stat_train(m, input_train))
            if stat_valid is not None:
                m.batch_optimizer.add_stat(stat_valid(m, input_valid))
        if y is not None:
            if report_accuracy is True:
                m.batch_optimizer.add_stat(
                        {"accuracy_valid": lambda:np.mean(m.predict(input_valid)==y_valid),
                         "accuracy_train": lambda:np.mean(m.predict(input_train)==y_train),
                         }
                )
            if report_supervised_error is True:
                m.batch_optimizer.add_stat(
                    {
                         "supervised_error_train": lambda:m.get_supervised_error(input_train, y_train),
                         "supervised_error_Valid": lambda:m.get_supervised_error(input_valid, y_valid)
                    }
                )
            m.batch_optimizer.add_stat(
                    {"loss_valid": lambda:m.get_loss(input_valid, y_valid)}
            )
        else:
            m.batch_optimizer.add_stat(
                    {"loss_valid": lambda:m.get_loss(input_valid)}
            )
        m.fit(input_train, y_train)
        input_train, = m.encode(input_train)
        input_valid, = m.encode(input_valid)
    return models
