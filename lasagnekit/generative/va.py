import numpy as np

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano


from lasagne import nonlinearities

from .. import easy

def kl_gaussian_term(mu, log_sigma, prior_mu, prior_log_sigma):
    return (
            -0.5 + prior_log_sigma - log_sigma +
            0.5 * (T.exp(2 * log_sigma) +
                  (mu - prior_mu) ** 2) / T.exp(2 * prior_log_sigma)
    ).sum(axis=range(1, mu.ndim))


def decoder_term_gaussian(X, x_mean_hat, x_log_sigma_hat):
    # x_mean_hat and x_sigma_hat computed by a neural network
    x_hat = X - x_mean_hat
    log_x_given_z = -0.5 * ((x_hat ** 2) / T.exp(2. * x_log_sigma_hat) + 2 * x_log_sigma_hat) - 0.5*np.log(2*np.pi)
    return (log_x_given_z.sum(axis=range(2, log_x_given_z.ndim)))


def decoder_term_bernoulli(X, x_mean_hat):
    #x_hat = -T.nnet.binary_crossentropy(x_mean_hat, X)
    x_hat = -(X * T.nnet.softplus(-x_mean_hat) + (1 - X) * T.nnet.softplus(x_mean_hat))
    return x_hat.sum(axis=range(2, x_hat.ndim))

def decoder_term_categorical(X, x_mean_hat):
    log_x_given_z = X *  T.log(x_mean_hat)
    log_x_given_z = log_x_given_z.sum(axis=range(2, log_x_given_z.ndim))
    return log_x_given_z

def loss_general(X, X_params_from_z, decoder_loss_function, z_mean_from_x, z_log_sigma_from_x, prior_mu, prior_log_sigma):
    return -(- kl_gaussian_term(z_mean_from_x, z_log_sigma_from_x, prior_mu, prior_log_sigma) +
               decoder_loss_function(X, X_params_from_z).mean(axis=1))


def z_from_epsilon(epsilon, z_mean, z_log_sigma):

    shape = [z_mean.shape[0], 1] + [z_mean.shape[i] for i in range(1, z_mean.ndim)]
    z_mean_ = z_mean.reshape(shape)

    shape = [z_log_sigma.shape[0], 1] + [z_log_sigma.shape[i] for i in range(1, z_log_sigma.ndim)]
    z_log_sigma_ = z_log_sigma.reshape(shape)

    z = z_mean_ + T.exp(z_log_sigma_) * epsilon
    return z

Binary = dict(
    mean_param_index=0,
    nonlinearity=nonlinearities.sigmoid,
    decoder_loss_function=lambda X, X_params: decoder_term_bernoulli(X, X_params[0]),
    sampler=lambda rng, params : 1*(rng.uniform(size=params[0].shape)) < params[0]
)

Real = dict(
    mean_param_index=0,
    nonlinearity=nonlinearities.linear,
    decoder_loss_function=lambda X, X_params: decoder_term_gaussian(X, X_params[0], X_params[1]),
    sampler = lambda rng, params: params[0]
)

Categorical = dict(
        mean_param_index=0,
        nonlinearity=nonlinearities.linear,
        decoder_loss_function=lambda X, X_params: decoder_term_categorical(X, X_params[0]),
        sampler=lambda rng, params: params[0].argmax(axis=2)
)


class VariationalAutoencoder(object):

    def __init__(self, nnet_x_to_z, nnet_z_to_x,
                 batch_optimizer=None, nb_z_samples=None,
                 input_type=Binary,
                 X_type=T.matrix,
                 Z_type=T.matrix,
                 rng=None):
        self.nnet_x_to_z = nnet_x_to_z
        self.nnet_z_to_x = nnet_z_to_x
        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        assert self.batch_optimizer.model is None
        self.batch_optimizer.model = self

        self.input_type = input_type
        self.nb_z_samples = nb_z_samples

        if rng is None:
            rng = RandomStreams(seed=10001)
        self.rng = rng

        self.X_type = X_type
        self.Z_type = Z_type

        self.encode_function = None  # only available after fit
        self.decode_function = None  # only available after fit
        self._prior_z_mean = None # only available after fit
        self._prior_z_log_sigma = None # only available after fit
        self.all_params = None # only available after fit

    def get_state(self):
        return [param.get_value() for param in self.all_params]
    
    def set_state(self, state):
        for cur_param, state_param in zip(self.all_params, state):
            cur_param.set_value(state_param, borrow=True)

    def fit(self, X):


        z_dim = self.nnet_x_to_z.output_layers[0].num_units
        z_shape = self.nnet_z_to_x.input_layers[0].shape[1:]
        self._prior_z_mean = theano.shared(np.zeros(z_shape, dtype=theano.config.floatX))
        self._prior_z_log_sigma = theano.shared(np.zeros(z_shape, dtype=theano.config.floatX))

        nb_batches = easy.get_nb_batches(X.shape[0], self.batch_optimizer.batch_size)

        #perm = self.rng.permutation(n=X.shape[0])
        #X = theano.shared(X, borrow=True)
        #X = X[perm]

        X_batch = self.X_type('X_batch')
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           self.batch_optimizer.batch_size)

        z_mean, z_log_sigma = self.nnet_x_to_z.get_output(X_batch)

        self.encode_function = theano.function([X_batch], self.nnet_x_to_z.get_output(X_batch, determnistic=True))

        nb_z_samples = (self.nb_z_samples
                        if self.nb_z_samples is not None
                        else 1)
        #z_dim = z_mean.shape[-1]

        epsilon_shape = [z_mean.shape[0], nb_z_samples] + [z_mean.shape[i] for i in range(1, z_mean.ndim)]
        epsilon = self.rng.normal(epsilon_shape)

        z = z_from_epsilon(epsilon, z_mean, z_log_sigma)

        z_for_nnet_shape = [z.shape[0]*z.shape[1]] + [z.shape[i] for i in range(2, z.ndim)]
        z_for_nnet = z.reshape(z_for_nnet_shape)

        X_params = self.nnet_z_to_x.get_output(z_for_nnet) # x_mean, x_log_sigma for example
        for i in range(len(X_params)):
            p = X_params[i]
            shape = [X_batch.shape[0], z.shape[1]] + [p.shape[j] for j in range(1, p.ndim)]
            X_params[i] = p.reshape(shape)
        # loss
        x_mean = X_params[self.input_type.get("mean_param_index")]
        self.reconstruction_error_function = theano.function([X_batch],
                                                             ((X_batch - self.input_type.get("nonlinearity")(x_mean.mean(axis=1)))**2).sum(axis=1).mean())

        shape = [X_batch.shape[0], 1] + [X_batch.shape[i] for i in range(1, X_batch.ndim)]
        X_batch_for_loss = X_batch.reshape(shape)
        loss = loss_general(X_batch_for_loss, X_params,
                            self.input_type.get("decoder_loss_function"),
                            z_mean, z_log_sigma,
                            self._prior_z_mean, self._prior_z_log_sigma).mean(axis=0)
        self.get_likelihood_lower_bound = theano.function([X_batch], loss)
        # the decode function
        Z_batch = self.Z_type('Z_batch')
        X_params_from_z = self.nnet_z_to_x.get_output(Z_batch, deterministic=True)

        ind_mean = self.input_type.get("mean_param_index")
        X_params_from_z[ind_mean] = self.input_type.get("nonlinearity")(X_params_from_z[ind_mean])
        self.decode_function = theano.function([Z_batch], X_params_from_z)
        self.sample_x_given_z = theano.function(
                [Z_batch],
                self.input_type.get("sampler")(self.rng, X_params_from_z)
        )
        # marginal log-likelihood approximation (using importance sampling)
        log_p_x_given_z = self.input_type.get("decoder_loss_function")(X_batch_for_loss, X_params)

        z_mean_shape = [z_mean.shape[0], 1] + [z_mean.shape[i] for i in range(1, z_mean.ndim)]
        z_log_sigma_shape = [z_log_sigma.shape[0], 1] + [z_log_sigma.shape[i] for i in range(1, z_log_sigma.ndim)]
        log_q_z_given_x = decoder_term_gaussian(z, z_mean.reshape(z_mean_shape), z_log_sigma.reshape(z_log_sigma_shape))
        log_p_z = decoder_term_gaussian(z, self._prior_z_mean, self._prior_z_log_sigma)
        log_likelihood_approximation =  -(easy.log_sum_exp(log_p_z + log_p_x_given_z - log_q_z_given_x, axis=1) - T.log(nb_z_samples)).mean()
        self.log_likelihood_approximation_function = theano.function([X_batch], log_likelihood_approximation)

        # Parameters and optimization
        all_params = (self.nnet_x_to_z.get_all_params() +
                      self.nnet_z_to_x.get_all_params() 
                     + [self._prior_z_mean, self._prior_z_log_sigma])
        self.all_params = all_params
        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure
        updates = opti_function(loss, all_params, **opti_kwargs)

        # mini-batch iteration function and optimization

        if self.batch_optimizer.whole_dataset_in_device == True:
            X = theano.shared(X, borrow=True)
            iter_update_batch = theano.function(
                [batch_index], loss,
                updates=updates,
                givens={
                    X_batch: X[batch_slice],
                },
            )
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

        self.batch_optimizer.optimize(nb_batches, iter_update_batch)
        return self

    def encode(self, X):
        assert self.encode_function is not None
        return self.encode_function(X)

    def transform(self, X):
        params = self.encode(X)
        return params[self.input_type.get("mean_param_index")]

    def decode(self, Z):
        assert self.decode_function is not None
        return self.decode_function(Z)

    def sample(self, nb, only_means=False):

        shape = self._prior_z_mean.get_value().shape
        prior_z_mean = self._prior_z_mean.get_value().flatten()
        prior_z_log_sigma = self._prior_z_log_sigma.get_value().flatten()
        Z = np.random.multivariate_normal(prior_z_mean,
                                          np.diag(np.exp(prior_z_log_sigma)),
                                          size=nb)
        Z = Z.astype(theano.config.floatX)
        Z = Z.reshape([nb] + list(shape))
        if only_means is True:
            X_params = self.decode(Z)
            return X_params[self.input_type.get("mean_param_index")]
        else:
            X = self.sample_x_given_z(Z)
            return X
