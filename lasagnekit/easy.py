from __future__ import print_function
from collections import Iterable, OrderedDict

import cPickle as pickle
import copy
import numpy as np
import lasagne
import theano
import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from lasagne import layers
from lasagne import updates

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


from tabulate import tabulate

def get_batch_slice(t_batch_index, batch_size):
    return slice(t_batch_index * batch_size, (t_batch_index + 1) * batch_size)


def split_data(data, ratios):
    split = []
    first = 0
    for ratio in ratios:
        nb = int(ratio * len(data))
        last = min(first + nb, len(data))
        split.append(data[first:last])
        first = last
    return split


def get_iter_update_supervision(inputs, outputs, t_X_batch, t_y_batch,
                                t_loss, f_update, t_batch_index, t_batch_slice,
                                sample_weight=None,
                                t_sample_weight=None):
    givens = {
        t_X_batch: inputs[t_batch_slice],
        t_y_batch: outputs[t_batch_slice],
    }
    if sample_weight is None:
        givens[t_sample_weight] = T.as_tensor_variable(np.array(1., dtype=theano.config.floatX))
    else:
        givens[t_sample_weight] = sample_weight[t_batch_slice]

    iter_train = theano.function(
        [t_batch_index], t_loss,
        updates=f_update,
        givens=givens
    )
    return iter_train


def get_iter_update_nonsupervision(inputs, t_X_batch,
                                   t_loss, f_update, t_batch_index,
                                   t_batch_slice):
    iter_train = theano.function(
        [t_batch_index], t_loss,
        updates=f_update,
        givens={
            t_X_batch: inputs[t_batch_slice],
        },
    )
    return iter_train


def get_accuracy(t_y_hat, t_y, softmax=True):
    pred = T.argmax(t_y_hat, axis=1)
    if softmax is True:
        accuracy = T.mean(T.eq(t_y, pred))
    else:
        y_ = T.argmax(t_y, axis=1)
        accuracy = T.mean(T.eq(y_, pred))
    return accuracy


def linearize(X):
    assert len(X.shape) > 1
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))


def to_hamming(targets, presence=1, absence=-1,
               class_mapping=None, nb_classes=None):

    t = sorted(set(list(targets)))

    if class_mapping is None:
        indices = dict((a, i) for i, a in enumerate(t))
        class_mapping = lambda x: indices[x]

    if nb_classes is None:
        nb_classes = len(t)

    mapping = {}
    M = [absence] * nb_classes
    for current_class in t:
        k = class_mapping(current_class)
        M[k] = presence
        mapping[current_class] = copy.copy(M)
        M[k] = absence
    return np.array([mapping[t] for t in targets])


def from_hamming(targets, classes):
    return np.array(map(lambda x: classes[x], np.argmax(targets, axis=1)))


def softmax_loss(output, t_y_batch):
    return -T.mean(T.log(output)[T.arange(t_y_batch.shape[0]), t_y_batch])


def patience_quit(stats, label, nb_epochs_to_check, progression=0.0):
    if len(stats) < nb_epochs_to_check:
        return False
    before = stats[-nb_epochs_to_check].get(label)
    now = stats[-1].get(label)
    if (float(before - now) / now) <= progression:
        return True
    else:
        return False


def put_stats_accuracy(stat, labels, data_list, get_accuracy_func):
    for label, data in zip(labels, data_list):
        accuracy = get_accuracy_func(data.X.get_value(), data.y.get_value())
        stat[label + "_accuracy"] = accuracy
    return stat


def shuffle_data(data):
    positions = np.range(0, data.X.shape[0])
    np.random.shuffle(positions)
    data.X = data.X[positions]
    data.y = data.y[positions]
    return data


def get_theano_batch_variables(batch_size, prefix='', y_softmax=False):

    batch_index = T.iscalar('batch_index' + prefix)
    X_batch = T.matrix('x' + prefix)
    if y_softmax is True:
        y_batch = T.ivector('y' + prefix)
    else:
        y_batch = T.matrix('y' + prefix)
    batch_slice = get_batch_slice(batch_index, batch_size)
    return batch_index, X_batch, y_batch, batch_slice


def main_loop(max_nb_epochs, iter_update, quitter, monitor, observer):
    for i in xrange(max_nb_epochs):
        update_status = iter_update(i)
        monitor_output = monitor(update_status)
        observer(monitor_output)
        if quitter(update_status):
            break

def get_1d_to_2d_square_shape(shape, rgb=False):
    if len(shape) == 2:
        if rgb == True:
            s = int(np.sqrt(shape[1] / 3))
            return (shape[0], s, s, 3)
        else:
            s = int(np.sqrt(shape[1]))
            return (shape[0], s, s)
    else:
        return shape


def get_2d_square_image_view(X, rgb=False):
    shape = get_1d_to_2d_square_shape(X.shape, rgb)
    return X.reshape(shape)


nonlinearities_from_str = {
    'relu': nonlinearities.rectify,
    'linear': nonlinearities.linear,
    'softmax': nonlinearities.softmax,
    'tanh': nonlinearities.tanh,
    'sigmoid': nonlinearities.sigmoid,
    'log': lambda x: (x>=1)*T.log(x)
}


class SimpleNeuralNet(object):

    def __init__(self, nb_hidden_list=None, activations=None, batch_size=10, is_classification=True,
                 learning_rate=0.5, momentum=0.8, max_nb_epochs=10, L1_factor=None, L2_factor=None, dropout_probs=None,
                 max_norm=None, optimization_method='adadelta',
                 patience_nb_epochs=-1,
                 patience_threshold_progression_rate=0.0,
                 validation_set_ratio=None,
                 initweights_list=None,
                 initbiases_list=None,
                 output_softener=1.,
                 input_noise_function=None,
                 patience_stat="loss_train",
                 min_nb_epochs=1,
                 verbose=0,
                 report_each=1,
                 ):

        if nb_hidden_list is None:
            nb_hidden_list = [100]

        if dropout_probs is None:
            dropout_probs = [0] * len(nb_hidden_list)
        elif not isinstance(dropout_probs, Iterable):
            dropout_probs = [dropout_probs] * len(nb_hidden_list)

        if max_norm is not None and not isinstance(max_norm, Iterable):
            max_norm = [max_norm] * len(nb_hidden_list)

        if activations is None:
            activations = ['relu'] * len(nb_hidden_list)
        elif isinstance(activations, basestring):
            activations = [activations] * len(nb_hidden_list)

        if L1_factor is not None and not isinstance(L1_factor, Iterable):
            L1_factor = [L1_factor] * len(nb_hidden_list)

        if L2_factor is not None and not isinstance(L2_factor, Iterable):
            L2_factor = [L2_factor] * len(nb_hidden_list)

        self.initweights_list = initweights_list
        self.initbiases_list = initbiases_list
        self.output_softener = output_softener
        self.input_noise_function = input_noise_function

        self.max_norm = max_norm
        self.is_classification = is_classification
        self.dropout_probs = dropout_probs
        self.batch_size = batch_size
        self.nb_hidden_list = nb_hidden_list
        self.activations = activations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_nb_epochs = max_nb_epochs
        self.min_nb_epochs = min_nb_epochs
        self.L1_factor = L1_factor
        self.L2_factor = L2_factor
        self.optimization_method = optimization_method

        self.patience_nb_epochs = patience_nb_epochs
        self.patience_threshold_progression_rate = patience_threshold_progression_rate
        self.patience_stat = patience_stat
        self.validation_set_ratio = validation_set_ratio
        self.verbose = verbose
        self.report_each = report_each

        self._layers = None
        self._iter_update_batch = None
        self._get_loss = None
        self._prediction_layer = None
        self._predict_function = None
        self._l_x_in = None
        self._stats = None
        self._class_label_encoder = None


        self._output_softener_coefs = None

    def __getstate__(self):
        d =  {}
        d.update(self.__dict__)
        del d["_iter_update_batch"]
        return d

    def get_params(self, deep=True):
        return dict(filter(lambda (k, v): not k.startswith("_"), self.__dict__.items()))

    def set_params(self, **params):
        params_ = dict(
            filter(lambda (k, v): not k.startswith("_"), params.items()))
        self.__dict__.update(params_)

    def _build_model(self, y_dim):

        cur_layer = self._l_x_in

        for i, (hidden, activation, dropout_prob) in enumerate(zip(self.nb_hidden_list, self.activations, self.dropout_probs)):
            if self.initbiases_list is not None and self.initbiases_list[i] is not None:
                b = self.initbiases_list[i]
            else:
                b = init.Constant(0.)

            if self.initweights_list is not None and self.initweights_list[i] is not None:
                W = self.initweights_list[i]
            else:
                W = init.GlorotUniform()

            if type(activation) != tuple and activation in nonlinearities_from_str:
                nonlinearity = nonlinearities_from_str[activation]
                cur_layer = layers.DenseLayer(
                    cur_layer, num_units=hidden, nonlinearity=nonlinearity, W=W, b=b)
                if dropout_prob > 0:
                    cur_layer = lasagne.layers.DropoutLayer(
                        cur_layer, dropout_prob)
            elif type(activation) == tuple:
                name, params = activation
                if name == 'maxout':
                    nb_components = params["nb_components"]
                    cur_layer = layers.DenseLayer(
                        cur_layer, num_units=hidden * nb_components, nonlinearity=nonlinearities.linear, W=W, b=b)
                    if dropout_prob > 0:
                        cur_layer = lasagne.layers.DropoutLayer(
                            cur_layer, dropout_prob)
                    cur_layer = maxout(cur_layer, nb_components)
                else:
                    pass
            else:
                cur_layer = lasagne.layers.NonlinearityLayer(cur_layer, activation)

        # output layer

        if (self.initbiases_list is not None and
            len(self.initbiases_list) == len(self.nb_hidden_list) + 1 and
            self.initbiases_list[-1] is not None):
            b = self.initbiases_list[-1]
        else:
            b = init.Constant(0.)

        if (self.initweights_list is not None and
            len(self.initweights_list) == len(self.nb_hidden_list) + 1 and
            self.initweights_list[-1] is not None):
            W = self.initweights_list[-1]
        else:
            W = init.GlorotUniform()
        if self.is_classification == True:
            if self.output_softener != 1.:

                cur_layer = layers.DenseLayer(
                    cur_layer, num_units = y_dim, nonlinearity=nonlinearities.linear, W=W, b=b)

                if self.output_softener == 'learned':
                    self._output_softener_coefs = theano.shared(np.ones((y_dim,)).astype(theano.config.floatX))
                    cur_layer = layers.NonlinearityLayer(cur_layer, lambda x:x*self._output_softener_coefs)
                else:
                    cur_layer = layers.NonlinearityLayer(cur_layer, lambda x:x*self.output_softener)
                cur_layer = layers.NonlinearityLayer(cur_layer, nonlinearities.softmax)

            else:
                cur_layer = layers.DenseLayer(
                    cur_layer, num_units = y_dim, nonlinearity=nonlinearities.softmax, W=W, b=b)
        else:
            cur_layer = layers.DenseLayer(
                cur_layer, num_units = y_dim, nonlinearity=nonlinearities.linear, W=W, b=b)
        return cur_layer

    def _build_prediction_functions(self, X_batch, prediction_layer):

        output_testing_phase = layers.get_output(prediction_layer,
            X_batch, deterministic=True)
        if self.is_classification:
            pred = T.argmax(output_testing_phase, axis=1)
        else:
            pred = output_testing_phase

        self._predict_function = theano.function(
            [X_batch], pred, allow_input_downcast=True)
        self._output_function = theano.function(
            [X_batch], output_testing_phase, allow_input_downcast=True)

    def _prepare(self, X, y, X_valid=None, y_valid=None, sample_weight=None,
                 whole_dataset_in_device=True):

        self._stats = []
        self._class_label_encoder = LabelEncoder()
        if self.is_classification is True:
            self._class_label_encoder.fit(y)
            self.classes_ = self._class_label_encoder.classes_
            y = self._class_label_encoder.transform(y).astype(y.dtype)
            if y_valid is not None:
                y_valid_transformed = self._class_label_encoder.transform(y_valid).astype(y_valid.dtype)

        self._l_x_in = layers.InputLayer(shape=(None, X.shape[1]))
        batch_index, X_batch, y_batch, batch_slice = get_theano_batch_variables(
            self.batch_size, y_softmax=self.is_classification)

        if sample_weight is not None:
            t_sample_weight = T.vector('sample_weight')
            sample_weight = sample_weight.astype(theano.config.floatX)
        else:
            t_sample_weight = T.scalar('sample_weight')

        if self.is_classification is True:
            y_dim = len(set(y.flatten().tolist()))
        else:
            y_dim = y.shape[1]

        self._prediction_layer = self._build_model(y_dim)
        self._layers = layers.get_all_layers(self._prediction_layer)
        self._build_prediction_functions(X_batch, self._prediction_layer)


        if self.input_noise_function is None:
            output = layers.get_output( self._prediction_layer, X_batch )

        else:
            X_batch_noisy = self.input_noise_function(X_batch)
            output = layers.get_output( self._prediction_layer, X_batch_noisy )

        if self.is_classification:
            loss = -T.mean(t_sample_weight * T.log(output)
                           [T.arange(y_batch.shape[0]), y_batch])
        else:
            loss = T.mean(
                t_sample_weight * T.sum((output - y_batch) ** 2, axis=1))

        all_params = layers.get_all_params(self._prediction_layer)
        if self._output_softener_coefs is not None:
            all_params.append(self._output_softener_coefs)

        W_params = layers.get_all_param_values(self._prediction_layer, regularizable=True)

        # regularization
        if self.L1_factor is not None:
            for L1_factor_layer, W in zip(self.L1_factor, W_params):
                loss += L1_factor_layer * T.sum(abs(W))

        if self.L2_factor is not None:
            for L2_factor_layer, W in zip(self.L2_factor, W_params):
                loss += L2_factor_layer * T.sum(W**2)


        if self.optimization_method == 'nesterov_momentum':
            gradient_updates = updates.nesterov_momentum(loss, all_params, learning_rate=self.learning_rate,
                                                         momentum=self.momentum)
        elif self.optimization_method == 'adadelta':
            # don't need momentum there
            gradient_updates = updates.adadelta(
                loss, all_params, learning_rate=self.learning_rate)
        elif self.optimization_method == 'adam':
            gradient_updates = updates.Adam(
                loss, all_params, learning_rate=self.learning_rate)
        elif self.optimization_method == 'momentum':
            gradient_updates = updates.momentum(
                loss, all_params, learning_rate=self.learning_rate,
                momentum=self.momentum
            )
        elif self.optimization_method == 'adagrad':
            gradient_updates = updates.adadelta(
                loss, all_params, learning_rate=self.learning_rate)
        elif self.optimization_method == 'rmsprop':
            gradient_updates = updates.adadelta(
                loss, all_params, learning_rate=self.learning_rate)
        elif self.optimization_method == 'sgd':
            gradient_updates = updates.sgd(
                loss, all_params, learning_rate=self.learning_rate,
            )
        else:
            raise Exception("wrong optimization method")

        nb_batches = X.shape[0] // self.batch_size
        if (X.shape[0] % self.batch_size) != 0:
            nb_batches += 1

        X = X.astype(theano.config.floatX)
        if self.is_classification == True:
            y = y.astype(np.int32)
        else:
            y = y.astype(theano.config.floatX)

        if whole_dataset_in_device == True:
            X_shared = theano.shared(X, borrow=True)
            y_shared = theano.shared(y, borrow=True)


            givens={
                X_batch: X_shared[batch_slice],
                y_batch : y_shared[batch_slice]
            }

            if sample_weight is not None:
                sample_weight_shared = theano.shared(sample_weight, borrow=True)
                givens[t_sample_weight] = sample_weight_shared[batch_slice]
            else:
                givens[t_sample_weight] = T.as_tensor_variable(np.array(1., dtype=theano.config.floatX))

            iter_update_batch = theano.function(
                [batch_index], loss,
                updates=gradient_updates,
                givens=givens,

            )
        else:
            if sample_weight is None:
                iter_update_gradients = theano.function(
                        [X_batch, y_batch],
                        loss,
                        updates=gradient_updates,
                        givens={t_sample_weight: T.as_tensor_variable(np.array(1., dtype=theano.config.floatX))},

                )
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_size,
                               (batch_index+1) * self.batch_size)
                    return iter_update_gradients(X[sl], y[sl])

            else:
                iter_update_gradients = theano.function(
                        [X_batch, y_batch, t_sample_weight],
                        loss,
                        updates=gradient_updates
                )
                def iter_update_batch(batch_index):
                    sl = slice(batch_index * self.batch_size,
                                    (batch_index+1) * self.batch_size)
                    return iter_update_gradients(X[sl], y[sl], sample_weight[sl])
        self._iter_update_batch = iter_update_batch
        self._get_loss = theano.function(
            [X_batch, y_batch, t_sample_weight], loss, allow_input_downcast=True)

        def iter_update(epoch):
            losses = []
            #self.learning_rate.set_value(self.learning_rate.get_value() * np.array(0.99, dtype=theano.config.floatX))
            for i in xrange(nb_batches):
                losses.append(self._iter_update_batch(i))
                # max norm
                if self.max_norm is not None:
                    for max_norm_layer, layer in zip(self.max_norm, self._layers):
                        layer.W = updates.norm_constraint(
                            layer.W, self.max_norm)

            losses = np.array(losses)

            d = OrderedDict()
            d["epoch"] = epoch
            d["loss_train_std"] = losses.std()
            d["loss_train"] = losses.mean()

            d["accuracy_train"] = (self.predict(self.X_train) == self.y_train).mean()

            if X_valid is not None and y_valid is not None:
                d["loss_valid"] = self._get_loss(X_valid, y_valid_transformed, 1.)

                if self.is_classification == True:
                    d["accuracy_valid"] = (self.predict(X_valid) == y_valid).mean()

            if self.verbose > 0:
                if (epoch % self.report_each) == 0:
                    print(tabulate([d], headers="keys"))
            self._stats.append(d)
            return d

        def quitter(update_status):
            if update_status["epoch"] < self.min_nb_epochs:
                return False
            if self.patience_nb_epochs > 0:
                return patience_quit(self._stats, self.patience_stat,
                                     self.patience_nb_epochs,
                                     self.patience_threshold_progression_rate)
            else:
                return False

        def monitor(update_status):
            pass

        def observer(monitor_output):
            pass

        return (iter_update, quitter, monitor, observer)

    def fit(self, X, y, X_valid=None, y_valid=None, sample_weight=None,
                        whole_dataset_in_device=True):

        if self.validation_set_ratio is not None and X_valid is None and y_valid is None:
            sp = train_test_split(X, y, test_size=self.validation_set_ratio)
            X, X_valid, y, y_valid = sp
            self.X_valid, self.y_valid = X_valid, y_valid

        self.X_train, self.y_train = X, y

        main_loop_funcs = self._prepare(X, y, X_valid, y_valid, sample_weight, whole_dataset_in_device)


        main_loop(self.max_nb_epochs, *main_loop_funcs)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        pred = self._predict_function(X)
        if self.is_classification == True:
            return self._class_label_encoder.inverse_transform(pred)
        else:
            if pred.shape[1] == 1:
                return pred[:, 0]
            else:
                return pred

    def predict_proba(self, X):
        return self._output_function(X)


from collections import OrderedDict

class BatchIterator(object):
    def __init__(self):
        pass

    def __call__(self, batch_size, nb_batches, V=None):
        self.V = V
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        return self

    def __iter__(self):
        for i in range(self.nb_batches):
            yield self.transform(i, self.V)

    def transform(self, batch_index, V):
        assert self.batch_size is not None
        assert self.nb_batches is not None

        if isinstance(batch_index, T.TensorVariable):
            batch_slice = get_batch_slice(batch_index,
                                          self.batch_size)
        else:
            batch_slice = slice(batch_index * self.batch_size,
                                (batch_index+1) * self.batch_size)

        d = OrderedDict()
        for name, value in V.items():
            d[name] = value[batch_slice]
        return d

import cPickle as pickle
import sys

class BatchOptimizer(object):

    def __init__(self, max_nb_epochs=10,
                 batch_size=100,
                 min_nb_epochs=1,
                 patience_nb_epochs=-1,
                 patience_progression_rate_threshold=1.,
                 patience_stat="loss_train",
                 patience_check_each=1,
                 optimization_procedure=None,
                 whole_dataset_in_device=False,
                 verbose=0,
                 report_each=1,
                 batch_iterator=None,
                 verbose_stat_show=None,
                 verbose_out=sys.stdout):

        self.max_nb_epochs = max_nb_epochs

        self.patience_nb_epochs = patience_nb_epochs
        self.patience_progression_rate_threshold = patience_progression_rate_threshold
        self.patience_stat = patience_stat
        self.patience_check_each = patience_check_each

        self.cur_best_patience_stat = None
        self.cur_best_model = None
        self.cur_best_epoch = None

        self.batch_size = batch_size
        self.min_nb_epochs = min_nb_epochs
        self.whole_dataset_in_device = whole_dataset_in_device

        if optimization_procedure is None:
            optimization_procedure = (updates.adadelta, {"learning_rate": 1.})
        self.optimization_procedure = optimization_procedure
        self.verbose = verbose
        self.report_each = report_each
        if batch_iterator is None:
            batch_iterator = BatchIterator()
        self.batch_iterator = batch_iterator
        self.verbose_stat_show = verbose_stat_show
        self.verbose_out = verbose_out

        self.model = None
        self.stats = []
        self.stats_batch = []
        self.func_stats = {}

    def add_stat(self, stat):
        self.func_stats.update(stat)

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        losses = []
        for i in range(nb_batches):
            batch_loss = iter_update_batch(i)
            losses.append(batch_loss)
            self.monitor_batch(dict(batch_loss=batch_loss, batch_index=i, total_nb_batches=(i + 1)*(epoch+1)))
        losses = np.array(losses)

        stat = OrderedDict()
        stat["epoch"] = epoch
        stat["loss_std"] = losses.std()
        stat["loss_train"] = losses.mean()

        for name, func in self.func_stats.items():
            stat[name] = func()
        self.stats.append(stat)
        return stat

    def quitter(self, update_status):
        cur_epoch = len(self.stats) - 1
        if cur_epoch < self.min_nb_epochs:
            return False
        if self.patience_nb_epochs > 0 and (cur_epoch % self.patience_check_each)==0:
            # patience heuristic (for early stopping)
            cur_patience_stat = update_status[self.patience_stat]

            if self.cur_best_patience_stat is None:
                self.cur_best_patience_stat = cur_patience_stat
                first_time = True
            else:
                first_time = False

            thresh = self.patience_progression_rate_threshold
            if cur_patience_stat < self.cur_best_patience_stat * thresh or first_time:

                if self.verbose >= 2:
                    fmt = "--Early stopping-- good we have a new best value : {0}={1}, last best : epoch {2}, value={3}"
                    print(fmt.format(self.patience_stat, cur_patience_stat, self.cur_best_epoch, self.cur_best_patience_stat))
                self.cur_best_epoch = cur_epoch
                self.cur_best_patience_stat = cur_patience_stat
                if hasattr(self.model, "set_state") and hasattr(self.model, "get_state"):
                    self.cur_best_model = self.model.get_state()
                else:
                    self.cur_best_model = pickle.dumps(self.model.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
            if (cur_epoch - self.cur_best_epoch) >= self.patience_nb_epochs:
                finish = True
                if hasattr(self.model, "set_state") and hasattr(self.model, "get_state"):
                    self.model.set_state(self.cur_best_model)
                else:
                    self.model.__dict__.update(pickle.loads(self.cur_best_model))

                self.stats = self.stats[0:self.cur_best_epoch + 1]
                if self.verbose >= 2:
                    print("out of patience...take the model at epoch {0} and quit".format(self.cur_best_epoch + 1), file=self.verbose_out)
            else:
                finish = False
            return finish
        else:
            return False

    def monitor(self, update_status):
        if self.verbose > 0:
            stat = self.stats[-1]
            if (stat["epoch"] % self.report_each) == 0:
                from tabulate import tabulate
                stat_ = dict()
                if self.verbose_stat_show is not None:
                    for k in self.verbose_stat_show:
                        stat_[k] = stat[k]
                else:
                    stat_ = stat
                print(tabulate([stat_], headers="keys"), file=self.verbose_out)

    def monitor_batch(self, update_status):
        if self.verbose >= 2:
            fmt = "batch #{0} loss : {1}, #{2} mini-batches processed"
            print(fmt.format(update_status["batch_index"],
                             update_status["batch_loss"],
                             update_status["total_nb_batches"]))

    def observer(self, monitor_output):
        pass

    def optimize(self, nb_batches, iter_update_batch):
        main_loop(self.max_nb_epochs,
                  lambda epoch: self.iter_update(
                      epoch, nb_batches, iter_update_batch),
                  lambda update_status: self.quitter(update_status),
                  lambda update_status: self.monitor(update_status),
                  lambda monitor_output: self.observer(monitor_output))


class LightweightModel(object):

    def __init__(self, input_layers, output_layers):
        self.input_layers = input_layers
        self.output_layers = output_layers

    def get_output(self, *X, **params):
        givens = {}
        for input_layer, x in zip(self.input_layers, X):
            givens[input_layer] = x
        return [layers.get_output(output_layer, givens, **params)
                for output_layer in self.output_layers]

    def get_all_params(self, **kwargs):
        return list(set(param
                        for output_layer in self.output_layers
                        for param in (
                            layers.helper.get_all_params(output_layer, **kwargs))))




# Taken from pylearn2
def log_sum_exp(A=None, axis=None, log_A=None):
    """
    A numerically stable expression for
    `T.log(T.exp(A).sum(axis=axis))`
    Parameters
    ----------
    A : theano.gof.Variable
        A tensor we want to compute the log sum exp of
    axis : int, optional
        Axis along which to sum
    log_A : deprecated
        `A` used to be named `log_A`. We are removing the `log_A`
        interface because there is no need for the input to be
        the output of theano.tensor.log. The only change is the
        renaming, i.e. the value of log_sum_exp(log_A=foo) has
        not changed, and log_sum_exp(A=foo) is equivalent to
        log_sum_exp(log_A=foo).
    Returns
    -------
    log_sum_exp : theano.gof.Variable
        The log sum exp of `A`
    """

    if log_A is not None:
        assert A is None
        warnings.warn("log_A is deprecated, and will be removed on or"
                      "after 2015-08-09. Switch to A")
        A = log_A
    del log_A

    A_max = T.max(A, axis=axis, keepdims=True)
    B = (
        T.log(T.sum(T.exp(A - A_max), axis=axis, keepdims=True)) +
        A_max
    )

    if axis is None:
        return B.dimshuffle(())
    else:
        if type(axis) is int:
            axis = [axis]
        return B.dimshuffle([i for i in range(B.ndim) if
                             i % B.ndim not in axis])

class InitializerFrom(object):

    def __init__(self, W):
        self.W = W

    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        assert shape == self.W.shape
        return self.W

def get_stat(name, stats):
    return [stat[name] for stat in stats]

# Source : https://github.com/Lasagne/Lasagne/issues/193
def maxout(incoming, ds):
    #l1a = layers.DenseLayer(incoming, nonlinearity=None,
    #                        num_units=num_units * ds, **kwargs)
    l1 = layers.FeaturePoolLayer(incoming, ds)
    return l1


def channel_out(block_size):

    def f(X):
        """
        Apply hard local winner-take-all on every rows of a theano matrix.
        Parameters
        ----------
        p: theano matrix
            Matrix on whose rows LWTA will be applied.
        block_size: int
            Number of units in each block.
        """
        p = X
        batch_size = p.shape[0]
        num_filters = p.shape[1]
        num_blocks = num_filters // block_size
        w = p.reshape((batch_size, num_blocks, block_size))
        block_max = w.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
        max_mask = T.cast(w >= block_max, 'float32')
        indices = np.array(range(1, block_size+1))
        max_mask2 = max_mask * indices
        block_max2 = max_mask2.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
        max_mask3 = T.cast(max_mask2 >= block_max2, 'float32')
        w2 = w * max_mask3
        w3 = w2.reshape((p.shape[0], p.shape[1]))
        return w3
    return f


def get_nb_batches(nb_examples, batch_size):
    nb_batches = nb_examples // batch_size
    if (nb_examples  % batch_size) > 0:
        nb_batches += 1
    return nb_batches


def iterate_minibatches(nb_inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(nb_inputs)
        np.random.shuffle(indices)
    for start_idx in range(0, nb_inputs- batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt

if __name__ == "__main__":

    import cPickle as pickle

    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    params = {"n_samples": 1000,
              "n_classes": 5,
              "n_informative": 5}

    X, y = make_classification(**params)

    X = X.astype(theano.config.floatX)
    y = y.astype('int32')
    X, y = shuffle(X, y)

    nnet = SimpleNeuralNet(nb_hidden_list=[100],
                           activations="relu",
                           batch_size=10,
                           optimization_method='adadelta',
                           verbose=1,
                           max_nb_epochs=100)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nnet', nnet)
    ])

    scores = cross_val_score(pipeline, X, y,
                             cv=5, scoring='accuracy', n_jobs=1,
                             verbose=1)
    print(scores.mean(), scores.std())
