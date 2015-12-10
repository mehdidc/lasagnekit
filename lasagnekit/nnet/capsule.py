import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano

from ..easy import BatchOptimizer, BatchIterator, get_nb_batches
from collections import OrderedDict

class Capsule(object):

    def __init__(self,
                 input_variables,
                 model,
                 loss_function,
                 batch_iterator=None,
                 batch_optimizer=None,
                 functions=None,
                 parse_grads=None,
                 rng=None):

        self.input_variables = input_variables
        self.model = model
        self.loss_function = loss_function
        if batch_optimizer is None:
            batch_optimizer = BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        if batch_iterator is None:
            batch_iterator = BatchIterator()
        batch_iterator.model = self
        self.batch_iterator = batch_iterator
        if rng is None:
            rng = RandomStreams(1234)
        self.rng = rng
        if functions is None:
            functions = dict()
        self.functions = functions

        self.nb_batches = None
        self.batch_optimizer.model = self
        self.model.capsule = self

        v_tensors = OrderedDict()
        for name, var in self.input_variables.items():
            v_tensors[name] = var.get("tensor_type", T.matrix)()
        self.v_tensors = v_tensors
        self.shared_vars = []
        self.built = False


    def get_state(self):
        return [param.get_value() for param in self.all_params]

    def set_state(self, state):
        for cur_param, state_param in zip(self.all_params, state):
            cur_param.set_value(state_param, borrow=True)

    def fit(self, **V):

        V_ = OrderedDict()
        for name in self.input_variables.keys():
            V_[name] = V[name]
        V = V_
        if "X" in V:
            X = V["X"]
        else:
            X = V[V.keys()[0]]

        self.nb_batches = get_nb_batches(len(X),
                                         self.batch_optimizer.batch_size)
        if self.built is False:
            self._build(V)
        self.V = V
        self.batch_optimizer.optimize(self.nb_batches, self.iter_update_batch)
        return self

    def _build_functions(self):
        for name, attrs in self.functions.items():

            params = attrs.get("params")
            params_tensors = [self.v_tensors[p] for p in params]
            func = theano.function(
                    params_tensors,
                    attrs.get("get_output")(self.model, *params_tensors)
            )
            setattr(self, name, func)


    def _build(self, V):
        self._build_functions()
        v_tensors = self.v_tensors
        all_params = self.model.get_all_params()
        all_params_regularizable = self.model.get_all_params(regularizable=True)
        self.all_params = all_params
        self.all_params_regularizable = all_params_regularizable

        r = self.loss_function(self.model, v_tensors)
        if type(r) == tuple:
            loss, updates = r
        else:
            loss = r
            updates = OrderedDict()
        self.loss = loss

        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure

        grads = T.grad(loss, all_params)
        updates.update(opti_function(grads, all_params, **opti_kwargs))
        self.updates = updates

        batch_index = T.iscalar('batch_index')

        if self.batch_optimizer.whole_dataset_in_device is True:
            V_device = {name: theano.shared(value, borrow=True) for name, value in V.items()}
            self.shared_vars.append(V_device)
            bi = self.batch_iterator(self.batch_optimizer.batch_size,
                                     self.nb_batches)
            V_device_transformed = bi.transform(batch_index, V_device)
            givens = {v_tensors[name]: value
                     for name, value in V_device_transformed.items()}
            iter_update_batch = theano.function(
                [batch_index], loss,
                updates=updates,
                givens=givens,
                on_unused_input='warn'
            )
            self.bi = bi
            self.iter_update_batch = iter_update_batch
            self.built = True
        else:
            L = v_tensors.values()
            iter_update = theano.function(
                L,
                loss,
                updates=updates,
                on_unused_input='warn'
            )
            bi = self.batch_iterator(self.batch_optimizer.batch_size,
                                     self.nb_batches)
            self.bi = bi
            self.V = V
            self.iter_update = iter_update
            self.built = True

    def iter_update_batch(self, batch_index):
        V_transformed = self.bi.transform(batch_index, self.V)
        params = V_transformed.values()
        return self.iter_update(*params)


    def __del__(self):
        #https://github.com/Lasagne/Lasagne/issues/311
        with self.all_params:
            self.params.popitem()

if __name__ == "__main__":
    from lasagnekit.easy import LightweightModel
    from lasagne import layers, nonlinearities
    import numpy as np
    from collections import OrderedDict
    x_dim = 10
    y_dim = 3
    x_in = layers.InputLayer((None, x_dim))
    l_hidden = layers.DenseLayer(x_in, num_units=100)
    l_out = layers.DenseLayer(l_hidden, num_units=y_dim, nonlinearity=nonlinearities.softmax)

    model = LightweightModel([x_in], [l_out])


    def loss_function(model, tensors):
        y_pred, = model.get_output(tensors["X"])
        return T.nnet.categorical_crossentropy(y_pred, tensors["y"]).mean()

    input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
            y=dict(tensor_type=T.ivector)
    )

    functions = dict(
           predict=dict(get_output=lambda model, X: model.get_output(X)[0].argmax(axis=1),
                         params=["X"]),
           predict_proba=dict(get_output=lambda model, X: model.get_output(X)[0],
                              params=["X"]),
    )

    batch_optimizer = BatchOptimizer(verbose=2)
    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)

    from sklearn.datasets import make_classification
    X, y = make_classification(n_classes=y_dim, n_features=x_dim, n_informative=3)
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    capsule.fit(X=X, y=y)

    print((capsule.predict(X)==y).mean())
