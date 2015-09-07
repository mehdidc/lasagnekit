
from lasagne.generative.capsule import Capsule
from lasagne.updates import norm_constraint
from collections import OrderedDict
import theano.tensor as T
from lasagne.easy import BatchOptimizer
import theano

class DummyModel(object):

    def __init__(self, all_params):
        self.all_params = all_params

    def get_all_params(self, **tags):
        if tags.get("regualarizable", False) is True:
            return []
        return self.all_params


def build_dreamer(model,
                  X,
                  batch_optimizer=None,
                  loss_function=None,
                  rng=None,
                  input_type=T.matrix):
    if batch_optimizer is None:
        batch_optimizer = BatchOptimizer()
    input_variables = OrderedDict()
    input_variables["X"] = dict(input_type=input_type)

    functions = dict()

    inputs = theano.shared(X, borrow=True)

    if loss_function is None:
        # by default maximize the L2 norm
        def loss_function(x):
            return -(x**2).sum()
    
    def parse_grads(grads):
        return [grad  for grad in grads]

    def loss(m, tensors):
        #inputs_ = T.exp(inputs)
        #inputs__ = inputs_ / inputs_.sum(axis=1).dimshuffle(0, 'x')
        return loss_function(model.get_output((inputs))[0])

    dreamer = Capsule(
        input_variables,
        DummyModel([inputs]),
        loss,
        functions=functions,
        batch_optimizer=batch_optimizer,
        parse_grads=parse_grads
    )
    dreamer.inputs = inputs
    return dreamer
