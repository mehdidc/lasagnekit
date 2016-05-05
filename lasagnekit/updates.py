import theano.tensor as T
from collections import OrderedDict
from lasagne.updates import get_or_compute_grads, sgd
import theano
from lasagne import utils
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse


def santa_euler(loss_or_grads, params,
                learning_rate=1,
                lambda_=1e-5,
                sigma=0.99,
                A=1,
                burnin=0,
                rng=RandomStreams()):
    n = learning_rate
    tprev = theano.shared(utils.floatX(0.))
    t = tprev + 1
    all_grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, g_t in zip(params, all_grads):
        s = rng.normal(size=param.shape)
        f = g_t
        b = A * t**lambda_

        value = param.get_value(borrow=True)
        vprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)
        aprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)

        gprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)

        if hasattr(n, "get_value"):
            n_ = n.get_value(borrow=True)
        else:
            n_ = n

        ufirst = (np.random.normal(size=value.shape) * np.sqrt(n_))
        ufirst = ufirst.astype(value.dtype)
        uprev = theano.shared(ufirst,
                              broadcastable=param.broadcastable)

        v = sigma * vprev + (1 - sigma) * f * f

        g = 1 / T.sqrt(lambda_ + T.sqrt(v))

        a = ifelse(t < burnin, aprev + (uprev * uprev - n / b), aprev)
        u = ifelse(t < burnin, (n / b) * (1 - g / gprev) / uprev + T.sqrt(2 * n / b * gprev) * s,
                   theano.shared(np.zeros(value.shape, dtype=value.dtype)))
        u = u + (1 - a) * uprev - n * g * f
        updates[param] = param + g * u
        updates[uprev] = u
        updates[aprev] = a
        updates[vprev] = v
        updates[gprev] = g
    updates[tprev] = t
    return updates


def santa_sss(loss_or_grads, params,
              learning_rate=1,
              lambda_=1e-5,
              sigma=0.99,
              A=1,
              burnin=0,
              rng=RandomStreams()):
    n = learning_rate
    tprev = theano.shared(utils.floatX(0.))
    t = tprev + 1
    all_grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    for param, g_t in zip(params, all_grads):
        s = rng.normal(size=param.shape)
        f = g_t
        b = A * t**lambda_

        value = param.get_value(borrow=True)
        vprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)
        aprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)

        gprev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)
        if hasattr(n, "get_value"):
            n_ = n.get_value(borrow=True)
        else:
            n_ = n
        ufirst = (np.random.normal(size=value.shape) * np.sqrt(n_))
        ufirst = ufirst.astype(theano.config.floatX)
        uprev = theano.shared(ufirst,
                              broadcastable=param.broadcastable)

        v = sigma * vprev + (1 - sigma) * f * f

        g = 1 / T.sqrt(lambda_ + T.sqrt(v))

        a = aprev + (uprev * uprev - n/b) / 2.
        u = T.exp(-a/2) * uprev
        u = u - g * f * n + T.sqrt(2 * gprev * n/b) * s + n/b*(1 - g / gprev) / uprev
        u = T.exp(-a/2) * u
        a = a + (u * u - n/b) / 2.

        a_explr = a
        u_explr = u

        a = aprev
        u = T.exp(-a/2.) * uprev
        u = u - g * f * n
        u = T.exp(-a/2.) * u
        u_refine = u
        a_refine = a

        a = ifelse(t < burnin, a_explr, a_refine)
        u = ifelse(t < burnin, u_explr, u_refine)
        updates[param] = param + g * uprev / 2. + g * u / 2.
        updates[uprev] = u
        updates[aprev] = a
        updates[vprev] = v
        updates[gprev] = g
    updates[tprev] = t
    return updates


if __name__ == "__main__":
    X = T.matrix()
    Y = T.vector()
    p = 1
    theta = theano.shared(np.random.normal(size=(p,)).astype(np.float32))
    L = ((T.dot(X, theta).flatten() - Y) ** 2).mean()
    updates = santa_sss(L, [theta], learning_rate=0.01)
    # updates = sgd(L, [theta], learning_rate=0.01)

    f = theano.function([X, Y], L, updates=updates)
    x = np.random.uniform(size=(99, p)).astype(np.float32)
    y = x[:, 0] * 2
    L_min = np.inf
    for i in range(500):
        k = np.random.randint(0, len(x), size=32)
        f(x[k], y[k])

        L = f(x, y)
        print(L)
        if L < L_min:
            L_min = L
            theta_min = theta.get_value()

    print(theta_min)
