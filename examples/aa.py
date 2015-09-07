import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from lasagnekit.generative import autoencoder
from lasagnekit.easy import (BatchOptimizer, LightweightModel,
                          get_2d_square_image_view)
from lasagnekit.misc.plot_weights import grid_plot
from lasagne import layers, nonlinearities, updates, init
from sklearn.datasets import load_digits
from lasagnekit.datasets.mnist import MNIST
import theano
from sklearn.utils import shuffle

import theano

import matplotlib.pyplot as plt

from theano.tensor.shared_randomstreams import RandomStreams


if __name__ == "__main__":
    np.random.seed(0)

    data = MNIST()
    data.load()
    X = data.X


    #data = load_digits()
    #X = data['data']
    #y = data['target']

    #X = X.astype(theano.config.floatX)
    #X /= X.max()
    n = 60000
    n = X.shape[0]
    X = X[0:n]

    #X, y = shuffle(X, y)
    X = shuffle(X, random_state=0)
    z_dim = 256

    # X to Z (decoder)
    sz = int(np.sqrt(X.shape[1]))
    x_in = layers.InputLayer(shape=(None, X.shape[1]))

    """
    # ConvnetModel
    x_in_2d = layers.ReshapeLayer(x_in, ([0], 1, sz, sz))
    h = layers.Conv2DLayer(x_in_2d, num_filters=20, filter_size=(3, 3))
    """

    h = layers.DenseLayer(x_in, num_units=z_dim,
                          W=init.GlorotUniform(),
                          nonlinearity=nonlinearities.rectify)
    z_out = layers.DenseLayer(h, num_units=z_dim,
                              W=init.GlorotUniform(),
                              nonlinearity=nonlinearities.rectify)
    nnet_x_to_z = LightweightModel([x_in],
                                   [z_out])
    # Z to X (encoder)

    z_in = layers.InputLayer(shape=(None, z_dim))
    x_out = layers.DenseLayer(z_in, num_units=X.shape[1],
                              W=init.GlorotUniform(),
                              nonlinearity=nonlinearities.sigmoid)
    nnet_z_to_x = LightweightModel([z_in],
                                   [x_out])
    # instantiate the model
    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)

    batch_optimizer = MyBatchOptimizer(max_nb_epochs=30,
                                       optimization_procedure=(updates.adadelta, {"learning_rate" : 1.}),
                                       batch_size=256,
                                       whole_dataset_in_device=True,
                                       verbose=1)
    rng = RandomStreams(seed=1000)
    #noise_function = lambda X_batch: X_batch * rng.binomial(size=X_batch.shape, p=0.7)
    noise_function = None
    model = autoencoder.Autoencoder(nnet_x_to_z, nnet_z_to_x, batch_optimizer, noise_function=noise_function, walkback=1)
    model.fit(X)

    """
    conv_filters = h.W.get_value()
    conv_filters = conv_filters.reshape( (conv_filters.shape[0], conv_filters.shape[2], conv_filters.shape[3]))
    grid_plot(conv_filters, imshow_options={"cmap": "gray"})
    plt.savefig('out-filters-conv.png')
    plt.show()
    """

    filters = h.W.get_value().T
    filters = get_2d_square_image_view(filters)
    grid_plot(filters, imshow_options={"cmap":"gray"})
    plt.savefig("out-filters.png")
    plt.show()

    plt.clf()
    samples = model.sample(nb=100, nb_iterations=10000)
    samples = get_2d_square_image_view(samples)
    grid_plot(samples, imshow_options={"cmap": "gray"})
    plt.savefig('out-samples.png')
    plt.show()
