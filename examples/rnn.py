import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from lasagnekit.generative import va, rnn
from lasagnekit.easy import (BatchOptimizer, LightweightModel,
                             get_2d_square_image_view)
from lasagnekit.misc.plot_weights import grid_plot
from lasagne import layers, nonlinearities, updates, init
from sklearn.datasets import load_digits
from lasagnekit.datasets.mnist import MNIST
import theano
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from theano.sandbox import rng_mrg

if __name__ == "__main__":
    from lightexperiments.light import Light
    light = Light()
    light.launch() # init the DB
    light.initials() # save the date and init the timer

    light.file_snapshot() # save the content of the python file running
    state = 1515
    np.random.seed(state)
    light.set_seed(state) # save the content of the seed

    light.tag("variational_recurrent_autoencoder_example")
    #light.tag("automatic-run")

    #z_dim = np.random.choice((2, 5, 10, 20, 30, 80, 100, 120, 160, 180))
    #hidden = np.random.randint(10, 1200)
    #max_epochs = np.random.randint(50, 500)
    #learning_rate = np.random.uniform(0.0001, 0.01)
    #batch_size = np.random.choice((10, 50, 100, 128, 164, 200, 256, 300, 500, 512))

    z_dim = 500
    hidden = 800
    max_epochs = 300
    learning_rate = 1.
    batch_size = 10

    light.set("z_dim", z_dim)
    light.set("hidden", hidden)
    light.set("max_epochs", max_epochs)
    light.set("learning_rate", learning_rate)
    light.set("batch_size", batch_size)


    data = np.load("/home/gridcl/mehdicherti/work/data/paul_graham/paul_graham.npz")
    X = data['X']
    vocab = data['inverse_mapping'].tolist()
    """   
    data = np.load("/home/gridcl/mehdicherti/work/data/abc/sounds.npy.npz")
    X = data['X']
    v = data['vocab_mapping'].tolist()
    new_data = np.zeros((X.shape[0], X.shape[1], len(v)))
    repr_symbol = []
    for i in range(len(v)):
        L = [0] * len(v)
        L[i] = 1
        repr_symbol.append(L)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            symbol = X[i, j]
            new_data[i, j, :] = repr_symbol[symbol]
    X = new_data 
    """

    #X = X[:, 0:10, :]

    time_length = X.shape[1]

    #data = MNIST(which='train')
    #data.load()
    #X = data.X
    #y = data.y
    #X = 1.*(np.random.uniform(size=X.shape) <= X)

    #data = load_digits()
    #print(data.keys())
    #X = data['data']
    #y = data['target']

    X = X.astype(theano.config.floatX)
    print(X.shape)
    X = X[0:10]

    #X /= X.max()
    #n = 60000
    #X = X[0:n]
    #y = y[0:n]

    X = shuffle(X, random_state=state)
    #X, y = shuffle(X, y, random_state=state)
    # X to Z (decoder)
    x_in = layers.InputLayer(shape=(None, time_length, len(vocab)))
    #h = layers.DropoutLayer(x_in, p=0.3)
    h = rnn.SimpleRNN(x_in, num_units=hidden,
                      nonlinearity=nonlinearities.tanh)
    h_encoder = h
    #h = layers.DropoutLayer(h, p=0.5)
    z_mean_out = rnn.SimpleRNN(h, num_units=z_dim,
                               nonlinearity=nonlinearities.linear)
    z_sigma_out = rnn.SimpleRNN(h, num_units=z_dim,
                                nonlinearity=nonlinearities.linear)

    nnet_x_to_z = LightweightModel([x_in],
                                   [z_mean_out, z_sigma_out])
    # Z to X (encoder)
    z_in = layers.InputLayer(shape=(None, time_length, z_dim))
    h = rnn.SimpleRNN(z_in, num_units=hidden,
                      nonlinearity=nonlinearities.tanh)
    h_decoder = h
    x_out = rnn.SimpleRNN(h, num_units=len(vocab),
                          nonlinearity=nonlinearities.softmax)
    nnet_z_to_x = LightweightModel([z_in], [x_out])

    # instantiate the model
    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):

            super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            for k, v in self.stats[-1].items():
                light.append(k, float(v))
            if (epoch % 10) == 0:
                s = self.model.sample(10)
                print(s.shape)
                print(s[0])
                mapping = data['inverse_mapping'].tolist()
                for a in s:
                    print("".join(map(lambda k:mapping[k], a)))


    batch_optimizer = MyBatchOptimizer(max_nb_epochs=max_epochs,
                                       optimization_procedure=(updates.adadelta, {"learning_rate": learning_rate}),
                                       verbose=1,
                                       whole_dataset_in_device=True,
                                       batch_size=batch_size)
    model = va.VariationalAutoencoder(nnet_x_to_z, nnet_z_to_x, 
                                      batch_optimizer, 
                                      rng=rng_mrg.MRG_RandomStreams(),
                                      X_type=T.tensor3,
                                      Z_type=T.tensor3,
                                      input_type=va.Categorical,
                                      nb_z_samples=1)

    model.fit(X)
    
    light.endings() # save the duration
    light.store_experiment() # update the DB
    light.close() # close the DB
