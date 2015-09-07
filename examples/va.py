
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from lasagnekit.generative import va
from lasagnekit.easy import (BatchOptimizer, LightweightModel,
                            get_2d_square_image_view)
from lasagnekit.misc.plot_weights import grid_plot
from lasagne import layers, nonlinearities, updates, init
from sklearn.datasets import load_digits
from lasagnekit.datasets.mnist import MNIST
from lasagnekit.datasets.faces import Faces
import theano
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.sandbox import rng_mrg
import time


def fig_to_list(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.tolist()

if __name__ == "__main__":
    from lightexperiments.light import Light
    light = Light()
    light.launch() # init the DB
    light.initials() # save the date and init the timer

    light.file_snapshot() # save the content of the python file running
    #state = 1515
    state = int(time.time())
    np.random.seed(state)
    light.set_seed(state) # save the content of the seed

    light.tag("variational_autoencoder_example")
    learning_trajectory = False
    save_samples = True
    random_search = True
    save_codes = True
    save_hidden_activations = True
    save_weights = True

    if learning_trajectory is True:
        light.tag("learning_trajectory")
    if save_samples is True:
        light.tag("save_samples")
    if random_search is True:
        light.tag("random_search")

    nb_exp = 5 if learning_trajectory is True else 1

    data = MNIST(which='train')
    #data = Faces(dataset='faces94')
    data.load()
    X = data.X

    X = X.astype(theano.config.floatX)
    y = data.y if hasattr(data, "y") else None

    if y is not None:
        X, y = shuffle(X, y)
    else:
        X = shuffle(X)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    
    for EXP in range(nb_exp):
        state = int(time.time())
        np.random.seed(state)
        
        if random_search is True:
            z_dim = np.random.choice((2, 5, 10, 20, 30, 80, 100, 120, 160, 180))
            hidden = np.random.randint(100, 1000)
            max_epochs = 300
            learning_rate = 10**np.random.uniform(-5, -3)
            batch_size = 256
            nb_layers = np.random.randint(1, 4)
            batch_normalization = False
        else:    
            z_dim = 10
            hidden = 1500
            learning_rate = 0.000112
            max_epochs = 450
            batch_size = 512
            batch_normalization = False 
            nb_layers = 1


        light.set("z_dim", z_dim)
        light.set("hidden", hidden)
        light.set("max_epochs", max_epochs)
        light.set("learning_rate", learning_rate)
        light.set("batch_size", batch_size)
        light.set("batch_normalization", batch_normalization)
        light.set("nb_layers", nb_layers)

        # X to Z (decoder)
        x_in = layers.InputLayer(shape=(None, X.shape[1]))
        h = x_in
        encoder_layers = []
        X_tensor = T.matrix()
        for i in range(nb_layers):

            h = layers.DenseLayer(h, num_units=hidden / (2**i),
                                  nonlinearity=nonlinearities.rectify)
            if batch_normalization is True:
                pass
            encoder_layers.append( theano.function( [X_tensor], layers.get_output(h, X_tensor) ) )

        z_mean_out = layers.DenseLayer(h, num_units=z_dim,
                                       nonlinearity=nonlinearities.linear)
        z_sigma_out = layers.DenseLayer(h, num_units=z_dim,
                                        nonlinearity=nonlinearities.linear)

        nnet_x_to_z = LightweightModel([x_in],
                                       [z_mean_out, z_sigma_out])
        # Z to X (encoder)
        z_in = layers.InputLayer(shape=(None, z_dim))
        Z_tensor = T.matrix()
        h = z_in
        decoder_layers = []
        for i in range(nb_layers):
            h = layers.DenseLayer(h, num_units=hidden/(2**(nb_layers-i-1)),
                                  nonlinearity=nonlinearities.rectify)
            if batch_normalization is True:
                pass
            #decoder_layers.append( theano.function( [X_tensor], layers.get_output(h, X_tensor) ) )

        x_out = layers.DenseLayer(h, num_units=X.shape[1],
                                  nonlinearity=nonlinearities.linear)
        nnet_z_to_x = LightweightModel([z_in], [x_out])

        # instantiate the model
        class MyBatchOptimizer(BatchOptimizer):

            def iter_update(self, epoch, nb_batches, iter_update_batch):
                status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
                #status["reconstruction_error"] = self.model.reconstruction_error_function(X)
                #status["lower_bound_train"] = self.model.get_likelihood_lower_bound(X_train)
                status["lower_bound_validation"] = self.model.get_likelihood_lower_bound(X_valid)
                status["log_likelihood_validation"] = self.model.log_likelihood_approximation_function(X_valid)
                for k, v in status.items():
                    light.append(k, float(v))
                
                if epoch % 10 == 0:

                    if save_codes is True:
                        plt.clf()

                        fig = plt.gcf()
                        code, code_sigma = self.model.encode(X_valid)
                        pca = PCA(n_components=2)
                        code_2d = pca.fit_transform(code)
                        if y is None:
                            plt.scatter(code_2d[:, 0], code_2d[:, 1])
                        else:
                            plt.scatter(code_2d[:, 0], code_2d[:, 1], c=y_valid)
                        fig.canvas.draw()
                        data = fig_to_list(fig)
                        light.append("codes", light.insert_blob(data))

                    if save_samples is True:
                        fig = plt.gcf()
                        X_ = self.model.sample(100, only_means=True)
                        X_ = get_2d_square_image_view(X_)
                        light.append("samples", light.insert_blob(X_.tolist()))

                    if learning_trajectory is True:
                        points, _ = self.model.encode(X[0:200])
                        points = points.ravel().tolist()
                        light.append("trajectories", {"points": points, "epoch": epoch, "seed": state})
                    
                    if save_hidden_activations is True:
                        samples = range(0, 200)
                        hidden = []
                        for layer in self.model.encoder_layers:
                            h = layer(X[samples])
                            h = h.tolist()
                            hidden.append(h)
                        light.append("hidden_activations_encoder", light.insert_blob(hidden))

                return status

        batch_optimizer = MyBatchOptimizer(max_nb_epochs=max_epochs,
                                           optimization_procedure=(updates.rmsprop, {"learning_rate": learning_rate}),
                                           verbose=2,
                                           whole_dataset_in_device=True,
                                           patience_nb_epochs=25,
                                           patience_stat="lower_bound_validation",
                                           patience_progression_rate_threshold=0.99,
                                           patience_check_each=10,
                                           batch_size=batch_size)

        model = va.VariationalAutoencoder(nnet_x_to_z, nnet_z_to_x, 
                                          batch_optimizer, 
                                          rng=rng_mrg.MRG_RandomStreams(seed=state),
                                          nb_z_samples=2)
        model.encoder_layers = encoder_layers
        model.decoder_layers = decoder_layers
        model.fit(X_train)

        """
        X_ = model.sample(200, only_means=True)
        if hasattr(data, "img_dim"):
            X_  = X_.reshape(  [X_.shape[0]]  + list(data.img_dim) )
        else:
            X_ = get_2d_square_image_view(X_)
        grid_plot(X_, imshow_options={"cmap": "gray", "interpolation": None})
        plt.savefig('out-samples.png')
        #light.set("out-samples.png", open("out-samples.png").read())
        plt.clf()
        features = h_encoder.W.get_value().T
        if hasattr(data, "img_dim"):
            features = features.reshape(   [features.shape[0]] + list(data.img_dim) )
        else:
            features = get_2d_square_image_view(features)
        grid_plot(features, imshow_options={"cmap": "gray"})
        plt.savefig('out-encoder-features.png')
        #light.set("out-encoder-features.png", open("out-encoder-features.png").read())

        plt.clf()

        features = x_out.W.get_value()
        if hasattr(data, "img_dim"):
            features = features.reshape(   [features.shape[0]] + list(data.img_dim) )
        else:
            features = get_2d_square_image_view(features)
        grid_plot(features, imshow_options={"cmap": "gray"})
        plt.show()
        plt.savefig('out-decoder-features.png')
        #light.set("out-decoder-features.png", open("out-decoder-features.png").read())
        plt.clf()


        iterations = (range(len(batch_optimizer.stats)))
        values = np.array([stats["loss_train"] for stats in batch_optimizer.stats])
        errors = np.array([stats["loss_std"] for stats in batch_optimizer.stats])
        #log_likelihood = np.array([stats["log_likelihood"] for stats in batch_optimizer.stats])

        plt.errorbar(iterations, values, yerr=errors)
        #plt.plot(iterations, log_likelihood)
        plt.show()
        plt.savefig("out-loss-per-epoch.png")

        plt.clf()

        from sklearn.decomposition import PCA
        code, code_sigma = model.encode(X)
        pca = PCA(n_components=2)
        code_2d = pca.fit_transform(code)
        if y is None:
            plt.scatter(code_2d[:, 0], code_2d[:, 1])
        else:
            plt.scatter(code_2d[:, 0], code_2d[:, 1], c=y)
        plt.show()
        plt.savefig("out-embedding.png")

        """
    light.endings() # save the duration
    light.store_experiment() # update the DB
    light.close() # close the DB
