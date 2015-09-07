import matplotlib.pyplot as plt
import copy
import numpy as np


def build_visualizations_by_averaging_inputs_maximizing_activations(inputs, activations,
                                                                    shape_2d=None, top=1.):
    """
    inputs must be (ninputs, dim inputs)
    activations must be (ninputs, nhidden units)
    averaging the inputs with the top 100%*top activations
    """
    if shape_2d is None:
        shape = int(np.sqrt(inputs.shape[1]))
        shape_2d = (shape, shape)

    visualizations = np.zeros( (activations.shape[1], shape_2d[0], shape_2d[1]  ) )

    for i in xrange(activations.shape[1]):
        acts = activations[:, i]
        inds = range(inputs.shape[0])
        inds = sorted(inds, key=lambda i: acts[i], reverse=True)
        last = max(1, int(top * inputs.shape[0]))
        inds = inds[0:last]
        visualizations[i] = inputs[inds].mean(axis=0).reshape((shape_2d[0], shape_2d[1]))
    return visualizations


def build_visualizations_by_weighted_combinations(layers_weights, top=1., shape_2d=None,
                                                  until_layer=0):

    """top is the proportion of the units of the previous layer to take into account
       when building the visualization of the units of the current layer
       if top = 1., take every unit in the previous layer

       if shape_2d is None then assume it is a square and infer it from the weight matrix of the
        first layer"""
    if shape_2d is None:
        shape = int(np.sqrt(layers_weights[0].shape[1]))
        shape_2d = (shape, shape)
    visualizations_cur_layer = layers_weights[0].reshape((layers_weights[0].shape[0],
                                                          shape_2d[0], shape_2d[1]))
    visualizations = [visualizations_cur_layer]
    for layer in xrange(1, until_layer + 1):
        W = layers_weights[layer]
        visualizations_cur_layer = build_visualizations_combinations_from_previous_layer(visualizations_cur_layer, W, top=top)
        visualizations.append(visualizations_cur_layer)

    return visualizations


def build_visualizations_combinations_from_previous_layer(visualizations_prev_layer, W, top=1.):
    """
    visualizations: axis 0 = units, axis=1 w axis=2 h
     W : axis=0 units , axis=1 prev layer units
    """
    visualizations_cur_layer = np.zeros((W.shape[0], visualizations_prev_layer.shape[1],
                                         visualizations_prev_layer.shape[2]))

    for unit in xrange(W.shape[0]):
        weights = copy.copy(W[unit])
        weights_ind = range(W.shape[1])
        weights_ind = sorted(weights_ind, key=lambda i: weights[i], reverse=True)
        lasts = max(int(top*W.shape[1]), 1)
        bottom_weights_ind = weights_ind[lasts:]
        weights[bottom_weights_ind] = 0.
        weights /= np.sum(weights)
        visualizations_cur_layer[unit] = np.sum(visualizations_prev_layer *
                                                weights[:, np.newaxis, np.newaxis], axis=0)
    return visualizations_cur_layer


def grid_plot(visualizations, nbrows=None, nbcols=None, random=False, imshow_options=None, fig=None):
    """
    if nbrows or nbcols are None, they are inferred from the number of visualizations
    (by taking the sqr root)
    """
    if fig is None:
        fig = plt.gcf()
    if imshow_options is None:
        imshow_options = {}
    if nbrows is None:
        nbrows = int(np.sqrt(visualizations.shape[0])) + 1
    if nbcols is None:
        nbcols = int(np.sqrt(visualizations.shape[0])) + 1

    for row in xrange(nbrows):
        for col in xrange(nbcols):

            k = (col + row * nbcols)
            if random is True:
                ind = np.random.randint(0, visualizations.shape[0] - 1)
            else:
                ind = k
            if ind >= visualizations.shape[0]:
                break
            ax = fig.add_subplot(nbrows, nbcols, k + 1)
            ax.axis('off')
            ax.imshow(visualizations[ind], **imshow_options)

if __name__ == "__main__":

    from pylearn2.utils import serial
    from pylearn2.datasets.fonts import Fonts

    model = serial.load("model2.pkl")
    #data = Fonts(kind='all', accept_only='.*-[a]-.*')

    X = model.get_input_space().make_theano_batch()
    code = model.encode(X)

    """
    activations = [act]
    for autoencoder in model.autoencoders:
        act = autoencoder.encode(act)
        activations.append(act)
    """
