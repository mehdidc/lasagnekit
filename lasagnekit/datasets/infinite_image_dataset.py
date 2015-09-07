import numpy as np

from lasagnekit.easy import get_1d_to_2d_square_shape, linearize
from skimage import transform as tf

def trans_scale_rotate_f(x, **params):
    imshape = params.get("imshape", get_1d_to_2d_square_shape(x.shape, rgb=params.get("rgb", False)))

    x_t_max = params.get("x_translate_max", imshape[2] - 1)
    y_t_max = params.get("y_translate_max", imshape[1] - 1)
    x_t_max = min(x_t_max, imshape[2] - 1)
    y_t_max = min(y_t_max, imshape[2] - 1)

    x_t_min = params.get("x_translate_min", -x_t_max)
    y_t_min = params.get("y_translate_min", -y_t_max)

    theta_range = params.get("theta_range", (0., 0.))

    x_scale_range = params.get("x_scale_range", (1., 1.))
    y_scale_range = params.get("y_scale_range", (1., 1.))
    
    x_im = x.reshape(imshape)
    rnd_stream = params.get("rnd_stream", np.random)


    x_t = params.get("x_translate", rnd_stream.randint(x_t_min, x_t_max, size=(x.shape[0],))  )
    y_t = params.get("y_translate", rnd_stream.randint(y_t_min, y_t_max, size=(x.shape[0],))  )
    theta_t = params.get("theta", rnd_stream.uniform(theta_range[0], theta_range[1], size=(x.shape[0],))  )
    x_scale_t = params.get("x_scale", rnd_stream.uniform(x_scale_range[0], x_scale_range[1], size=(x.shape[0],))  )
    y_scale_t = params.get("y_scale", rnd_stream.uniform(y_scale_range[0], y_scale_range[1], size=(x.shape[0],))  )

    output = params.get("copy_to", x)
    for i in xrange(x.shape[0]):
        t = tf.AffineTransform(translation=(x_t[i], y_t[i]), rotation=theta_t[i], scale=(x_scale_t[i], y_scale_t[i]))
        o = (tf.warp(x_im[i], t))
        o = o.reshape( np.prod(o.shape) )
        output[i] = o.astype(output.dtype)
    params = np.concatenate((x_t[:, np.newaxis], y_t[:, np.newaxis], theta_t[:, np.newaxis], x_scale_t[:, np.newaxis], y_scale_t[:, np.newaxis]), axis=1)
    return output, params

def rotate_f(x, **params):
    return x


class InfiniteImageDataset(object):

    def __init__(self, initial_X, initial_y, batch_size, params):
        self.initial_X = initial_X
        self.initial_y = initial_y
        self.params = params
        self.batch_size = batch_size

    def load(self):
        s = np.random.randint(0, self.initial_X.shape[0], size=self.batch_size)
        samples =  self.initial_X[s]
        self.X, self.X_params = trans_scale_rotate_f(samples, **self.params)

        if self.initial_y is not None:
            self.y = self.initial_y[s]

if __name__ == "__main__":
    import numpy as np
    from skimage.data import lena
    X = lena()[np.newaxis, :]
    translate_f(X, rgb=True)
