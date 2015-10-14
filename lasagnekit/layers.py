from lasagne import layers
from lasagne import nonlinearities
from lasagne import init

class Conv2DDenseLayer(layers.Conv2DLayer):
    
  def __init__(self, incoming, num_units,
               W=init.GlorotUniform(), 
               b=init.Constant(0.),
               nonlinearity=nonlinearities.rectify,
               **kwargs):
    num_filters = num_units
    filter_size = kwargs.get("filter_size", incoming.output_shape[2:])
    if "filter_size" in kwargs:
        del kwargs["filter_size"]
    super(Conv2DDenseLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)
