import functools

import objax
from objax.typing import JaxArray

from jax.nn.initializers import glorot_uniform

from typing import Tuple

def glrt_uniform(shape: Tuple[int, ...], generator = objax.random.DEFAULT_GENERATOR):
    """Returns a ``JaxArray`` of shape ``shape`` with random numbers from a Glorot uniform distribution [0, 1].

    NOTE: if random numbers are generated inside a jitted, parallelized or vectorized function
    then generator variables (including DEFAULT_GENERATOR) have to be added to the
    variable collection."""
    initializer = glorot_uniform()
    return initializer(key=generator(), shape=shape)

class MLPerceptron(objax.nn.Sequential):
    """MLP implementation."""

    def __init__(self, nin, nclass, layer_units=None, **kwargs):
        """Creates ConvNet instance.

        Args:
            nin: number of channels in the input image.
            nclass: number of output classes.
            layer_units: list of layer units for hidden layers (e.g. [512, 256, 128, 64])
        """
        
        if layer_units is None or len(layer_units)==0 :
            ops = [objax.nn.Linear(nin, nclass, w_init=glrt_uniform)]
            super().__init__(ops)
        else:
            ops = [objax.nn.Linear(nin, layer_units[0], w_init=glrt_uniform), 
                   objax.functional.relu
                   ]

            if layer_units is None or len(layer_units)>1:
                for k in range(len(layer_units)-1):
                    ops += [objax.nn.Linear(layer_units[k], layer_units[k+1], w_init=glrt_uniform), 
                            objax.functional.relu
                            ]

            ops +=  [objax.nn.Linear(layer_units[-1], nclass, w_init=glrt_uniform)] # no relu because we manually compute softmax
            super().__init__(ops)
            

