import jax.numpy as jnp

class interp1d:
    '''
    Interpolate a 1-D function.

    only for linear interpolation.
    
    '''
    def __init__(self, xp, yp):
        if not isinstance(yp, list):
            if xp.shape[-1] != yp.shape[-1]:
                raise ValueError("x and y arrays must be equal in length along "
                                "interpolation axis.")
        self.xp = xp
        self.yp = yp

    def __call__(self, x):
        if isinstance(self.yp, list):
            result = []
            for y in self.yp:
                result.append(jnp.interp(x, self.xp, y))
            return jnp.array(result)
        else:
            return jnp.interp(x, self.xp, self.yp)
    

