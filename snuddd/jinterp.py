import jax.numpy as jnp
import jax

class interp1d:
    '''
    Interpolate a 1-D function.

    Only for linear interpolation.
    '''
    def __init__(self, xp, yp):
        if not isinstance(yp, list):
            if xp.shape[-1] != yp.shape[-1]:
                raise ValueError("x and y arrays must be equal in length along "
                                 "interpolation axis.")
        self.xp = xp
        self.yp = yp

    # @jit
    # def interpolate(self, x, xp, yp):
        # return jnp.interp(x, xp, yp)

    def __call__(self, x):
        if isinstance(self.yp, list):
            # Use vmap to vectorize the interpolation over the list of arrays
            try:
                # Attempt to stack the list of arrays
                yp_stacked = jnp.stack(self.yp)
            except ValueError as e:
                raise ValueError("All arrays in the input list must have the same shape.") from e

            return jnp.array(jax.vmap(lambda y: jnp.interp(x, self.xp, y))(yp_stacked))
        
        else:
            return jnp.interp(x, self.xp, self.yp)

