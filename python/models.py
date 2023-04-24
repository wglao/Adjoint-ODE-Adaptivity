from typing import Sequence, Callable, Union

import flax.linen as nn
import jax.numpy as jnp


class ResNetBlock(nn.Module):
  """ResNet for update funcitons"""
  feature_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, u_n, t_n, dt_n):
    in_sz = u_n.size
    f = jnp.concatenate((jnp.array([u_n]), jnp.array([t_n])), None)

    for size in self.feature_sizes:
      # Residual Block
      f = nn.Dense(features=size)(f)
      f = self.activation(f)
      f = nn.Dense(features=size)(f)

      # output
      # f = self.activation(f)

    # return output to size of input
    f = nn.Dense(features=in_sz)(f)
    u_n_plus_1 = u_n + f*dt_n
    return u_n_plus_1
  
class ResNetODE(nn.Module):
  """A Sequence of ResNetBlocks Modeled after an Adapted 1D mesh"""
  feature_sizes: Union[Sequence[Sequence[int]], Sequence[int]]    
  dt: jnp.ndarray
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, u_0, return_all: bool = False):
    t: jnp.ndarray = jnp.pad(jnp.cumsum(self.dt),(1,0),constant_values=0)
    u_prev = u_0
    
    if return_all:
      u_ret = jnp.zeros((len(t),u_0.size))
      u_ret = u_ret.at[0].set(u_0)

    for l, dt_l in enumerate(self.dt):
      u_l = ResNetBlock(self.feature_sizes)(u_prev,t[l],dt_l)
      u_prev = u_l
    
      if return_all:
        u_ret = u_ret.at[l+1].set(u_prev)
    
    if return_all:
      return u_ret
    return u_prev
