from typing import Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp


class ResNetBlock(nn.Module):
  """ResNet for update funcitons"""
  feature_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, u_n, t_n, dt):
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
    u_n_plus_1 = u_n + f*dt
    return u_n_plus_1