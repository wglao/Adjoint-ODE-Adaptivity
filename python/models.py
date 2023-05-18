from typing import Any, Callable, Iterable, Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from jax.random import PRNGKey, KeyArray

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any

default_kernel_init = initializers.lecun_normal()


def bias_init(key: KeyArray, shape: Sequence[int], dtype: Any = jnp.float_):
  return jnp.sort(default_kernel_init(key, shape, dtype), axis=None)


class SingleNeuronLayers(nn.Module):
  layers: int = 1
  activation: callable = nn.relu
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs) -> Any:
    f = inputs
    out = jnp.empty((self.layers+1,))
    out = out.at[0].set(jnp.squeeze(inputs))
    for l in range(self.layers):
      bias = self.param('bias', self.bias_init, (1, 1), self.param_dtype)
      weight = self.param('weight', self.kernel_init, (1, 1), self.param_dtype)
      f = f + self.activation(weight*f + bias)
      out = out.at[l+1].set(jnp.squeeze(f))
    return out


class ResBlockSimple(nn.Module):
  """Single Layer Residual Block
  U_{n+1} = U_n + W2@ \sigma(W1*(U_{n}-b))
  """
  features: int
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = bias_init
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, u_n, t_n, dt_n):
    if not isinstance(u_n, jnp.ndarray) or len(jnp.shape(u_n)) == 0:
      u_n = jnp.array([u_n])
    bias = self.param('bias', self.bias_init,
                      (self.features, jnp.shape(u_n)[-1]), self.param_dtype)
    weights1 = self.param('weights1', self.kernel_init,
                          (self.features, jnp.shape(u_n)[-1]), self.param_dtype)
    weights2 = self.param('weights2', self.kernel_init,
                          (jnp.shape(u_n)[-1], self.features), self.param_dtype)

    # f = nn.Dense(features=self.features)(f)
    f = u_n - bias
    f = self.activation(weights1*f)
    f = weights2 @ f

    u_n_plus_1 = u_n + f*dt_n
    return u_n_plus_1


class ResBlock(nn.Module):
  """Single Layer Residual Block
  U_{n+1} = U_n + \sigma (W*U_{n} + b)
  """
  feature_size: int
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

  @nn.compact
  def __call__(self, u_n, t_n, dt_n):
    in_sz = u_n.size
    f = jnp.array([u_n])

    f = nn.Dense(features=self.feature_size)(f)
    f = self.activation(f)

    f = nn.Dense(features=in_sz)(f)
    u_n_plus_1 = u_n + jnp.squeeze(f)*dt_n
    return u_n_plus_1


class ResNetBlock(nn.Module):
  """ResNet for update funcitons"""
  size: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, u_n, t_n, dt_n):
    f = jnp.array([u_n])
    in_sz = f.size

    f = nn.Dense(features=self.size)(f)
    f = self.activation(f)

    f = nn.Dense(features=in_sz)(f)
    u_n_plus_1 = u_n + jnp.squeeze(f)*dt_n
    return u_n_plus_1


class ResNetODE(nn.Module):
  """A Sequence of ResNetBlocks Modeled after an Adapted 1D mesh"""
  feature_sizes: Union[Sequence[Sequence[int]], Sequence[int]]
  dt: jnp.ndarray
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, u_0):
    t: jnp.ndarray = jnp.pad(jnp.cumsum(self.dt), (1, 0), constant_values=0)
    u_prev = u_0

    u_ret = jnp.zeros((len(t), u_0.size))
    u_ret = u_ret.at[0].set(u_0)

    for l, dt_l in enumerate(self.dt):
      u_l = ResNetBlock(self.feature_sizes[l])(u_prev, t[l], dt_l)
      u_prev = u_l

      u_ret = u_ret.at[l + 1].set(u_prev)

    return u_ret
