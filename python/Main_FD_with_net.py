"""
Adaptively solve ODE with finite difference in time,
using Adjoint-Weighted Residual for a posterior error
estimates to guide time step refinement
"""

import os
import shutil
from functools import partial

import cv2
import flax
import flax.linen as nn
from jax import grad, jit, vmap
from jax.lax import dynamic_slice_in_dim as dyslice
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import models
import numpy as np
import optax
import scipy.integrate as integrate
from clu import metrics
from factory import *
from flax import struct
from flax.training import train_state

case = "ResNet_test"
is_net = True
linear_ode = False
linear_out_functional = True
ode = 'ResNet'
out_functional = 'J=u_N'  # J = { int(u), int(u^2), u_N, ... }

# time discretization
n_steps = 3
t_span = np.array([0, 1])
n_nodes = n_steps + 1
times = np.linspace(t_span[0], t_span[1], n_nodes)
dt_n = np.diff(times)

ref_factor = 4  # <-- must be > 2

problem = Problem(case, is_net, linear_ode, linear_out_functional, ode,
                  out_functional, ref_factor, t_span)
a_state = AdaptState(problem, times)
ff = FunFactory(problem)
funs = ff.getFunctions()
afuns = ff.getAdaptFunctions()

szs = [100, 500]
net = models.ResNetBlock(szs)
rng = jrand.PRNGKey(0)
params = net.init(rng, np.ones(1), np.ones(1), np.ones(1))['params']
del rng

n_epochs = 10000
n_refine = 100
batch_size = 10
n_ivp = 1000
n_test = n_ivp // 100
n_batches = int(np.ceil((n_ivp-n_test) / batch_size))
learning_rate = 1e-4

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


@struct.dataclass
class Metrics(metrics.Collection):
  err: metrics.Average.from_output('err')
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics


t_state = TrainState.create(
    apply_fn=vmap(afuns.forwardSolve, in_axes=(None, None, 0, None, None)),
    params=params,
    tx=optimizer,
    metrics=Metrics.empty())


def trueODE(t, y):
  # return jnp.cos(10*t)*jnp.sin(y)**2 / jnp.max(jnp.array([y, 1e-8]))
  # return jnp.sin(y)**2 / jnp.where(jnp.abs(y) > 1e-8, y, 1e-8)
  return t*jnp.sin(y)


@jit
def trainStep(state: TrainState, dt_n: jnp.ndarray, u0_batch: jnp.ndarray,
              true_batch: jnp.ndarray) -> TrainState:

  def lossFunction(params, u0_batch, true_batch):
    u = state.apply_fn(funs, dt_n, u0_batch, net, params)
    loss = jnp.mean(jnp.square(u - true_batch))
    return loss

  grads = grad(lossFunction)(state.params, u0_batch, true_batch)
  state = state.apply_gradients(grads=grads)
  return state


@jit
def computeMetrics(state: TrainState, dt_n: jnp.ndarray, u0_train: jnp.ndarray,
                   u0_test: jnp.ndarray, true_train: jnp.ndarray,
                   true_test: jnp.ndarray) -> TrainState:
  u = state.apply_fn(funs, dt_n, u0_train, net, state.params)
  loss = jnp.square(u - true_train)
  u = state.apply_fn(funs, dt_n, u0_test, net, state.params)
  err = jnp.square(u - true_test)
  metric_updates = state.metrics.single_from_model_output(err=err, loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state, jnp.mean(loss), jnp.mean(err)


@jit
def getTrainBatches(seed, u0_train, true_train):
  rng = jrand.PRNGKey(seed)
  u0_shuffle = jrand.permutation(rng, u0_train, independent=False)
  true_shuffle = jrand.permutation(rng, true_train, independent=False)
  batch = lambda u, b: dyslice(u, b*batch_size, batch_size)
  u0_batches = vmap(
      batch, in_axes=(None, 0))(u0_shuffle,
                                jnp.arange(n_batches, dtype=jnp.int32))
  true_batches = vmap(
      batch, in_axes=(None, 0))(true_shuffle,
                                jnp.arange(n_batches, dtype=jnp.int32))
  return u0_batches, true_batches


if __name__ == "__main__":
  # folder for saving plots as images
  if os.path.isdir(case):
    shutil.rmtree(case)
  os.mkdir(case)

  wandb_upload = False
  if wandb_upload:
    import wandb
    wandb.init(project="Adjoint Adaptivity", entity="wglao", name=case)
    wandb.config.problem = 'ResNet'
    wandb.config.method = 'Recurrent'
    wandb.config.batchsize = str(batch_size)

  # randomly sample initial conditions
  u0_vec = jrand.normal(jrand.PRNGKey(10), (100,))
  true_arr = jnp.array([integrate.odeint(trueODE, u0, times) for u0 in u0_vec])
  # true_arr = jnp.expand_dims(
  #     jnp.array([jnp.exp(times)*u0 for u0 in u0_vec]), -1)

  u0_test, u0_train = jnp.split(u0_vec, [10])
  true_test, true_train = jnp.split(true_arr, [10])

  for epoch in range(n_epochs):
    # get batches
    u0_batches, true_batches = getTrainBatches(epoch, u0_train, true_train)
    for b in range(n_batches):
      t_state = trainStep(t_state, dt_n, u0_batches[b], true_batches[b])

    t_state, loss, err = computeMetrics(t_state, dt_n, u0_batches[0, ...],
                                        u0_test, true_batches[0,
                                                              ...], true_test)
    # err, loss = t_state.metrics.compute().values()

    if epoch % 100 == 0:
      fig = plt.figure()
      for i in range(n_test):
        u_plot = afuns.forwardSolve(funs, dt_n, u0_test[i], net, t_state.params)
        true_plot = true_test[i]
        plt.plot(times, u_plot, '-', color='tab:blue')
        plt.plot(times, true_plot, '-', color='tab:orange')
      plt.savefig(a_state.problem.case + '/' + 'ep{:d}.png'.format(epoch))
      plt.close(fig)

      if wandb_upload:
        wandb.log({
            'Epoch': epoch,
            'Error': err,
            'Loss': loss,
        })
      else:
        print('Epoch: {:d}'.format(epoch))
        print('Error: {:.2e}'.format(err))
        print('Loss: {:.2e}'.format(loss))

    if epoch % (n_epochs//n_refine) == 0:
      rng = jrand.PRNGKey(epoch)
      u0_adapt = jrand.permutation(rng, u0_test)[0]
      a_state = afuns.adapt(a_state, u0_adapt, net, params)
