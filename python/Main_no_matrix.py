import os
import shutil
from functools import partial

import cv2
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtr
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.integrate as integrate
from jax import grad, jit, vmap
from jax.lax import dynamic_slice_in_dim as dySlice
from matplotlib.patches import Rectangle

from models import ResNetBlock, ResNetODE


def forwardFn(u, t, dt, params: dict, net: nn.Module):
  return net.apply({'params': params}, u[-1], t[-1], dt)


# @jit
def forwardSolve(u_0, dt, params: dict, net: nn.Module):
  # return net.apply({'params': params}, u0, return_all=True)
  u = u_0*jnp.ones((len(dt) + 1, 1))
  t = jnp.pad(jnp.cumsum(dt), (1, 0), constant_values=0)
  u_prev = u
  for l, dt_l in enumerate(dt):
    u_next = forwardFn(u[:l + 1], t[:l + 1], dt_l, params, net)
    u = u.at[l + 1].set(u_next)
  return u


def outFnl(u, t=None):
  return u[-1]


# @partial(jit, static_argnums=2)
def adjointSolve(u, dt, ref_factor):
  # refine u
  _, dt_fine, t_fine, u_fine = refineSolution(u, dt, ref_factor)

  dJdU = grad(outFnl)(u_fine, t_fine)
  v0 = dJdU[-1]
  v = v0*jnp.ones_like(u_fine)

  # t = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dt)), None)

  def sumTerm(v_j, u, t, dt, i, j):
    u_prev = u[:j]
    t_prev = t[:j]
    dt_j = dt[j - 1]
    return v_j*(grad(forwardFn)(u_prev, t_prev, dt_j))[i]

  for i in jnp.arange(len(v) - 2, -1, -1):
    js = jnp.arange(i + 1, len(v), dtype=jnp.int32)
    v_next = dJdU[i]
    for j in js:
      v_next = v_next + sumTerm(v[j], u_fine, t_fine, dt_fine, i, j)
    v = v.at[i].set(v_next)
  return v


def refineTime(dt, ref_factor):
  dt_fine = jnp.zeros((len(dt)*ref_factor))
  for i, dt_i in enumerate(dt):
    dt_fine = dt_fine.at[i*ref_factor:(i+1)*ref_factor].set(dt_i / ref_factor)
  t_fine = jnp.cumsum(dt_fine)
  t_fine = jnp.pad(t_fine, (1, 0), constant_values=0)
  return dt_fine, t_fine


def refineSolution(u, dt, ref_factor):
  t_coarse = jnp.cumsum(dt)
  t_coarse = jnp.pad(t_coarse, (1, 0), constant_values=0)
  dt_fine, t_fine = refineTime(dt, ref_factor)
  u_fine = jnp.interp(t_fine, jnp.squeeze(t_coarse), jnp.squeeze(u))
  return t_coarse, dt_fine, t_fine, u_fine


# @partial(jit, static_argnums=2)
def errorIndicator(u, v, dt, ref_factor):
  err = jnp.zeros(len(u) - 1)
  t_coarse, dt_fine, t_fine, u_fine = refineSolution(u, dt, ref_factor)
  res_u = jnp.zeros_like(u_fine)
  for n in jnp.arange(1, len(res_u)):
    residual = u_fine[n] - forwardFn(u_fine[:n], t_fine[:n], dt_fine[n - 1])
    res_u = res_u.at[n].set(residual)
  err_fine = res_u*v
  for i in jnp.arange(len(err)):
    err = err.at[i].set(
        jnp.sum(err_fine[i*ref_factor + 1:(i+1)*ref_factor + 1]))
  return jnp.abs(err)


def lossFn(u_0, t, dt, params, net):
  u = forwardSolve(u_0, dt, params, net)
  true = jnp.sin(t) + u[0]
  loss = jnp.mean(jnp.square(u - true))
  return loss


def trainStep(u_0, t, dt, params, net, opt_state,
              tx: optax.GradientTransformation):
  grads = jtr.tree_map(
      lambda m: jnp.mean(m, axis=0),
      vmap(grad(lossFn, argnums=(3,)),
           in_axes=(0, None, None, None, None))(u_0, t, dt, params, net))[0]
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state


def metricCalc(u_0, t, dt, params, net):
  loss = lossFn(u_0[0], t, dt, params, net)
  err = lossFn(u_0[1], t, dt, params, net)
  return loss, err


if __name__ == "__main__":
  case = "ResNet_no_matrix"
  wandb_upload = True
  if wandb_upload:
    import wandb
    wandb.init(project="Adjoint Adaptivity", entity="wglao", name=case)
    wandb.config.problem = 'ResNet'
    wandb.config.method = 'Recurrent'

  t_span = jnp.array([0, 1])
  n_steps = 2
  t = jnp.linspace(t_span[0], t_span[1], n_steps + 1)
  dt = jnp.diff(t)
  ref_factor = 4
  dt_fine, t_fine = refineTime(dt, ref_factor)

  it = 0
  maxit = 100
  err_total = 1
  tol = 1e-8

  # folder for saving plots as images
  if os.path.isdir(case):
    shutil.rmtree(case)
  os.mkdir(case)

  # net and training
  rng = jrand.PRNGKey(0)
  net = ResNetBlock((10,))
  params = net.init(rng, jnp.ones(1), jnp.ones(1), jnp.ones(1))['params']

  n_epochs = 100
  learning_rate = 1e-3
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)

  u_0_train = jrand.normal(rng, (100,))
  u_0_test = jnp.concatenate((jnp.array([u_0_train[0]]), jnp.ones((1, 1))),
                             None)

  while err_total > tol and it <= maxit:
    # train
    for ep in range(n_epochs):
      params, opt_state = trainStep(u_0_train, t, dt, params, net, opt_state,
                                    optimizer)
      loss, err = metricCalc(u_0_test, t, dt, params, net)

      if wandb_upload:
        wandb.log({
            'Epoch': ep + it*n_epochs,
            'Loss': loss,
            'Error': err,
            'Refinements': it,
        })

    # solve
    u = forwardSolve(u_0_test[1], dt, params, net)
    v = adjointSolve(u, dt, ref_factor)
    err = errorIndicator(u, v, dt, ref_factor)

    # plot
    fig, ax1 = plt.subplots()
    ax1.bar(
        t[:-1] + dt/2, err, dt, color='darkseagreen', label='Error Indicator')
    ax1.set_ylabel('Error Contribution')
    if it == 0:
      bar_ylim = ax1.get_ylim()
    else:
      ax1.set_ylim(*bar_ylim)

    ax2 = ax1.twinx()
    ax2.plot(t, u, '-', marker='.', color='tab:blue', label='Forward')
    ax2.plot(t_fine, v, '-', marker='.', color='tab:orange', label='Ajoint')
    ax2.set_ylabel('Solution')
    ax2.set_xlabel('Time')

    fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

    f_name = case + '_{:d}'.format(it)
    fig.savefig(case + '/' + f_name + '.png')
    plt.close(fig)

    # adapt
    t_new = jnp.zeros(len(t) + 1)
    idx = jnp.argmax(err) + 1
    t_new = t_new.at[0:idx].set(t[0:idx])
    t_new = t_new.at[idx + 1:].set(t[idx:])
    t_new = t_new.at[idx].set(jnp.mean(t[idx - 1:idx + 1]))

    t = t_new
    dt = jnp.diff(t)
    dt_fine, t_fine = refineTime(dt, ref_factor)

    err_total = jnp.sum(err)
    it += 1

  plots = os.listdir(case)
  frame = cv2.imread(os.path.join(case, plots[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(case + '/' + case + '.mp4',
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 12,
                          (width, height))
  for i, p in enumerate(plots):
    p_path = os.path.join(case, p)
    video.write(cv2.imread(p_path))
    if i > 0 and i < len(plots) - 1:
      os.remove(p_path)

  cv2.destroyAllWindows()
  video.release()
