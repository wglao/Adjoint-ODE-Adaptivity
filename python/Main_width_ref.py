import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node", default=1, type=int)
parser.add_argument("--GPU_index", default=0, type=int)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

NODE = args.node
GPU_index = args.GPU_index

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_index)

import shutil
from functools import partial
from typing import Iterable, Dict
from animate import animate
import cv2
import flax.core.frozen_dict as fdict
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy.stats as jst
import jax.tree_util as jtr
import matplotlib.pyplot as plt
import numpy as np
import optax
import plotly.express as px
import plotly.graph_objects as go
import scipy.integrate as integrate
from jax import grad, jit, vmap, value_and_grad
from jax.lax import dynamic_slice_in_dim as dySlice
from jax.lax import scan
from models import ResNetBlock, ResNetODE, ResBlockSimple


@jit
def odeFn(u, t):
  return 10*jnp.cos(u)


def forwardFn(u, t, dt, params: dict, net: nn.Module):
  return net.apply({'params': params}, u[-1], t[-1], dt)


def forwardSolve(u_0, dt, params_list: Iterable[dict],
                 net_list: Iterable[nn.Module]):
  # return net.apply({'params': params}, u0, return_all=True)
  u = u_0*jnp.ones((len(dt) + 1, 1))
  t = jnp.pad(jnp.cumsum(dt), (1, 0), constant_values=0)
  u_prev = u

  # for loop
  for l, dt_l in enumerate(dt):
    u_next = forwardFn(u[:l + 1], t[:l + 1], dt_l, params_list[l], net_list[l])
    u = u.at[l + 1].set(u_next)

  # scan
  # def scanFn(u, l):
  #   u_next = forwardFn(u[:l + 1], t[:l + 1], dt[l], params, net)
  #   u = u.at[l + 1].set(u_next)
  #   return u, u_next

  # u, _ = scan(scanFn,u,jnp.arange(len(dt)))

  return u


def outFnl(u, t, true):
  loss = jnp.abs(jnp.squeeze(u[-1]) - jnp.squeeze(true))
  return loss


# @partial(jit, static_argnums=2)
def adjointSolve(u, dt, true, ref_factor, params_list, net_list):
  # refine u
  _, dt_fine, t_fine, u_fine = refineSolution(u, dt, ref_factor)

  dJdU = grad(outFnl)(u_fine, t_fine, true)
  v0 = dJdU[-1]
  v = v0*jnp.ones_like(u_fine)

  # t = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dt)), None)

  def sumTerm(u, t, dt, i, j):
    u_prev = u[:j]
    t_prev = t[:j]
    dt_j = dt[j - 1]
    params = params_list[(j-1) // ref_factor]
    net = net_list[(j-1) // ref_factor]
    return (grad(lambda u, t, dt, p, n: forwardFn(u, t, dt, p, n)[0])(u_prev,
                                                                      t_prev,
                                                                      dt_j,
                                                                      params,
                                                                      net))[i]

  for i in jnp.arange(len(v) - 2, -1, -1):
    js = jnp.arange(i + 1, len(v), dtype=jnp.int32)
    v_next = dJdU[i]
    for j in js:
      v_next = v_next + v[j]*sumTerm(u_fine, t_fine, dt_fine, i, j)
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
def errorIndicator(u, v, dt, ref_factor, params_list, net_list):
  err = jnp.zeros(len(u) - 1)
  t_coarse, dt_fine, t_fine, u_fine = refineSolution(u, dt, ref_factor)
  res_u = jnp.zeros_like(u_fine)
  for n in jnp.arange(1, len(res_u)):
    residual = u_fine[n] - forwardFn(u_fine[:n], t_fine[:n], dt_fine[n - 1],
                                     params_list[(n-1) // ref_factor],
                                     net_list[(n-1) // ref_factor])
    res_u = res_u.at[n].set(jnp.squeeze(residual))
  err_fine = res_u*v
  for i in jnp.arange(len(err)):
    err = err.at[i].set(
        jnp.sum(err_fine[i*ref_factor + 1:(i+1)*ref_factor + 1]))
  return jnp.abs(err)


def lossFn(u_0, t, dt, true, params_list, net_list):
  u = forwardSolve(u_0, dt, params_list, net_list)
  loss = jnp.square(jnp.squeeze(u[-1]) - jnp.squeeze(true))
  return loss


def layerLoss(u_l, bias_l):
  loss = jnp.min(jnp.square(jnp.squeeze(u_l) - bias_l))
  i = jnp.argmin(jnp.square(jnp.squeeze(u_l) - bias_l))
  sgn = jnp.sign(jnp.squeeze(u_l) - bias_l[i])
  return loss, i, sgn


def lastLayerLoss(true, u_last, u_minus_1, bias_last):
  loss = jnp.square(u_last - true)
  i = jnp.argmin(jnp.square(bias_last - u_minus_1))
  sgn = jnp.sign(u_minus_1 - bias_last[i])
  return loss, i, sgn


def trainStep(u_0, t, dt, true, params_list, net_list, opt_state_list,
              tx: optax.GradientTransformation):
  loss_list, grads_list = vmap(
      value_and_grad(lossFn, argnums=(4,)),
      in_axes=(0, None, None, 0, None, None))(u_0, t, dt, true, params_list,
                                              net_list)
  grads_list = jtr.tree_map(lambda m: jnp.mean(m, axis=0), grads_list[0])
  loss = jnp.mean(loss_list)
  for i in range(len(params_list)):
    updates, opt_state_list[i] = tx.update(grads_list[i], opt_state_list[i])
    params_list[i] = optax.apply_updates(params_list[i], updates)
  return params_list, opt_state_list, loss


def metricCalc(u_0, t, dt, true, params_list, net_list):
  loss = lossFn(u_0[0], t, dt, true[0], params_list, net_list)
  err = jnp.mean(
      vmap(lossFn,
           in_axes=(0, None, None, 0, None, None))(u_0[1:], t, dt, true[1:],
                                                   params_list, net_list))
  return loss, err


def adaptDepth(t, params_list: list, net_list: Iterable[nn.Module],
               opt_state_list: list, tx: optax.GradientTransformation, i):
  """Add new layer with h refinement"""
  t_new = jnp.zeros((len(t) + 1,))
  t_new = t_new.at[0:i].set(t[0:i])
  t_new = t_new.at[i + 1:].set(t[i:])
  t_new = t_new.at[i].set(jnp.mean(t[i - 1:i + 1]))

  dt_new = jnp.diff(t_new)

  # rng = jrand.PRNGKey(len(t_new))
  # params_new = net.init(rng, jnp.ones(1), jnp.ones(1), jnp.ones(1))['params']

  # params_new = jtr.tree_map(lambda p: 1e-8*jrand.normal(rng, p.shape),
  #                           params_list[0])

  params_new = params_list[i - 1]

  # params_new = jtr.tree_map(lambda p: jnp.zeros_like(p), params_list[0])
  # for p in params_list:
  #   params_new = jtr.tree_map(lambda q,r: q+r/len(params_list),params_new,p)

  opt_state_new = tx.init(params_new)
  params_list.insert(i, params_new)
  opt_state_list.insert(i, opt_state_new)
  net_list.insert(i, net_list[i - 1])
  return t_new, dt_new, params_list, opt_state_list, net_list


def fillBins(loss, i, sgn, u, bias):
  bins = jnp.zeros((len(bias) + 1, 1))
  u_bins = jnp.zeros_like(bins)
  l_bins = jnp.zeros_like(bins)
  k = i + jnp.where(sgn > 0, 0, -1) + 1
  bins = bins.at[k].set(1)
  u_bins = u_bins.at[k].set(u)
  l_bins = l_bins.at[k].set(loss)
  return bins, u_bins, l_bins


def adaptWidth(u_0,
               dt,
               true,
               params_list: list,
               opt_state_list: list,
               net_list: Iterable[nn.Module],
               tol=5e-5):
  # hidden layers
  u_arr = vmap(
      forwardSolve, in_axes=(0, None, None, None))(u_0, dt, params_list,
                                                   net_list)
  for l in range(len(dt) - 1):
    bias_l = params_list[l]['bias']
    sorted_idxs = jnp.argsort(bias_l)
    bias_l = bias_l[sorted_idxs]
    weights_l = params_list[l]['weights'].T[sorted_idxs].T

    loss_arr, i_arr, sgn_arr = vmap(
        layerLoss, in_axes=(0, None))(u_arr[:, l], bias_l)
    bins, u_bins, l_bins = jtr.tree_map(
        lambda t: jnp.sum(t, axis=0),
        vmap(fillBins, in_axes=(0, 0, 0, 0, None))(loss_arr, i_arr, sgn_arr,
                                                   u_arr[:, l], bias_l))
    u_bins = u_bins / jnp.where(bins > 0, bins, 1)
    l_bins = l_bins / jnp.where(bins > 0, bins, 1)

    # add a neuron where loss is greatest
    if jnp.max(l_bins) > tol:
      k = jnp.argmax(l_bins)
      new_net = ResBlockSimple(len(bins))
      _ = new_net.init(jrand.PRNGKey(0), 1, 1, 1)
      bias_l = jnp.pad(bias_l, (0, 1), 'edge')
      weights_l = jnp.pad(weights_l, ((0, 0), (0, 1)), 'edge')
      if k < len(bins) - 1:
        bias_l = bias_l.at[k + 1:].set(bias_l[k:-1])
        weights_l = weights_l.at[0, k + 1:].set(weights_l[0, k:-1])
      bias_l = bias_l.at[k].set(jnp.squeeze(u_bins[k]))
      weights_l = weights_l.at[0, k].set(1e-5)
      new_params = {
          'bias': bias_l,
          'weights': weights_l,
      }
      new_params = fdict.freeze(new_params)

      params_list[l] = new_params
      opt_state_list[l] = optimizer.init(new_params)
      net_list[l] = new_net

  # last layer
  bias_last = params_list[-1]['bias']
  sorted_idxs = jnp.argsort(bias_last)
  bias_last = bias_last[sorted_idxs]
  weights1_last = params_list[-1]['weights1'].T[sorted_idxs].T
  weights2_last = params_list[-1]['weights2'][sorted_idxs]
  loss_arr, i_arr, sgn_arr = vmap(
      lastLayerLoss, in_axes=(0, 0, 0, None))(true_train, u_arr[:, -1],
                                              u_arr[:, -2], bias_last)
  bins, u_bins, l_bins = jtr.tree_map(
      lambda t: jnp.sum(t, axis=0),
      vmap(fillBins, in_axes=(0, 0, 0, 0, None))(loss_arr, i_arr, sgn_arr,
                                                 u_arr[:, -1], bias_last))
  u_bins = u_bins / jnp.where(bins > 0, bins, 1)
  l_bins = l_bins / jnp.where(bins > 0, bins, 1)
  # add a neuron where loss is greatest
  if jnp.max(l_bins) > tol:
    k = jnp.argmax(l_bins)
    new_net = ResBlockSimple(len(bins))
    bias_last = jnp.pad(bias_last, (0, 1), 'edge')
    weights1_last = jnp.pad(weights1_last, ((0, 0), (0, 1)), 'edge')
    weights2_last = jnp.pad(weights2_last, ((0, 1), (0, 0)), 'edge')
    if k < len(bins) - 1:
      bias_last = bias_last.at[k + 1:].set(bias_last[k:-1])
      weights1_last = weights1_last.at[0, k + 1:].set(weights1_last[0, k:-1])
      weights2_last = weights2_last.at[k + 1:].set(weights2_last[k:-1])
    bias_last = bias_last.at[k].set(jnp.squeeze(u_bins[k]))
    weights1_last = weights1_last.at[0, k].set(1e-5)
    weights2_last = weights2_last.at[k].set(1e-5)
    new_params = {
        'bias': bias_last,
        'weights1': weights1_last,
        'weights2': weights2_last,
    }
    new_params = fdict.freeze(new_params)

    params_list[-1] = new_params
    opt_state_list[-1] = optimizer.init(new_params)
    net_list[-1] = new_net
  return params_list, opt_state_list, net_list


if __name__ == "__main__":
  case = "ResNetODE_mix_ref_untl-10pct-chng_start2x100_med_data_" + str(
      args.seed)
  wandb_upload = True
  if wandb_upload:
    import wandb
    wandb.init(project="Adjoint Adaptivity", entity="wglao", name=case)
    wandb.config.problem = 'ResNet'
    wandb.config.method = 'mixed refinement'

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
  seed = int(args.seed)
  rng = jrand.PRNGKey(seed)
  net = ResBlockSimple(100)
  params = net.init(rng, jnp.ones(1), jnp.ones(1), jnp.ones(1))['params']

  n_epochs_base = 500
  w_per_d_ref = 10
  learning_rate = 1e-3
  # schedule = optax.cosine_onecycle_schedule(n_epochs*25, learning_rate)
  optimizer = optax.adam(learning_rate)
  # optimizer = optax.eve(learning_rate)
  opt_state = optimizer.init(params)

  net_list = [net]*n_steps
  params_list = [params]*n_steps
  opt_state_list = [opt_state]*n_steps

  u_0_train = jrand.uniform(rng, (100000,), minval=-1, maxval=1)
  u_0_test = jnp.concatenate(
      (jnp.reshape(jnp.max(u_0_train), (1, 1)), -1*jnp.ones(
          (1, 1)), jrand.normal(rng, (99, 1))) / 3, None)

  true_train = integrate.odeint(odeFn, u_0_train, t_span)[-1]
  true_test = integrate.odeint(odeFn, u_0_test, t_span)[-1]

  # define decaying values for err and loss
  loss_hist = jnp.zeros((n_epochs_base,))
  ep_total = 0
  old_bar_ylim = []
  while err_total > tol and it <= maxit:
    # n_epochs = n_epochs_base*(it//w_per_d_ref + 1)
    n_epochs = n_epochs_base
    # train
    for ep in range(n_epochs):
      params, opt_state, loss = trainStep(u_0_train, t, dt, true_train,
                                          params_list, net_list, opt_state_list,
                                          optimizer)
      _, err = metricCalc(u_0_test, t, dt, true_test, params_list, net_list)

      if wandb_upload:
        wandb.log({
            'Epoch': ep + ep_total,
            'Loss': loss,
            'Error': err,
            'Refinements': it,
            # 'Learning Rate': learning_rate/(opt_state[-1][0][-2])
        })
      else:
        print('Epoch: {:d}'.format(ep + ep_total), 'Loss: {:.2e}'.format(loss),
              'Error: {:.2e}'.format(err), 'Refinements: {:d}'.format(it))

      loss_hist = loss_hist.at[0:-1].set(loss_hist[1:])
      loss_hist = loss_hist.at[-1].set(loss)

    ep_total = ep_total + n_epochs

    # solve
    u_train_plot = forwardSolve(u_0_test[0], dt, params_list, net_list)
    v_train_plot = adjointSolve(u_train_plot, dt, true_test[0], ref_factor,
                                params_list, net_list)
    err_train_plot = errorIndicator(u_train_plot, v_train_plot, dt, ref_factor,
                                    params_list, net_list)
    u_test_plot = forwardSolve(u_0_test[1], dt, params_list, net_list)
    v_test_plot = adjointSolve(u_test_plot, dt, true_test[1], ref_factor,
                               params_list, net_list)
    err_test_plot = errorIndicator(u_test_plot, v_test_plot, dt, ref_factor,
                                   params_list, net_list)
    err_plot = 0.5*(err_test_plot+err_train_plot)
    # plot
    fig, ax1 = plt.subplots()
    ax1.bar(
        t[:-1] + dt/2,
        err_plot,
        dt,
        color='darkseagreen',
        label='Error Indicator')
    ax1.set_ylabel('Error Contribution')
    new_bar_ylim = ax1.get_ylim()
    if it == 0:
      old_bar_ylim = new_bar_ylim
    elif new_bar_ylim[-1] < 0.01*old_bar_ylim[-1]:
      old_bar_ylim = new_bar_ylim
    else:
      ax1.set_ylim(*old_bar_ylim)

    ax2 = ax1.twinx()

    # exact
    ax2.plot(
        t_span,
        jnp.array([u_0_test[0], true_test[0]]),
        color='midnightblue',
        marker='o',
        linestyle='None',
        label='Seen Solution')
    ax2.plot(
        t_span,
        jnp.array([u_0_test[1], true_test[1]]),
        color='saddlebrown',
        marker='o',
        linestyle='None',
        label='Unseen Solution')

    ax2.plot(t, u_train_plot, '-', marker='.', color='tab:blue', linewidth=1.25)
    ax2.plot(
        t_fine, jnp.abs(v_train_plot), '--', color='darkblue', linewidth=1.25)
    ax2.set_ylabel('Solution')
    ax2.plot(
        t, u_test_plot, '-', marker='.', color='tab:orange', linewidth=1.25)
    ax2.plot(t_fine, jnp.abs(v_test_plot), '--', color='peru', linewidth=1)
    ax2.set_ylabel('Solution')

    ax2.set_xlabel('Time')

    fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

    f_name = case + '_{:d}'.format(it)
    fig.savefig(case + '/' + f_name + '.png')
    # if wandb_upload:
    #   wandb.log({'Refinement Plot': ax2})
    plt.close(fig)

    # adapt
    u_refine = vmap(
        forwardSolve, in_axes=(0, None, None, None))(u_0_train, dt, params_list,
                                                     net_list)
    v_refine = vmap(
        adjointSolve,
        in_axes=(0, None, 0, None, None, None))(u_refine, dt, true_train,
                                                ref_factor, params_list,
                                                net_list)
    err_refine = vmap(
        errorIndicator,
        in_axes=(0, 0, None, None, None, None))(u_refine, v_refine, dt,
                                                ref_factor, params_list,
                                                net_list)
    err_refine = jnp.mean(err_refine, axis=0)
    err_total = jnp.sum(err_refine)

    # Alternate width and depth
    # if it % w_per_d_ref == w_per_d_ref - 1:
    # Only refine depth if loss is low
    # if loss < 1e-8:
    # Only refine depth after loss decrease is less than 10% improvement
    if (loss_hist[0] - loss_hist[-1]) / loss_hist[0] < 0.1:
      # Only Width
      # if False:
      # depth
      idx = jnp.argmax(err_refine) + 1

      t, dt, params_list, opt_state_list, net_list = adaptDepth(
          t, params_list, net_list, opt_state_list, optimizer, idx)

      dt_fine, t_fine = refineTime(dt, ref_factor)
    else:
      # width
      params_list, opt_state_list, net_list = adaptWidth(
          u_0_train, dt, true_train, params_list, opt_state_list, net_list)

    it += 1

  animate(case)
