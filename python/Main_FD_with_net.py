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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from factory import FunFactory, Funs, Problem, ResNetBlock


def refineAll(dt_n, ref_factor):
  n_steps = len(dt_n)*ref_factor
  dt_fine = np.zeros(n_steps)
  for f in range(ref_factor):
    dt_fine[f:n_steps - ref_factor + f + 1:ref_factor] = dt_n / ref_factor
  return dt_fine, n_steps


def interpU(dt_fine, dt_n, u):
  dt_fine, _ = refineAll(dt_n, ref_factor)

  # interpolate u onto fine grid
  t_coarse = np.concatenate(([0], np.cumsum(dt_n)), axis=None)
  t_fine = np.concatenate(([0], np.cumsum(dt_fine)), axis=None)
  u_fine = np.interp(t_fine, t_coarse, u)
  return u_fine


def forwardSolve(updateRule, dt_n, u0=None, is_net=False, net=None, params=None):
  """Step through time with finite difference
  """
  n_nodes = len(dt_n) + 1

  if not is_net:
    def solve(u0):
      u_vec = np.zeros(n_nodes)
      for n in range(n_nodes):
        if n != 0:
          u_vec[n] = updateRule(u_vec, dt_n, n)
        else:
          u_vec[0] = u0
      return u_vec
    if u0 is None:
      return solve
    else:
      return solve(u0)
  else:
    def solve(u0, net, params):
      u_vec = np.zeros(n_nodes)
      for n in range(n_nodes):
        if n != 0:
          u_vec[n] = updateRule(u_vec, dt_n, n, net, params)
        else:
          u_vec[0] = u0
      return u_vec
    if u0 is None:
      return solve
    else:
      return solve(u0, net, params)
  


def adjSolve(getK, getJF, dt_n, u, ref_factor, is_net=False, params=None):
  """Create a Linear System to solve the adjoint equation
    :math:`K^T U + \lambda^T AU = 0`
    
    or rather, its transpose
    :math:`A^T \lambda = -K`

    K is given by the partial derivative of the output functional
    :math:`\\frac{\partial J}{\partial u} = [0, 0, ..., 0, 1]`
  """
  # n_steps = len(dt_n)

  # refine grid for adjoint
  dt_fine, _ = refineAll(dt_n, ref_factor)
  u_fine = interpU(dt_fine, dt_n, u)
  if not is_net:
    f_jacobian = getJF(u_fine, dt_fine)
  else:
    f_jacobian = getJF(u_fine, dt_fine, params)
  k_vec = getK(dt_fine, u_fine)

  def solve(v0=0):
    v_vec = np.linalg.solve(f_jacobian.T - np.eye(f_jacobian.shape[0]), -k_vec)
    return v_vec

  return solve(0)


def errEst(fwdUpdate, u, v, dt_n, ref_factor, is_net=False, net=None, params=None):
  """Calculate the Adjoint-Weighted Residual as an error estimate
    for each time step
  """
  # n_steps = len(dt_n)

  # refine grid for adjoint
  dt_fine, n_steps = refineAll(dt_n, ref_factor)
  u_fine = interpU(dt_fine, dt_n, u)

  res_u = np.zeros_like(u_fine)
  for n in np.arange(n_steps) + 1:
    if not is_net:
      res_u[n] = u_fine[n] - fwdUpdate(u_fine, dt_fine, n)
    else:
      res_u[n] = u_fine[n] - fwdUpdate(u_fine, dt_fine, n, net, params)
  err = res_u*v

  return err


if __name__ == "__main__":
  case = "ff_test"
  is_net = True
  linear_ode = False
  linear_out_functional = False
  ode = 'ResNet'
  out_functional = 'J=int(u^2)'  # J = { int(u), int(u^2), u_N, ... }

  # time discretization
  n_steps = 2
  t_span = np.array([0, 2])
  n_nodes = n_steps + 1
  times = np.linspace(t_span[0], t_span[1], n_nodes)

  problem = Problem(is_net, linear_ode, linear_out_functional, ode,
                            out_functional, t_span)
  ff = FunFactory(problem)
  funs = ff.getFunctions()
  del ff, problem

  if is_net:
    net = ResNetBlock([100,1])
    rng = jax.random.PRNGKey(1)
    params = net.init(rng, 1)['params']
    del rng
  else:
    net = None
    params = None

  # folder for saving plots as images
  if os.path.isdir(case):
    shutil.rmtree(case)
  os.mkdir(case)

  ref_factor = 4  # <-- must be > 2

  # exact solution
  x_plot = np.linspace(times[0], times[-1], 500)
  u0 = 1
  exact_fwd = funs.exactFwd(x_plot, u0)
  if linear_ode and linear_out_functional:
    exact_adj = funs.exactAdj(x_plot)
  else:
    exact_adj = funs.exactAdj(x_plot, exact_fwd)

  it = 0
  maxit = 100
  err = 1
  tol = 1e-5

  while it <= maxit and err > tol:
    dt_n = np.diff(times, 1)
    n_steps = len(dt_n)
    u = forwardSolve(funs.fwdUpdate, dt_n, u0, is_net, net, params)
    # sum errors within each step
    # stride = ref_factor, window_len = ref_factor - 1
    v = adjSolve(funs.getK, funs.getJF, dt_n, u, ref_factor, is_net, params)
    err_steps = np.abs(errEst(funs.fwdUpdate, u, v, dt_n, ref_factor, is_net, net, params))[2:]

    n_rows = (err_steps.size - (ref_factor-1)) // ref_factor + 1
    n = err_steps.strides[0]
    err_steps = np.lib.stride_tricks.as_strided(err_steps,
                                                shape=(n_rows, ref_factor - 1),
                                                strides=(ref_factor*n, n))
    err_steps = np.sum(err_steps, 1)
    dt_fine, n_steps_fine = refineAll(dt_n, ref_factor)
    times_fine = np.concatenate(([0], np.cumsum(dt_fine)), axis=None)

    # plot
    fig, ax1 = plt.subplots()

    bar_x = times[0:-1] + dt_n/2
    ax1.bar(bar_x,
            err_steps,
            dt_n,
            color='darkseagreen',
            label='Error Estimate')
    if it == 0:
      bar_ylim = ax1.get_ylim()
    else:
      ax1.set_ylim(*bar_ylim)
    ax1.set_ylabel('Error Contribution')

    ax2 = ax1.twinx()

    ax2.plot(x_plot,
             exact_fwd,
             '-',
             color='k',
             label='Exact Primal',
             linewidth=4)
    ax2.plot(x_plot,
             exact_adj,
             '-',
             color='saddlebrown',
             label='Exact Adjoint',
             linewidth=4)

    # FD solution
    ax2.plot(times,
             u,
             '-',
             marker='.',
             color='tab:blue',
             label='FD Primal',
             linewidth=1.25)
    ax2.plot(times_fine,
             v,
             '-',
             marker='.',
             color='tab:orange',
             label='FD Adjoint',
             linewidth=1.25)
    ax2.set_ylabel('Solution')
    ax2.set_xlabel('Time')

    fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

    f_name = case + '_{:d}'.format(it)
    fig.savefig(case + '/' + f_name + '.png')
    plt.close(fig)

    # adapt
    times_new = np.zeros(n_steps + 2)
    ref_idx = np.argmax(err_steps) + 1
    times_new[0:ref_idx] = times[0:ref_idx]
    times_new[ref_idx + 1:] = times[ref_idx:]
    times_new[ref_idx] = np.mean(times[ref_idx - 1:ref_idx + 1])
    times = times_new
    err = np.sum(err_steps)
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
