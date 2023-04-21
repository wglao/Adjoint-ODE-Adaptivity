import os
import shutil
from functools import partial

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from jax import grad, jit, vmap
from jax.lax import dynamic_slice_in_dim as dySlice
from matplotlib.patches import Rectangle


def forwardFn(u, t, dt):
  if len(u) == 1:
    return u[-1]
  return u[-1] + jnp.trapz(u, t)*dt
  # return u[-1] + dt*u[-1]


@jit
def forwardSolve(u0, dt_n):
  u = u0*jnp.ones((len(dt_n) + 1))
  times = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dt_n)), None)
  for n, dt in enumerate(dt_n):
    u_next = forwardFn(u[:n + 1], times[:n + 1], dt)
    u = u.at[n + 1].set(u_next)
  return u


def outFnl(u, t=None):
  return u[-1]


# @partial(jit, static_argnums=2)
def adjointSolve(u, dt_n, ref_factor):
  # refine u
  _, dt_fine, t_fine, u_fine = refineSolution(u, dt_n, ref_factor)

  dJdU = grad(outFnl)(u_fine, t_fine)
  v0 = dJdU[-1]
  v = v0*jnp.ones_like(u_fine)
  times = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dt_n)), None)

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


def refineTime(dt_n, ref_factor):
  dt_fine = jnp.zeros((len(dt_n)*ref_factor))
  for i, dt in enumerate(dt_n):
    dt_fine = dt_fine.at[i*ref_factor:(i+1)*ref_factor].set(dt / ref_factor)
  t_fine = jnp.cumsum(dt_fine)
  t_fine = jnp.pad(t_fine, (1, 0), constant_values=0)
  return dt_fine, t_fine


def refineSolution(u, dt_n, ref_factor):
  t_coarse = jnp.cumsum(dt_n)
  t_coarse = jnp.pad(t_coarse, (1, 0), constant_values=0)
  dt_fine, t_fine = refineTime(dt_n, ref_factor)
  u_fine = jnp.interp(t_fine, t_coarse, u)
  return t_coarse, dt_fine, t_fine, u_fine


# @partial(jit, static_argnums=2)
def errorIndicator(u, v, ref_factor):
  err = jnp.zeros(len(u) - 1)
  t_coarse, dt_fine, t_fine, u_fine = refineSolution(u, dt_n, ref_factor)
  res_u = jnp.zeros_like(u_fine)
  for n in jnp.arange(1, len(res_u)):
    residual = u_fine[n] - forwardFn(u_fine[:n], t_fine[:n], dt_fine[n - 1])
    res_u = res_u.at[n].set(residual)
  err_fine = res_u*v
  for i in jnp.arange(len(err)):
    err = err.at[i].set(
        jnp.sum(err_fine[i*ref_factor + 1:(i+1)*ref_factor + 1]))
  return jnp.abs(err)


if __name__ == "__main__":
  case = "no_matrix"
  u0 = 1
  t_span = jnp.array([0, 1])
  n_steps = 4
  times = jnp.linspace(t_span[0], t_span[1], n_steps + 1)
  dt_n = jnp.diff(times)
  ref_factor = 4
  dt_fine, t_fine = refineTime(dt_n, ref_factor)

  it = 0
  maxit = 100
  err_total = 1
  tol = 1e-8

  # folder for saving plots as images
  if os.path.isdir(case):
    shutil.rmtree(case)
  os.mkdir(case)

  while err_total > tol and it <= maxit:
    # solve
    u = forwardSolve(u0, dt_n)
    v = adjointSolve(u, dt_n, ref_factor)
    err = errorIndicator(u, v, ref_factor)

    # plot
    fig, ax1 = plt.subplots()
    ax1.bar(
        times[:-1] + dt_n/2,
        err,
        dt_n,
        color='darkseagreen',
        label='Error Indicator')
    ax1.set_ylabel('Error Contribution')
    if it == 0:
      bar_ylim = ax1.get_ylim()
    else:
      ax1.set_ylim(*bar_ylim)

    ax2 = ax1.twinx()
    ax2.plot(times, u, '-', marker='.', color='tab:blue', label='Forward')
    ax2.plot(t_fine, v, '-', marker='.', color='tab:orange', label='Ajoint')
    ax2.set_ylabel('Solution')
    ax2.set_xlabel('Time')

    fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

    f_name = case + '_{:d}'.format(it)
    fig.savefig(case + '/' + f_name + '.png')
    plt.close(fig)

    # adapt
    times_new = jnp.zeros(len(times) + 1)
    idx = jnp.argmax(err) + 1
    times_new = times_new.at[0:idx].set(times[0:idx])
    times_new = times_new.at[idx + 1:].set(times[idx:])
    times_new = times_new.at[idx].set(jnp.mean(times[idx - 1:idx + 1]))

    times = times_new
    dt_n = jnp.diff(times)
    dt_fine, t_fine = refineTime(dt_n, ref_factor)

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
