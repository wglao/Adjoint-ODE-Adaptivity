"""
Adaptively solve ODE with finite difference in time,
using Adjoint-Weighted Residual for a posterior error
estimates to guide time step refinement
"""

import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.integrate as integrate
import shutil


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


def forwardSolve(updateRule, dt_n, u0=None):
  """Step through time with finite difference
  """
  n_nodes = len(dt_n) + 1

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


def adjSolve(getK, getJF, dt_n, u, ref_factor):
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
  f_jacobian = getJF(u_fine, dt_fine)
  k_vec = getK(dt_fine, u_fine)

  def solve(v0=0):
    v_vec = np.linalg.solve(f_jacobian.T - np.eye(f_jacobian.shape[0]), -k_vec)
    return v_vec

  return solve(0)


def errEst(fwdUpdate, u, v, dt_n, ref_factor):
  """Calculate the Adjoint-Weighted Residual as an error estimate
    for each time step
  """
  # n_steps = len(dt_n)

  # refine grid for adjoint
  dt_fine, n_steps = refineAll(dt_n, ref_factor)
  u_fine = interpU(dt_fine, dt_n, u)

  res_u = np.zeros_like(u_fine)
  for n in np.arange(n_steps) + 1:
    res_u[n] = u_fine[n] - fwdUpdate(u_fine, dt_fine, n)
  err = res_u*v

  return err


if __name__ == "__main__":
  case = "FD_nonlinear_u_sq"
  ode = 'du/dt=sin(u)'
  linear_ode = False
  linear_out_functional = False
  out_functional = 'J=int(u^2)'  # J = { int(u), int(u^2), u_N, ... }

  # time discretization
  n_steps = 2
  t_span = np.array([0, 2])
  n_nodes = n_steps + 1
  times = np.linspace(t_span[0], t_span[1], n_nodes)

  # Fwd and Adj Requirements
  if ode == 'du/dt=u':

    def fwdUpdate(u, dt_n, n):
      return (1 + dt_n[n - 1])*u[n - 1]

    def getF(u, dt_n):
      f_vec = (1+dt_n)*u[:-1]
      return np.concatenate((u[0], f_vec), axis=None)

    def getJF(u, dt_n):
      jf_diag = 1 + dt_n
      return np.diag(jf_diag, -1)

    if linear_ode:

      def exactFwd(t, u0):
        return u0*np.exp(t)

  elif ode == 'du/dt=sin(u)':

    def fwdUpdate(u, dt_n, n):
      return u[n - 1] + np.sin(u[n - 1])*dt_n[n - 1]

    def getF(u, dt_n):
      f_vec = np.sin(u[:-1])*dt_n
      return np.concatenate((u0, f_vec), axis=None)

    def getJF(u, dt_n):
      jf_diag = 1 + np.cos(u[:-1])*dt_n
      return np.diag(jf_diag, -1)

    def exactFwd(t, u0):
      return 2*np.arctan2(np.sin(u0 / 2)*np.exp(t), np.cos(u0 / 2))

  # output functional requirements
  def integral(fn, a, b):
    return integrate.quad(fn, a, b)[0]

  if linear_ode:
    if linear_out_functional:
      if out_functional == 'J=int(u)':

        def getK(dt_n, u=None, v0=0):
          k = dt_n
          return np.concatenate((k, v0), axis=None)

        def exactAdj(t):
          return np.exp(-t)*np.exp(t_span[-1]) - 1

      elif out_functional == 'J=u_N':

        def getK(dt_n, u=None, v0=0):
          k = np.zeros_like(dt_n)
          k[-1] = 1
          return np.concatenate((k, v0), axis=None)

        def exactAdj(t):
          return -(np.sign(t - t_span[-1])*np.exp(-t)*np.exp(t_span[-1]))

    else:
      if out_functional == 'J=int(u^2)':

        def getK(dt_n, u, v0=0):
          k = 2*u[:-1]*dt_n
          return np.concatenate((k, v0), axis=None)

        def exactAdj(t, u):
          a = np.zeros_like(u)
          u_interp = lambda x: np.interp(x, t, u)
          fn_1 = lambda y: np.exp(y)*u_interp(y)* -2
          for i in range(len(u) - 1):
            a[i] = np.exp(-t[i])*integral(fn_1, t_span[-1], t[i])
          return a

  else:
    if linear_out_functional:
      if out_functional == 'J=int(u)':

        def getK(dt_n, u=None, v0=0):
          k = dt_n
          return np.concatenate((k, v0), axis=None)

        if ode == 'du/dt=sin(u)':

          def exactAdj(t, u):
            a = np.zeros_like(u)
            u_interp = lambda x: np.interp(x, t, u)
            fn_1 = lambda x: np.cos(u_interp(x))
            for i in range(len(u) - 1):
              fn_2 = lambda y: -np.exp(integral(fn_1, t_span[-1], y))
              a[i] = np.exp(-1*integral(fn_1, t_span[-1], t[i]))*integral(
                  fn_2, t_span[-1], t[i])
            return a

      elif out_functional == 'J=u_N':

        def getK(dt_n, u=None, v0=0):
          k = np.zeros_like(dt_n)
          k[-1] = 1
          return np.concatenate((k, v0), axis=None)

        if ode == 'du/dt=sin(u)':

          def exactAdj(t, u):
            u_interp = lambda x: np.interp(x, t, u)
            a = np.zeros_like(u)
            for i in range(len(u) - 1):
              fn_1 = lambda y: np.cos(u_interp(y))
              a[i] = -np.exp(-integral(fn_1, t_span[-1], t[i]))*(
                  1 - np.heaviside(t_span[-1] - t[i], 1)*2)
            return a
    else:
      if out_functional == 'J=int(u^2)':

        def getK(dt_n, u, v0=0):
          k = 2*u[:-1]*dt_n
          return np.concatenate((k, v0), axis=None)

        if ode == 'du/dt=sin(u)':

          def exactAdj(t, u):
            a = np.zeros_like(u)
            u_interp = lambda x: np.interp(x, t, u)
            fn_1 = lambda y: np.cos(u_interp(y))
            fn_2 = lambda z: np.exp(integral(fn_1, t_span[-1], z))*u_interp(z
                                                                           )* -2
            for i in range(len(u) - 1):
              a[i] = np.exp(-integral(fn_1, t_span[-1], t[i]))*integral(
                  fn_2, t_span[-1], t[i])
            return a

  # folder for saving plots as images
  if os.path.isdir(case):
    shutil.rmtree(case)
  os.mkdir(case)

  ref_factor = 4  # <-- must be > 2

  # exact solution
  x_plot = np.linspace(times[0], times[-1], 500)
  u0 = 1
  exact_fwd = exactFwd(x_plot, u0)
  if linear_ode and linear_out_functional:
    exact_adj = exactAdj(x_plot)
  else:
    exact_adj = exactAdj(x_plot, exact_fwd)

  it = 0
  maxit = 100
  err = 1
  tol = 1e-5

  while it <= maxit and err > tol:
    dt_n = np.diff(times, 1)
    n_steps = len(dt_n)
    u = forwardSolve(fwdUpdate, dt_n, u0)
    # sum errors within each step
    # stride = ref_factor, window_len = ref_factor - 1
    v = adjSolve(getK, getJF, dt_n, u, ref_factor)
    err_steps = np.abs(errEst(fwdUpdate, u, v, dt_n, ref_factor))[2:]

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
