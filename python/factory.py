import os
import shutil
from functools import partial
from typing import Iterable, NamedTuple, Union

import cv2
import flax
import flax.linen as nn
from jax import grad, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import models
import numpy as np
import scipy.integrate as integrate
"""Classes that handle necessary function creation"""


class Problem(NamedTuple):
  case: str
  is_net: bool
  linear_ode: bool
  linear_out_functional: bool
  ode: str
  out_functional: str
  ref_factor: Union[float, int]
  t_span: np.ndarray


class Funs(NamedTuple):
  exactAdj: callable
  exactFwd: callable
  fwdUpdate: callable
  getF: callable
  getJF: callable
  getK: callable


class AdaptFuns(NamedTuple):
  adapt: callable
  adjointSolve: callable
  animate: callable
  errorEstimate: callable
  forwardSolve: callable
  interpU: callable
  plot: callable
  refineAll: callable


class AdaptState():
  err_steps: np.ndarray
  it: int
  problem: Problem
  times: np.ndarray
  times_new: np.ndarray
  u: np.ndarray
  v: np.ndarray
  bar_ylim: np.ndarray = None

  def __init__(self, problem, times):
    self.it = 0
    self.problem = problem
    self.times = times
    self.times_new = times

  def iterate(self, err_steps, times, times_new, u, v):
    self.it += 1
    self.err_steps = err_steps
    self.times = times
    self.times_new = times_new
    self.u = u
    self.v = v


class FunFactory():

  def __init__(self, problem: Problem):
    self.problem = problem

  def getFunctions(self) -> Funs:
    problem = self.problem

    def integral(fn, a, b):
      return integrate.quad(fn, a, b)[0]

    if not problem.is_net:
      # Fwd and Adj Requirements
      if problem.ode == 'du/dt=u':

        def fwdUpdate(dt_n, u, n):
          return (1 + dt_n[n - 1])*u[n - 1]

        def getF(u, dt_n):
          f_vec = (1+dt_n)*u[:-1]
          return np.concatenate((u[0], f_vec), axis=None)

        def getJF(u, dt_n):
          jf_diag = 1 + dt_n
          return np.diag(jf_diag, -1)

        if problem.linear_ode:

          def exactFwd(t, u0):
            return u0*np.exp(t)

      elif problem.ode == 'du/dt=sin(u)':

        def fwdUpdate(dt_n, u, n):
          return u[n - 1] + np.sin(u[n - 1])*dt_n[n - 1]

        def getF(u, dt_n):
          f_vec = u[:-1] + np.sin(u[:-1])*dt_n
          return np.concatenate((u[0], f_vec), axis=None)

        def getJF(u, dt_n):
          jf_diag = 1 + np.cos(u[:-1])*dt_n
          return np.diag(jf_diag, -1)

        def exactFwd(t, u0):
          return 2*np.arctan2(np.sin(u0 / 2)*np.exp(t), np.cos(u0 / 2))

      # output functional requirements
      if problem.linear_ode:
        if problem.linear_out_functional:
          if problem.out_functional == 'J=int(u)':

            def getK(dt_n, u=None, v0=0):
              k = dt_n
              return np.concatenate((k, v0), axis=None)

            def exactAdj(t, u=None):
              return np.exp(-t)*np.exp(problem.t_span[-1]) - 1

          elif problem.out_functional == 'J=u_N':

            def getK(dt_n, u=None, v0=0):
              k = np.zeros_like(dt_n)
              k[-1] = 1
              return np.concatenate((k, v0), axis=None)

            def exactAdj(t, u=None):
              return -(
                  np.sign(t - problem.t_span[-1])*np.exp(-t)*
                  np.exp(problem.t_span[-1]))

        else:
          if problem.out_functional == 'J=int(u^2)':

            def getK(dt_n, u, v0=0):
              k = 2*u[:-1]*dt_n
              return np.concatenate((k, v0), axis=None)

            def exactAdj(t, u):
              a = np.zeros_like(u)
              u_interp = lambda x: np.interp(x, t, u)
              fn_1 = lambda y: np.exp(y)*u_interp(y)* -2
              for i in range(len(u) - 1):
                a[i] = np.exp(-t[i])*integral(fn_1, problem.t_span[-1], t[i])
              return a

      else:
        if problem.linear_out_functional:
          if problem.out_functional == 'J=int(u)':

            def getK(dt_n, u=None, v0=0):
              k = dt_n
              return np.concatenate((k, v0), axis=None)

            if problem.ode == 'du/dt=sin(u)':

              def exactAdj(t, u):
                a = np.zeros_like(u)
                u_interp = lambda x: np.interp(x, t, u)
                fn_1 = lambda x: np.cos(u_interp(x))
                for i in range(len(u) - 1):
                  fn_2 = lambda y: -np.exp(
                      integral(fn_1, problem.t_span[-1], y))
                  a[i] = np.exp(
                      -1*integral(fn_1, problem.t_span[-1], t[i]))*integral(
                          fn_2, problem.t_span[-1], t[i])
                return a

          elif problem.out_functional == 'J=u_N':

            def getK(dt_n, u=None, v0=0):
              k = np.zeros_like(dt_n)
              k[-1] = 1
              return np.concatenate((k, v0), axis=None)

            if problem.ode == 'du/dt=sin(u)':

              def exactAdj(t, u):
                u_interp = lambda x: np.interp(x, t, u)
                a = np.zeros_like(u)
                for i in range(len(u) - 1):
                  fn_1 = lambda y: np.cos(u_interp(y))
                  a[i] = -np.exp(-integral(fn_1, problem.t_span[-1], t[i]))*(
                      1 - np.heaviside(problem.t_span[-1] - t[i], 1)*2)
                return a
        else:
          if problem.out_functional == 'J=int(u^2)':

            def getK(dt_n, u, v0=0):
              k = 2*u[:-1]*dt_n
              return np.concatenate((k, v0), axis=None)

            if problem.ode == 'du/dt=sin(u)':

              def exactAdj(t, u):
                a = np.zeros_like(u)
                u_interp = lambda x: np.interp(x, t, u)
                fn_1 = lambda y: np.cos(u_interp(y))
                fn_2 = lambda z: np.exp(integral(fn_1, problem.t_span[-1], z)
                                       )*u_interp(z)* -2
                for i in range(len(u) - 1):
                  a[i] = np.exp(-integral(fn_1, problem.t_span[-1], t[i])
                               )*integral(fn_2, problem.t_span[-1], t[i])
                return a
    else:
      if problem.ode == 'ResNet':

        def fwdUpdate(dt_n, u, n, net, params):
          dt_in = jnp.array([dt_n[n - 1]])
          u_in = jnp.array([u[n - 1]])
          t_in = jnp.sum(dt_n[:n])
          return jnp.squeeze(net.apply({'params': params}, u_in, t_in, dt_in))

        def getF(dt_n, u, net, params):
          f_vec = vmap(
              net.apply, in_axes=(None, 0, 0))({
                  'params': params
              }, u[:-1], dt_n)
          return np.concatenate((u[0], f_vec), axis=None)

        def getJF(dt_n, u, net, params):
          times = jnp.concatenate((jnp.array([0]), jnp.cumsum(dt_n)), None)
          netOut = lambda u, t, dt: net.apply({'params': params}, u, t, dt)[0]
          jf_diag = vmap(
              grad(netOut), in_axes=(0, 0, 0))(u[:-1], times[:-1], dt_n)
          return np.diag(jf_diag, -1)

        def exactFwd(t, u0):
          return None

        def exactAdj(t, u):
          return None

      if problem.linear_out_functional:
        if problem.out_functional == 'J=int(u)':

          def getK(dt_n, u=None, v0=0):
            k = dt_n
            return np.concatenate((k, v0), axis=None)

        elif problem.out_functional == 'J=u_N':

          def getK(dt_n, u=None, v0=0):
            k = np.zeros_like(dt_n)
            k[-1] = 1
            return np.concatenate((k, v0), axis=None)
      else:
        if problem.out_functional == 'J=int(u^2)':

          def getK(dt_n, u, v0=0):
            k = 2*u[:-1]*dt_n
            return np.concatenate((k, v0), axis=None)

    return Funs(exactAdj, exactFwd, fwdUpdate, getF, getJF, getK)

  def getAdaptFunctions(self) -> AdaptFuns:
    problem = self.problem

    def refineAll(dt_n):
      n_steps = len(dt_n)*problem.ref_factor
      dt_fine = np.zeros(n_steps)
      for f in range(problem.ref_factor):
        dt_fine[f:n_steps - problem.ref_factor + f +
                1:problem.ref_factor] = dt_n / problem.ref_factor
      return dt_fine, n_steps

    def interpU(dt_fine, dt_n, u):
      # interpolate u onto fine grid
      t_coarse = np.concatenate(([0], np.cumsum(dt_n)), axis=None)
      t_fine = np.concatenate(([0], np.cumsum(dt_fine)), axis=None)
      u_fine = np.interp(t_fine, t_coarse, u)
      return u_fine

    def animate(state):
      plots = os.listdir(state.problem.case)
      frame = cv2.imread(os.path.join(state.problem.case, plots[0]))
      height, width, _ = frame.shape
      video = cv2.VideoWriter(
          state.problem.case + '/' + state.problem.case + '.mp4',
          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 12, (width, height))
      for i, p in enumerate(plots):
        p_path = os.path.join(state.problem.case, p)
        video.write(cv2.imread(p_path))
        if i > 0 and i < len(plots) - 1:
          os.remove(p_path)

      cv2.destroyAllWindows()
      video.release()

    if not problem.is_net:

      def adapt(state: AdaptState, u0, plot: bool = True):
        ff = FunFactory(state.problem)
        funs = ff.getFunctions()
        afuns = ff.getAdaptFunctions()
        del ff
        times = state.times_new
        dt_n = np.diff(times, 1)
        err_steps = np.zeros_like(dt_n)
        u = afuns.forwardSolve(funs, dt_n, u0)
        # sum errors within each step
        # stride = ref_factor, window_len = ref_factor - 1
        v = afuns.adjointSolve(funs, dt_n, u)
        err_steps_n = np.abs(afuns.errorEstimate(funs, dt_n, u, v))[2:]

        rows = (err_steps_n.size -
                (state.problem.ref_factor - 1)) // state.problem.ref_factor + 1
        strides = err_steps_n.strides[0]
        err_steps_n = np.lib.stride_tricks.as_strided(
            err_steps_n,
            shape=(rows, state.problem.ref_factor - 1),
            strides=(state.problem.ref_factor*strides, strides))
        err_steps_n = np.sum(err_steps_n, 1)
        err_steps += err_steps_n
        # adapt
        times_new = np.zeros(len(times) + 1)
        ref_idx = np.argmax(err_steps) + 1
        times_new[0:ref_idx] = times[0:ref_idx]
        times_new[ref_idx + 1:] = times[ref_idx:]
        times_new[ref_idx] = np.mean(times[ref_idx - 1:ref_idx + 1])
        state.iterate(err_steps, times, times_new, u, v)

        if plot:
          if state.it == 0:
            state.bar_ylim = plotIteration(state, funs)
          else:
            plotIteration(state, funs, state.bar_ylim)

        return state

      def adjointSolve(funs: Funs, dt_n, u):
        """Create a Linear System to solve the adjoint equation
          :math:`K^T U + \lambda^T AU = 0`
          
          or rather, its transpose
          :math:`A^T \lambda = -K`

          K is given by the partial derivative of the output functional
          :math:`\\frac{\partial J}{\partial u} = [0, 0, ..., 0, 1]`
        """
        # n_steps = len(dt_n)

        # refine grid for adjoint
        dt_fine, _ = refineAll(dt_n)
        u_fine = interpU(dt_fine, dt_n, u)
        f_jacobian = funs.getJF(dt_fine, u_fine)
        k_vec = funs.getK(dt_fine, u_fine)
        v_vec = np.linalg.solve(f_jacobian.T - np.eye(f_jacobian.shape[0]),
                                -k_vec)
        return v_vec

      def errorEstimate(funs: Funs, dt_n, u, v):
        """Calculate the Adjoint-Weighted Residual as an error estimate
          for each fine time step
        """
        # refine grid for adjoint
        dt_fine, n_steps = refineAll(dt_n)
        u_fine = interpU(dt_fine, dt_n, u)

        res_u = np.zeros_like(u_fine)
        for n in np.arange(n_steps) + 1:
          res_u[n] = u_fine[n] - funs.fwdUpdate(dt_fine, u_fine, n)
        err = res_u*v

        return err

      def forwardSolve(funs: Funs, dt_n, u0=None):
        """Step through time with finite difference
        """
        n_nodes = len(dt_n) + 1

        def solve(u0):
          u_vec = np.zeros(n_nodes)
          for n in range(n_nodes):
            if n != 0:
              u_vec[n] = funs.fwdUpdate(dt_n, u_vec, n)
            else:
              u_vec[0] = u0
          return u_vec

        if u0 is None:
          return solve
        else:
          return solve(u0)

      def plotIteration(state: AdaptState, funs: Funs = None, bar_ylim=None):
        dt_n = np.diff(state.times)
        dt_fine, _ = refineAll(dt_n)
        times_fine = np.concatenate((0, np.cumsum(dt_fine)), axis=None)
        x_plot = np.linspace(state.times[0], state.times[-1], 500)
        exact_fwd = funs.exactFwd(x_plot)
        exact_adj = funs.exactAdj(x_plot, exact_fwd)

        fig, ax1 = plt.subplots()

        bar_x = state.times[0:-1] + dt_n/2
        ax1.bar(
            bar_x,
            state.err_steps,
            dt_n,
            color='darkseagreen',
            label='Error Estimate')
        ax1.set_ylabel('Error Contribution')
        if state.it == 0:
          bar_ylim = ax1.get_ylim()
        else:
          ax1.set_ylim(*state.bar_ylim)
        ax2 = ax1.twinx()

        ax2.plot(
            x_plot,
            funs.exactFwd(x_plot),
            '-',
            color='k',
            label='Exact Primal',
            linewidth=4)
        ax2.plot(
            x_plot,
            exact_adj,
            '-',
            color='saddlebrown',
            label='Exact Adjoint',
            linewidth=4)

        # FD solution
        ax2.plot(
            state.times,
            state.u,
            '-',
            marker='.',
            color='tab:blue',
            label='FD Primal',
            linewidth=1.25)
        ax2.plot(
            times_fine,
            state.v,
            '-',
            marker='.',
            color='tab:orange',
            label='FD Adjoint',
            linewidth=1.25)
        ax2.set_ylabel('Solution')
        ax2.set_xlabel('Time')

        fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

        f_name = state.problem.case + '_{:d}'.format(state.it)
        fig.savefig(state.problem.case + '/' + f_name + '.png')
        plt.close(fig)
        if state.it == 0:
          return bar_ylim

    else:

      def adapt(state: AdaptState, u0, net, params, plot: bool = True):
        ff = FunFactory(state.problem)
        funs = ff.getFunctions()
        afuns = ff.getAdaptFunctions()
        del ff
        times = state.times_new
        dt_n = np.diff(times, 1)
        err_steps = np.zeros_like(dt_n)
        u = afuns.forwardSolve(funs, dt_n, u0, net, params)
        # sum errors within each step
        # stride = ref_factor, window_len = ref_factor - 1
        v = afuns.adjointSolve(funs, dt_n, u, net, params)
        err_steps_n = np.abs(
            afuns.errorEstimate(funs, dt_n, u, v, net, params))[2:]

        rows = (err_steps_n.size -
                (state.problem.ref_factor - 1)) // state.problem.ref_factor + 1
        strides = err_steps_n.strides[0]
        err_steps_n = np.lib.stride_tricks.as_strided(
            err_steps_n,
            shape=(rows, state.problem.ref_factor - 1),
            strides=(state.problem.ref_factor*strides, strides))
        err_steps_n = np.sum(err_steps_n, 1)
        err_steps += err_steps_n
        # adapt
        times_new = np.zeros(len(times) + 1)
        ref_idx = np.argmax(err_steps) + 1
        times_new[0:ref_idx] = times[0:ref_idx]
        times_new[ref_idx + 1:] = times[ref_idx:]
        times_new[ref_idx] = np.mean(times[ref_idx - 1:ref_idx + 1])
        state.iterate(err_steps, times, times_new, u, v)

        if plot:
          if state.it == 0:
            state.bar_ylim = plotIteration(state, funs)
          else:
            plotIteration(state, funs, state.bar_ylim)

        return state

      def adjointSolve(funs: Funs, dt_n, u, net=None, params=None):
        """Create a Linear System to solve the adjoint equation
          :math:`K^T U + \lambda^T AU = 0`
          
          or rather, its transpose
          :math:`A^T \lambda = -K`

          K is given by the partial derivative of the output functional
          :math:`\\frac{\partial J}{\partial u} = [0, 0, ..., 0, 1]`
        """
        # n_steps = len(dt_n)

        # refine grid for adjoint
        dt_fine, _ = refineAll(dt_n)
        u_fine = interpU(dt_fine, jnp.squeeze(dt_n), jnp.squeeze(u))

        def solve(net, params):
          f_jacobian = funs.getJF(dt_fine, u_fine, net, params)
          k_vec = funs.getK(dt_fine, u_fine)
          v_vec = np.linalg.solve(f_jacobian.T - np.eye(f_jacobian.shape[0]),
                                  -k_vec)
          return v_vec

        if net is None:
          return solve
        else:
          return solve(net, params)

      def errorEstimate(funs: Funs, dt_n, u, v, net=None, params=None):
        """Calculate the Adjoint-Weighted Residual as an error estimate
          for each fine time step
        """
        # refine grid for adjoint
        dt_fine, n_steps = refineAll(dt_n)
        u_fine = interpU(dt_fine, dt_n, u)

        def est(net, params):
          res_u = np.zeros_like(u_fine)
          for n in np.arange(n_steps) + 1:
            res_u[n] = u_fine[n] - funs.fwdUpdate(dt_fine, u_fine, n, net,
                                                  params)
          err = res_u*v
          return err

        if net is None:
          return est
        else:
          return est(net, params)

      def forwardSolve(funs: Funs, dt_n, u0=None, net=None, params=None):
        """Step through time with finite difference
        """
        n_nodes = len(dt_n) + 1

        def solve(u0, net, params):
          u_vec = jnp.zeros((n_nodes, 1))
          for n in range(n_nodes):
            if n != 0:
              u_vec = u_vec.at[n].set(
                  funs.fwdUpdate(dt_n, u_vec, n, net, params))
            else:
              u_vec = u_vec.at[0].set(u0)
          return u_vec

        if u0 is None:
          return solve
        elif net is None:
          return partial(solve, u0=u0)
        else:
          return solve(u0, net, params)

      def plotIteration(state: AdaptState, funs: Funs = None, bar_ylim=None):
        dt_n = jnp.diff(state.times)
        dt_fine, _ = refineAll(dt_n)
        times_fine = jnp.concatenate((jnp.array([0]), jnp.cumsum(dt_fine)),
                                     axis=None)

        fig, ax1 = plt.subplots()

        bar_x = state.times[0:-1] + dt_n/2
        ax1.bar(
            bar_x,
            state.err_steps,
            dt_n,
            color='darkseagreen',
            label='Error Estimate')
        ax1.set_ylabel('Error Contribution')
        if state.it == 0:
          bar_ylim = ax1.get_ylim()
        else:
          ax1.set_ylim(*state.bar_ylim)
        ax2 = ax1.twinx()

        # FD solution
        ax2.plot(
            state.times,
            state.u,
            '-',
            marker='.',
            color='tab:blue',
            label='FD Primal',
            linewidth=1.25)
        ax2.plot(
            times_fine,
            state.v,
            '-',
            marker='.',
            color='tab:orange',
            label='FD Adjoint',
            linewidth=1.25)
        ax2.set_ylabel('Solution')
        ax2.set_xlabel('Time')

        fig.legend(bbox_to_anchor=(0.65, 1), bbox_transform=ax2.transAxes)

        f_name = state.problem.case + '_{:d}'.format(state.it)
        fig.savefig(state.problem.case + '/' + f_name + '.png')
        plt.close(fig)
        if state.it == 0:
          return bar_ylim

    return AdaptFuns(adapt, adjointSolve, animate, errorEstimate, forwardSolve,
                     interpU, plotIteration, refineAll)