import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from typing import NamedTuple, Iterable
"""Classes that handle necessary function creation"""


class Problem(NamedTuple):
  is_net: bool
  linear_ode: bool
  linear_out_functional: bool
  ode: str
  out_functional: str
  t_span: np.ndarray


class Funs(NamedTuple):
  exactAdj: callable
  exactFwd: callable
  fwdUpdate: callable
  getF: callable
  getJF: callable
  getK: callable


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

        def fwdUpdate(u, dt_n, n):
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

        def fwdUpdate(u, dt_n, n):
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

            def exactAdj(t):
              return np.exp(-t)*np.exp(problem.t_span[-1]) - 1

          elif problem.out_functional == 'J=u_N':

            def getK(dt_n, u=None, v0=0):
              k = np.zeros_like(dt_n)
              k[-1] = 1
              return np.concatenate((k, v0), axis=None)

            def exactAdj(t):
              return -(np.sign(t - problem.t_span[-1])*np.exp(-t)*
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
                  fn_2 = lambda y: -np.exp(integral(fn_1, problem.t_span[-1], y)
                                          )
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
        # du/dt=W_2*ReLU(W_1*u+b)
        def fwdUpdate(u, dt_n, n, net: ResNetBlock, params: dict):
          return net.apply({'params': params}, u[n - 1], dt_n[n - 1])

        def getF(u, dt_n, net: ResNetBlock, params: dict):
          f_vec = jax.vmap(net.apply, in_axes=(None, 0, 0))({
              'params': params
          }, u[:-1], dt_n)
          return np.concatenate((u[0], f_vec), axis=None)

        def getJF(u, dt_n, params: dict):
          w1 = params['dense1']
          w2 = params['dense2']
          jf_diag = 1 + dt_n*np.dot(
              w1, np.dot(w2, np.where(np.dot(w1, u[:-1]) > 0, [1, 0])))
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


# Define NNs


class ResNetBlock(nn.Module):

  def __init__(self, szs: Iterable[int]) -> None:
    super().__init__()
    self.hidden = szs[0]
    self.outsz = szs[1]

  def setup(self):
    self.dense1 = nn.Dense(self.hidden)
    self.dense2 = nn.Dense(self.outsz)

  def __call__(self, u, dt):
    out = self.dense1(u)
    out = nn.relu(out)
    out = self.dense2(out)
    return u + dt*out