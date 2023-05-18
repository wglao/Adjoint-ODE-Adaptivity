import sys
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap
from jax.lax import while_loop, fori_loop, scan
from typing import Iterable
"""Python Implementation of (Dis)Continuous Galerkin"""


def gamma(z):
  return jnp.exp(jsp.special.gammaln(z))


class BaseGalerkin1D():
  """Holds structures shared across all Galerkin Methods
  (Basis for test/trial functions, element connectivity, etc.)
  """
  n: int = 1
  k: int = 2
  domain: Iterable[jnp.float_] = jnp.array([0., 1.])
  n_gq: int = 2
  node_tol: int = 1e-10
  n_fp = 1
  n_faces = 2

  def jacobiGQ(self, a, b, n):
    if n == 0:
      return jnp.array([-(a - b) / (a+b+2)]), jnp.array([2])

    h1 = 2*jnp.arange(n + 1) + a + b
    j_mat = jnp.diag(-0.5*(a**2 - b**2) / (h1+2) / h1) + jnp.diag(
        2 / (h1[:-1] + 2)*jnp.sqrt(
            jnp.arange(1, n + 1)*(jnp.arange(1, n + 1) + a + b)*
            (jnp.arange(1, n + 1) + a)*(jnp.arange(1, n + 1) + b) /
            (h1[:-1] + 1) / (h1[:-1] + 3)), 1)
    if a + b < 10*jnp.sqrt(sys.float_info.epsilon):
      j_mat = j_mat.at[0, 0].set(0)

    j_mat = j_mat + j_mat.T

    # quadrature via eigenvalues
    d, v = jnp.linalg.eig(j_mat)
    w = jnp.square(v[0, :].T)*2**(a + b + 1) / (
        a+b+1)*gamma(a + 1)*gamma(b + 1) / gamma(a + b + 1)
    return d, w

  def jacobiGL(self, a, b, n):
    if n == 1:
      return jnp.array([-1, 1])
    x_int, w = self.jacobiGQ(a + 1, b + 1, n - 2)
    x = jnp.array([-1, *x_int, 1]).T
    return x

  def jacobiP(self, x, a, b, n):
    xp = x
    dims = xp.shape
    if dims[1] == 1:
      xp = xp.T

    pl = jnp.zeros((n + 1, len(xp)))

    g0 = 2**(a + b + 1) / (a+b+1)*gamma(a + 1)*gamma(b + 1) / (gamma(a + b + 1))
    pl = pl.at[0, :].set(1 / jnp.sqrt(g0))

    if n == 0:
      return pl.T

    g1 = (a+1)*(b+1) / (a+b+3)*g0
    pl = pl.at[1, :].set(((a+b+2)*xp / 2 + (a-b) / 2) / jnp.sqrt(g1))

    if n == 1:
      return pl[n, :].T

    # recurrence
    a_old = 2 / (2+a+b)*jnp.sqrt((a+1)*(b+1) / (a+b+3))

    def scanBody(carry, x):
      pl, a_old = carry

      h1 = 2*x + a + b
      a_new = 2 / (h1+2)*jnp.sqrt(
          (x+1)*(x+1+a+b)*(x+1+a)*(x+1+b) / (h1+1) / (h1+3))
      b_new = -(a**2 - b**2) / h1 / (h1+2)
      pl = pl.at[x - 1, :].set(1 / a_new*(-a_old*pl[x - 1, :] +
                                          (xp-b_new)*pl[x, :]))
      a_old = a_new
      return (pl, a_old), None

    (pl, _), _ = scan(scanBody, pl, jnp.arange(1, n))

    return pl[n, :].T

  def vandermonde1D(self, n, r):
    return vmap(
        self.jacobiP,
        in_axes=(None, None, None, 0),
        out_axes=(None, None, None, 1))(r.ravel(), 0, 0, jnp.arange(n + 1))

  def gradJacobiP(self, r, a, b, n):
    if n == 0:
      return 0.
    return jnp.sqrt(n*(n+a+b+1))*self.jacobiP(r.ravel(), a + 1, b + 1, n - 1)

  def gradVandermonde1D(self, n, r):
    return vmap(
        self.gradJacobiP,
        in_axes=(None, None, None, 0),
        out_axes=(None, None, None, 1))(r.ravel(), 0, 0, jnp.arange(n + 1))

  def dMatrix1D(self, n, r, v):
    v_r = self.gradVandermonde1D(n, r)
    return jnp.linalg.solve(v.T, v_r.T).T

  def lift1D(self, n_p, n_faces, n_fp, v):
    e_mat = jnp.zeros((n_p, n_faces*n_fp))
    e_mat = e_mat.at[[0, n_p - 1], [0, 1]] = 1
    return v @ (v.T @ e_mat)

  def geometricFactors1D(self, x, d_r):
    x_r = d_r @ x
    r_x = 1 / x_r
    return x_r, r_x

  def normals1D(self):
    return jnp.stack(-jnp.ones((self.k,)), jnp.ones((self.k,)))

  def connect1D(self, e_to_v):
    total_faces = self.n_faces*self.k
    n_v = self.k + 1

    # local face to vertex
    v_n = jnp.arange(2)

    # global face to node
    f_to_v = jnp.zeros((total_faces, n_v))
    f = jnp.arange(total_faces)
    f_to_v = f_to_v.at[f, (f+1) // 2].set(1)
    f = jnp.arange

    # global face to face
    f_to_f = f_to_v @ f_to_v.T - jnp.eye(total_faces)
    f_1, f_2 = jnp.argwhere(f_to_f == 1).T

    # convert global face to element and face
    gf_1, gf_2 = jnp.argwhere(f_to_f == 1).T
    e_1, f_1 = jnp.divmod(gf_1, self.n_faces)
    e_2, f_2 = jnp.divmod(gf_2, self.n_faces)

    # rearrange into full connectivity matrices
    e_to_e = jnp.reshape(jnp.arange(self.k), (self.k, 1)) @ jnp.ones(
        (1, self.n_faces))
    e_to_f = jnp.ones(
        (self.k, 1)) @ jnp.reshape(jnp.arange(self.n_faces), (1, self.n_faces))

    e_to_e = e_to_e.at[e_1, f_1].set(e_2)
    e_to_f = e_to_f.at[e_1, f_1].set(f_2)
    return e_to_e, e_to_f

  def buildMaps1D(self):
    node_ids = jnp.reshape(jnp.arange(self.k*self.n_p), (self.n_p, self.k))
    v_map_m = vmap(
        vmap(
            lambda f, k: node_ids[self.f_mask[:, f], k],
            in_axes=(0, None),
            out_axes=(1)),
        in_axes=(None, 0),
        out_axes=(1))(jnp.arange(self.n_faces), jnp.arange(self.k))

    def getVMapP(f,k):
      k2 = self.e_to_e[k,f]
      f2 = self.e_to_f[k,f]
      v_id_m = v_map_m[:,f,k]
      v_id_p = v_map_m[:,f2,k2]
      x1 = self.x[v_id_m]
      x2 = self.x[v_id_p]
      d = (x1 - x2)**2
      if d<self.node_tol:
        return v_id_p
      return 0

    v_map_p = vmap(
        vmap(getVMapP, in_axes=(0, None), out_axes=(1)),
        in_axes=(None, 0),
        out_axes=(2))(jnp.arange(self.n_faces), jnp.arange(self.k))

    # boundary nodes
    map_b = jnp.argwhere(v_map_p==v_map_m)
    v_map_b = v_map_m[map_b]

    # inflow and outflow maps
    self.map_i = 0
    self.map_o = self.k*self.n_faces-1
    self.v_map_i = 0
    self.v_map_o = self.k*self.n_p

    return v_map_m, v_map_p, v_map_b, map_b


  def startUp1D(self):
    self.n_p = self.n + 1

    # LGL grid
    self.r = self.jacobiGL(0, 0, self.n)

    # reference element
    self.v = self.vandermonde1D(self.n, self.r)
    self.inv_v = jnp.linalg.inv(self.v)
    self.d_r = self.dMatrix1D(self.n_p, self.r, self.v)

    # surface integral
    self.lift = self.lift1D(self.n_p, self.n_faces)

    # node coords
    v_a = self.e_to_v[:, 0].T
    v_b = self.e_to_v[:, 1].T
    self.x = jnp.ones((self.n + 1, 1)) @ self.v_x[v_a] + 0.5*(self.r + 1) @ (
        self.v_x[v_b] - self.v_x[v_a])

    # geometric factors
    self.r_x, self.j_mat = self.geometricFactors1D(self.x, self.d_r)

    # edge node masks
    self.f_mask = jnp.concatenate(
        (jnp.argwhere(jnp.abs(self.r + 1) < self.node_tol),
         jnp.argwhere(jnp.abs(self.r - 1) < self.node_tol)),
        axis=0).T
    self.f_x = self.x[self.f_mask.ravel(), :]

    # surface normals
    self.n_x = self.normals1D()
    self.f_scale = 1 / self.j_mat[self.f_mask, :]

    # connectivity matrix
    self.e_to_e, self.e_to_f = self.connect1D(self.e_to_v)

    # connectivity maps
    self.v_map_m, self.v_map_p, self.v_map_b, self.map_b = self.buildMaps1D()



  def __init__(self) -> None:
    nv = self.k + 1

    # node coords
    self.v_x = jnp.diff(self.domain)*(jnp.arange(nv) / (nv-1)) + self.domain[0]

    # element to node connectivity
    self.e_to_v = jnp.stack((jnp.arange(self.k), jnp.arange(1, self.k + 1))).T

    self.startUp1D()

    self.r, self.w = self.jacobiGQ(0,0,self.n_gq)
    self.n_r = self.r.shape[0]

    def getP(nn,k,i):
      return self.inv_v.T[i,nn]@self.jacobiP(self.r[k],0,0,nn-1)

    def getPhi(k,i):
      return jnp.sum(vmap(getP, in_axes=(0,None,None))(jnp.arange(self.n_p),k,i))

    self.phi = vmap(
        vmap(getPhi, in_axes=(None, 0), out_axes=(1)),
        in_axes=(0, None))(jnp.arange(self.n_r), jnp.arange(self.n_p))
