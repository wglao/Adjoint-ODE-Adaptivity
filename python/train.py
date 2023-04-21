from typing import Iterable, NamedTuple

import flax.linen as nn
from jax import vmap, grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from factory import FunFactory, Funs, Problem

ode = lambda t,y: -t*np.sin(y**2)/y

def 
