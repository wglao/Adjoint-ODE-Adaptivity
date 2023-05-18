import argparse
from typing import Any

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

from animate import animate
import cv2
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrand
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
from models import SingleNeuronLayers as SNL

"""
Karin Kraft Adaptive FEM for ODEs applied to NNs

Begin with layers of single neurons
"""

seed = int(args.seed)
rng = jrand.PRNGKey(seed)
net = SNL(2)
params = net.init()


