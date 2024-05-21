import ray
import ray.rllib.agents.ppo as ppo
import tensorflow as tf
import numpy as np
import scipy

print(f"Ray version: {ray.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
