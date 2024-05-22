import ray
import ray.rllib.agents.ppo as ppo
import tensorflow as tf
import numpy as np
import scipy

print(f"Ray version: {ray.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

print("testing pyglet (for rendering)")
import pyglet
from pyglet.gl import gl_info

# Print OpenGL info
print("OpenGL Info:")
print(f"Vendor: {gl_info.get_vendor()}")
print(f"Renderer: {gl_info.get_renderer()}")
print(f"Version: {gl_info.get_version()}")

try:
    window = pyglet.window.Window()
    @window.event
    def on_draw():
        window.clear()
    pyglet.app.run()
except Exception as e:
    print(f"An error occurred: {e}")
