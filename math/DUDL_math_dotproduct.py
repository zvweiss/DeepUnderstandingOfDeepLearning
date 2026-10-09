# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Math, numpy, pytorch<h1>|
# |<h2>Lecture:</h2>|<h1><b>OMG it's the dot product!<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# import libraries
import numpy as np
import torch

# # Using numpy

# %%
# create a vector
nv1 = np.array([1, 2, 3, 4])
nv2 = np.array([0, 1, 0, -1])

# dot product via function
print(np.dot(nv1, nv2))

# dot product via computation
print(np.sum(nv1 * nv2))


# # Using pytorch

# %%
# create a vector
tv1 = torch.tensor([1, 2, 3, 4])
tv2 = torch.tensor([0, 1, 0, -1])

# dot product via function
print(torch.dot(tv1, tv2))

# dot product via computation
print(torch.sum(tv1 * tv2))
