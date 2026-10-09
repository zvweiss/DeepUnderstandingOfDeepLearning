# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Math, numpy, pytorch<h1>|
# |<h2>Lecture:</h2>|<h1><b>Vector and matrix transpose<b></h1>|
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
nv = np.array([[1, 2, 3, 4]])
print(nv), print(" ")

# transpose it
print(nv.T), print(" ")

# transpose the transpose!
nvT = nv.T
print(nvT.T)


# %%
# repeat for a matrix
nM = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(nM), print(" ")

# transpose it
print(nM.T), print(" ")

# transpose the transpose!
nMT = nM.T
print(nMT.T)


# # Using pytorch

# %%
# create a vector
tv = torch.tensor([[1, 2, 3, 4]])
print(tv), print(" ")

# transpose it
print(tv.T), print(" ")

# transpose the transpose!
tvT = tv.T
print(tvT.T)


# %%
# repeat for a matrix
tM = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(tM), print(" ")

# transpose it
print(tM.T), print(" ")

# transpose the transpose!
tMT = tM.T
print(tMT.T)


# %%
# %%
# examine data types
print(f"Variable nv is of type {type(nv)}")
print(f"Variable nM is of type {type(nM)}")
print(f"Variable tv is of type {type(tv)}")
print(f"Variable tM is of type {type(tM)}")
