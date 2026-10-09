# This file was generated from a Jupyter notebook.
# ruff: noqa: B018

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Math, numpy, pytorch<h1>|
# |<h2>Lecture:</h2>|<h1><b>Min/max and argmin/max<b></h1>|
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
v = np.array([1, 40, 2, -3])

# find and report the maximum and minimum values
minval = np.min(v)
maxval = np.max(v)

print(f"Min,max: {minval:g},{maxval:g}")


# now for the argmin/max
minidx = np.argmin(v)
maxidx = np.argmax(v)

print(f"Min,max indices: {minidx:g},{maxidx:g}"), print(" ")

# confirm
print(f"Min val is {v[minidx]}, max val is {v[maxidx]}")


# %%
# repeat with matrix
M = np.array([[0, 1, 10], [20, 8, 5]])
print(M), print(" ")

# various minima in this matrix!
minvals1 = np.min(M)  # minimum from ENTIRE matrix
minvals2 = np.min(M, axis=0)  # minimum of each column (across rows)
minvals3 = np.min(M, axis=1)  # minimum of each row (across columns)

# print them out
print(minvals1)
print(minvals2)
print(minvals3)


# %%
# various minima in this matrix!
minidx1 = np.argmin(M)  # minimum from ENTIRE matrix
minidx2 = np.argmin(M, axis=0)  # minimum of each column (across rows)
minidx3 = np.argmin(M, axis=1)  # minimum of each row (across columns)

# print them out
print(M), print(" ")  # reminder
print(minidx1)
print(minidx2)
print(minidx3)


# # Now using pytorch

# %%
# create a vector
v = torch.tensor([1, 40, 2, -3])

# find and report the maximum and minimum values
minval = torch.min(v)
maxval = torch.max(v)

print(f"Min,max: {minval:g},{maxval:g}")


# now for the argmin/max
minidx = torch.argmin(v)
maxidx = torch.argmax(v)

print(f"Min,max indices: {minidx:g},{maxidx:g}"), print(" ")

# confirm
print(f"Min val is {v[minidx]}, max val is {v[maxidx]}")


# %%
# repeat with matrix
M = torch.tensor([[0, 1, 10], [20, 8, 5]])
print(M), print(" ")

# various minima in this matrix!
min1 = torch.min(M)  # minimum from ENTIRE matrix
min2 = torch.min(M, axis=0)  # minimum of each column (across rows)
min3 = torch.min(M, axis=1)  # minimum of each row (across columns)

# print them out
print(min1), print(" ")
print(min2), print(" ")
print(min2.values)
print(min2.indices)


# %%
min2.count
