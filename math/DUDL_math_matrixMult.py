# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Math, numpy, pytorch<h1>|
# |<h2>Lecture:</h2>|<h1><b>Matrix multiplication<b></h1>|
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
# create some random matrices
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(3, 7)

# try some multiplications...
print(np.round(A @ B, 2)), print(" ")
# print(np.round( A@C   ,2)), print(' ')
# print(np.round( B@C   ,2)), print(' ')
print(np.round(C.T @ A, 2))


# # Using pytorch

# %%
# create some random matrices
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C1 = np.random.randn(4, 7)
C2 = torch.tensor(C1, dtype=torch.float)

# try some multiplications...
# print(np.round( A@B   ,2)), print(' ')
# print(np.round( A@B.T ,2)), print(' ')
print(np.round(A @ C1, 2)), print(" ")
print(np.round(A @ C2, 2))
