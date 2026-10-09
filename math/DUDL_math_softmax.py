# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Math, numpy, pytorch<h1>|
# |<h2>Lecture:</h2>|<h1><b>Softmax<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# import libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# %%
# "manually" in numpy

# the list of numbers
z = [1, 2, 3]

# compute the softmax result
num = np.exp(z)
den = np.sum(np.exp(z))
sigma = num / den

print(sigma)
print(np.sum(sigma))


# %%
# repeat with some random integers
z = np.random.randint(-5, high=15, size=25)
print(z)

# compute the softmax result
num = np.exp(z)
den = np.sum(num)
sigma = num / den

# compare
plt.plot(z, sigma, "ko")
plt.xlabel("Original number (z)")
plt.ylabel(r"Softmaxified $\sigma$")
plt.yscale("log")
plt.title(rf"$\sum\sigma$ = {np.sum(sigma):g}")
plt.show()


# # Using pytorch

# %%
# slightly more involved using torch.nn

# create an instance of the softmax activation class
softfun = nn.Softmax(dim=0)

# then apply the data to that function
sigmaT = softfun(torch.Tensor(z))

# now we get the results
print(sigmaT)


# %%
# show that they are the same
plt.plot(sigma, sigmaT, "ko")
plt.xlabel('"Manual" softmax')
plt.ylabel("Pytorch nn.Softmax")
plt.title(f"The two methods correlate at r={np.corrcoef(sigma, sigmaT)[0, 1]}")
plt.show()

print("pause")
