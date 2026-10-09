# This file was generated from a Jupyter notebook.
# ruff: noqa: B018

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>ANNs<h1>|
# |<h2>Lecture:</h2>|<h1><b>ANN for regression<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# import libraries
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import torch
from torch import nn

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


# %%
# create data

N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1) / 2

# and plot
plt.plot(x, y, "s")
plt.show()


# %%
# build model
ANNreg = nn.Sequential(
    nn.Linear(1, 1),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(1, 1),  # output layer
)

ANNreg


# %%
# learning rate
learningRate = 0.05

# loss function
lossfun = nn.MSELoss()

# optimizer (the flavor of gradient descent to implement)
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learningRate)


# %%
# train the model
numepochs = 500
losses = torch.zeros(numepochs)


## Train the model!
for epochi in range(numepochs):
    # forward pass
    yHat = ANNreg(x)

    # compute loss
    loss = lossfun(yHat, y)
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %%
# show the losses

# manually compute losses
# final forward pass
predictions = ANNreg(x)

# final loss (MSE)
testloss = (predictions - y).pow(2).mean()

plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
plt.plot(numepochs, testloss.detach(), "ro")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Final loss = {testloss.item():g}")
plt.show()


# %%
testloss.item()


# %%
# plot the data
plt.plot(x, y, "bo", label="Real data")
plt.plot(x, predictions.detach(), "rs", label="Predictions")
plt.title(f"prediction-data r={np.corrcoef(y.T, predictions.detach().T)[0, 1]:.2f}")
plt.legend()
plt.show()


# %%
# # Additional explorations

# %%
# 1) How much data is "enough"? Try different values of N and see how low the loss gets.
#    Do you still get low loss ("low" is subjective, but let's say loss<.25) with N=10? N=5?
#
# 2) Does your conclusion above depend on the amount of noise in the data? Try changing the noise level
#    by changing the division ("/2") when creating y as x+randn.
#
# 3) Notice that the model doesn't always work well. Put the original code (that is, N=30 and /2 noise)
#    into a function or a for-loop and repeat the training 100 times (each time using a fresh model instance).
#    Then count the number of times the model had a loss>.25.
