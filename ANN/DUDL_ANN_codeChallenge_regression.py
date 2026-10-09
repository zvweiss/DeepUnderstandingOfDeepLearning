# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>ANNs<h1>|
# |<h2>Lecture:</h2>|<h1><b>CodeChallenge: manipulate regression slopes<b></h1>|
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


# # A function that creates and trains the model


# %%
def buildAndTrainTheModel(x, y):

    # build the model
    ANNreg = nn.Sequential(
        nn.Linear(1, 1),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(1, 1),  # output layer
    )

    # loss and optimizer functions
    lossfun = nn.MSELoss()
    optimizer = torch.optim.SGD(ANNreg.parameters(), lr=0.05)

    #### train the model
    numepochs = 500
    losses = torch.zeros(numepochs)

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
    # end training loop

    ### compute model predictions
    predictions = ANNreg(x)

    # output:
    return predictions, losses


# # A function that creates the data


# %%
def createTheData(m):
    N = 50
    x = torch.randn(N, 1)
    y = m * x + torch.randn(N, 1) / 2
    return x, y


# # Test it once

# %%
# create a dataset
x, y = createTheData(0.8)

# run the model
yHat, losses = buildAndTrainTheModel(x, y)
yHat = yHat.detach()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
ax[0].set_xlabel("Epoch")
ax[0].set_title("Loss")

ax[1].plot(x, y, "bo", label="Real data")
ax[1].plot(x, yHat, "rs", label="Predictions")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title(f"prediction-data corr = {np.corrcoef(y.T, yHat.T)[0, 1]:.2f}")
ax[1].legend()

plt.show()


# # Now for the experiment!

# %%
# (takes 3 mins with 21 slopes and 50 exps)

# the slopes to simulate
slopes = np.linspace(-2, 2, 21)

numExps = 50

# initialize output matrix
results = np.zeros((len(slopes), numExps, 2))

for slopei in range(len(slopes)):
    for N in range(numExps):
        # create a dataset and run the model
        x, y = createTheData(slopes[slopei])
        yHat, losses = buildAndTrainTheModel(x, y)
        yHat = yHat.detach()

        # store the final loss and performance
        results[slopei, N, 0] = losses[-1]
        results[slopei, N, 1] = np.corrcoef(y.T, yHat.T)[0, 1]


# correlation can be 0 if the model didn't do well. Set nan's->0
results[np.isnan(results)] = 0


# %%
# plot the results!

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(
    slopes, np.mean(results[:, :, 0], axis=1), "ko-", markerfacecolor="w", markersize=10
)
ax[0].set_xlabel("Slope")
ax[0].set_title("Loss")

ax[1].plot(
    slopes, np.mean(results[:, :, 1], axis=1), "ms-", markerfacecolor="w", markersize=10
)
ax[1].set_xlabel("Slope")
ax[1].set_ylabel("Real-predicted correlation")
ax[1].set_title("Model performance")

plt.show()


# %%
# extra code to visualize data with different slopes

m = 2

x, y = createTheData(m)

plt.title("Slope = " + str(m))
plt.plot(x, y, "o")
plt.ylim([-4, 4])
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# %%
