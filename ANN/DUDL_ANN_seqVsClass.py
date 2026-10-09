# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>ANNs<h1>|
# |<h2>Lecture:</h2>|<h1><b>Defining models using sequential vs. class<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# NOTE: copied from notebook DUDL_ANN_classifyQwerties.ipynb


# %%
# import libraries
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import torch

# NEW!
import torch.nn.functional as F
from torch import nn

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


# %%
# create data

nPerClust = 100
blur = 1

A = [1, 1]
B = [5, 1]

# generate data
a = [A[0] + np.random.randn(nPerClust) * blur, A[1] + np.random.randn(nPerClust) * blur]
b = [B[0] + np.random.randn(nPerClust) * blur, B[1] + np.random.randn(nPerClust) * blur]

# true labels
labels_np = np.vstack((np.zeros((nPerClust, 1)), np.ones((nPerClust, 1))))

# concatanate into a matrix
data_np = np.hstack((a, b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5, 5))
plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], "bs")
plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], "ko")
plt.title("The qwerties!")
plt.xlabel("qwerty dimension 1")
plt.ylabel("qwerty dimension 2")
plt.show()


# %%
# inspect types
print(type(data_np))
print(np.shape(data_np))
print(" ")

print(type(data))
print(np.shape(data))


# %%
# # build the model
# ANNclassify = nn.Sequential(
#     nn.Linear(2,1),   # input layer
#     nn.ReLU(),        # activation unit
#     nn.Linear(1,1),   # output unit
#     nn.Sigmoid(),     # final activation unit (here for conceptual reasons; in practice, better to use BCEWithLogitsLoss)
#       )


# %%
### define the class


class theClass4ANN(nn.Module):
    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(2, 1)

        ### output layer
        self.output = nn.Linear(1, 1)

    # forward pass
    def forward(self, x):

        # pass through the input layer
        x = self.input(x)

        # apply relu
        x = F.relu(x)

        # output layer
        x = self.output(x)
        x = torch.sigmoid(x)

        return x


### create an instance of the class
ANNclassify = theClass4ANN()


# %%
# other model features

learningRate = 0.01

# loss function
lossfun = nn.BCELoss()
# Note: You'll learn in the "Metaparameters" section that it's better to use BCEWithLogitsLoss, but this is OK for now.

# optimizer
optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=learningRate)


# %%
# train the model
numepochs = 1000
losses = torch.zeros(numepochs)

for epochi in range(numepochs):
    # forward pass
    yHat = ANNclassify(data)

    # compute loss
    loss = lossfun(yHat, labels)
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %%
# show the losses

plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# %%
# compute the predictions

# manually compute losses
# final forward pass
predictions = ANNclassify(data)

predlabels = predictions > 0.5

# find errors
misclassified = np.where(predlabels != labels)[0]

# total accuracy
totalacc = 100 - 100 * len(misclassified) / (2 * nPerClust)

print(f"Final accuracy: {totalacc:g}%")


# %%
# plot the labeled data
fig = plt.figure(figsize=(5, 5))
plt.plot(
    data[misclassified, 0],
    data[misclassified, 1],
    "rx",
    markersize=12,
    markeredgewidth=3,
)
plt.plot(data[np.where(~predlabels)[0], 0], data[np.where(~predlabels)[0], 1], "bs")
plt.plot(data[np.where(predlabels)[0], 0], data[np.where(predlabels)[0], 1], "ko")

plt.legend(["Misclassified", "blue", "black"], bbox_to_anchor=(1, 1))
plt.title(f"{totalacc}% correct")
plt.show()


# %%
