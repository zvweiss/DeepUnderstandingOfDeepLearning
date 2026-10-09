# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>ANNs<h1>|
# |<h2>Lecture:</h2>|<h1><b>Depth vs. breadth: number of parameters<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# import libraries
import numpy as np
from torch import nn

# %%
# build two models

widenet = nn.Sequential(
    nn.Linear(2, 4),  # hidden layer
    nn.Linear(4, 3),  # output layer
)


deepnet = nn.Sequential(
    nn.Linear(2, 2),  # hidden layer
    nn.Linear(2, 2),  # hidden layer
    nn.Linear(2, 3),  # output layer
)

# print them out to have a look
print(widenet)
print(" ")
print(deepnet)


# %%
# `widenet.` was used in the notebook as an autocomplete demonstration.


# # Peeking inside the network

# %%
# check out the parameters
for p in deepnet.named_parameters():
    print(p)
    print(" ")


# %%
# count the number of nodes ( = the number of biases)

# named_parameters() is an iterable that returns the tuple (name,numbers)
numNodesInWide = 0
for p in widenet.named_parameters():
    if "bias" in p[0]:
        numNodesInWide += len(p[1])

numNodesInDeep = 0
for paramName, paramVect in deepnet.named_parameters():
    if "bias" in paramName:
        numNodesInDeep += len(paramVect)


print(f"There are {numNodesInWide} nodes in the wide network.")
print(f"There are {numNodesInDeep} nodes in the deep network.")


# %%
# just the parameters
for p in widenet.parameters():
    print(p)
    print(" ")


# %%
# now count the total number of trainable parameters
nparams = 0
for p in widenet.parameters():
    if p.requires_grad:
        print(f"This piece has {p.numel()} parameters")
        nparams += p.numel()

print(f"\n\nTotal of {nparams} parameters")


# %%
# btw, can also use list comprehension

nparams = np.sum([p.numel() for p in widenet.parameters() if p.requires_grad])
print(f"Widenet has {nparams} parameters")

nparams = np.sum([p.numel() for p in deepnet.parameters() if p.requires_grad])
print(f"Deepnet has {nparams} parameters")


# %%
# %%
# A nice simple way to print out the model info.
from torchsummary import summary

summary(widenet, (1, 2))


### NOTE ABOUT THE CODE IN THIS CELL:
# torchsummary is being replaced by torchinfo.
# If you are importing these libraries on your own (via pip), then see the following website:
#        https://pypi.org/project/torch-summary/
# However, torchsummary will continue to be supported, so if the code in this cell works (meaning torchsummary is already installed),
# then you don't need to do anything!


# %%
