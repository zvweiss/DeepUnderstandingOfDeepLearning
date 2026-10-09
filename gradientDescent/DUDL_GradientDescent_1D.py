# This file was generated from a Jupyter notebook.
# ruff: noqa: B018

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Gradient descent<h1>|
# |<h2>Lecture:</h2>|<h1><b>Gradient descent in 1D<b></h1>|
#
# <br>
#
# <h5><b>Teacher:</b> Mike X Cohen, <a href="https://sincxpress.com" target="_blank">sincxpress.com</a></h5>
# <h5><b>Course URL:</b> <a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">udemy.com/course/deeplearning_x/?couponCode=202508</a></h5>
# <i>Using the code without the course may lead to confusion or errors.</i>

# %%
# import all necessary modules
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


# # Gradient descent in 1D


# %%
# function (as a function)
def fx(x):
    return 3 * x**2 - 3 * x + 4


# derivative function
def deriv(x):
    return 6 * x - 3


# %%
# plot the function and its derivative

# define a range for x
x = np.linspace(-2, 2, 2001)

# plotting
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["y", "dy"])
plt.show()


# %%
# %%
# random starting point
localmin = np.random.choice(x, 1)
print(localmin)

# learning parameters
learning_rate = 0.01
training_epochs = 100

# run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad

localmin


# %%
# plot the results

plt.plot(x, fx(x), x, deriv(x))
plt.plot(localmin, deriv(localmin), "ro")
plt.plot(localmin, fx(localmin), "ro")

plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["f(x)", "df", "f(x) min"])
plt.title(f"Empirical local minimum: {localmin[0]}")
plt.show()


# # Store the model parameters and outputs on each iteration

# %%
# random starting point
localmin = np.random.choice(x, 1)

# learning parameters
learning_rate = 0.01
training_epochs = 100

# run through training and store all the results
modelparams = np.zeros((training_epochs, 2))
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad
    modelparams[i, 0] = localmin[0]
    modelparams[i, 1] = grad[0]


# %%
# %%
# plot the gradient over iterations

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for i in range(2):
    ax[i].plot(modelparams[:, i], "o-")
    ax[i].set_xlabel("Iteration")
    ax[i].set_title(f"Final estimated minimum: {localmin[0]:.5f}")

ax[0].set_ylabel("Local minimum")
ax[1].set_ylabel("Derivative")

plt.show()


# %%
# # Additional explorations

# %%
# 1) Most often in DL, the model trains for a set number of iterations, which is what we do here. But there are other ways
#    of defining how long the training lasts. Modify the code so that training ends when the derivative is smaller than
#    some threshold, e.g., 0.1. Make sure your code is robust for negative derivatives.
#
# 2) Does this change to the code produce a more accurate result? What if you change the stopping threshold?
#
# 3) Can you think of any potential problems that might arise when the stopping criterion is based on the derivative
#    instead of a specified number of training epochs?
#
