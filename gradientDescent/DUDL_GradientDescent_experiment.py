# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Gradient descent<h1>|
# |<h2>Lecture:</h2>|<h1><b>Parametric experiments on g.d.<b></h1>|
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


# # Running experiments to understand gradient descent

# %%
# the function
x = np.linspace(-2 * np.pi, 2 * np.pi, 401)
fx = np.sin(x) * np.exp(-(x**2) * 0.05)

# and its derivative
df = np.cos(x) * np.exp(-(x**2) * 0.05) + np.sin(x) * (-0.1 * x) * np.exp(
    -(x**2) * 0.05
)

# quick plot for inspection
plt.plot(x, fx, x, df)
plt.legend(["f(x)", "df"])


# %%
# function (note: over-writing variable names!)
def fx(x):
    return np.sin(x) * np.exp(-(x**2) * 0.05)


# derivative function
def deriv(x):
    return np.cos(x) * np.exp(-(x**2) * 0.05) - np.sin(x) * 0.1 * x * np.exp(
        -(x**2) * 0.05
    )


# %%
# random starting point
localmin = np.random.choice(x, 1)  # np.array([6])#

# learning parameters
learning_rate = 0.01
training_epochs = 1000

# run through training
for i in range(training_epochs):
    grad = deriv(localmin)
    localmin = localmin - learning_rate * grad


# plot the results
plt.plot(x, fx(x), x, deriv(x), "--")
plt.plot(localmin, deriv(localmin), "ro")
plt.plot(localmin, fx(localmin), "ro")

plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["f(x)", "df", "f(x) min"])
plt.title(f"Empirical local minimum: {localmin[0]}")
plt.show()


# # Run parametric experiments

# %%
# Experiment 1: systematically varying the starting locations

startlocs = np.linspace(-5, 5, 50)
finalres = np.zeros(len(startlocs))

# loop over starting points
for idx, localmin in enumerate(startlocs):
    # run through training
    for i in range(training_epochs):
        grad = deriv(localmin)
        localmin = localmin - learning_rate * grad

    # store the final guess
    finalres[idx] = localmin


# plot the results
plt.plot(startlocs, finalres, "s-")
plt.xlabel("Starting guess")
plt.ylabel("Final guess")
plt.show()


# %%
# Experiment 2: systematically varying the learning rate

learningrates = np.linspace(1e-10, 1e-1, 50)
finalres = np.zeros(len(learningrates))

# loop over learning rates
for idx, learningRate in enumerate(learningrates):
    # force starting guess to 0
    localmin = 0

    # run through training
    for i in range(training_epochs):
        grad = deriv(localmin)
        localmin = localmin - learningRate * grad

    # store the final guess
    finalres[idx] = localmin


plt.plot(learningrates, finalres, "s-")
plt.xlabel("Learning rate")
plt.ylabel("Final guess")
plt.show()


# %%
# Experiment 3: interaction between learning rate and training epochs

# setup parameters
learningrates = np.linspace(1e-10, 1e-1, 50)
training_epochs = np.round(np.linspace(10, 500, 40))

# initialize matrix to store results
finalres = np.zeros((len(learningrates), len(training_epochs)))


# loop over learning rates
for Lidx, learningRate in enumerate(learningrates):
    # loop over training epochs
    for Eidx, trainEpochs in enumerate(training_epochs):
        # run through training (again fixing starting location)
        localmin = 0
        for i in range(int(trainEpochs)):
            grad = deriv(localmin)
            localmin = localmin - learningRate * grad

        # store the final guess
        finalres[Lidx, Eidx] = localmin


# %%
# plot the results

fig, ax = plt.subplots(figsize=(7, 5))

plt.imshow(
    finalres.T,
    extent=[
        learningrates[0],
        learningrates[-1],
        training_epochs[0],
        training_epochs[-1],
    ],
    aspect="auto",
    origin="lower",
    vmin=-1.45,
    vmax=-1.2,
)

plt.xlabel("Learning rate")
plt.ylabel("Training epochs")
plt.title("Final guess")
plt.colorbar()
plt.show()

# another visualization
plt.plot(learningrates, finalres)
plt.xlabel("Learning rates")
plt.ylabel("Final function estimate")
plt.title("Each line is a training epochs N")
plt.show()


# %%
# # Additional explorations

# %%
# 1) In experiment 3, set the starting location to be 1.6. Re-run the experiment and the image. You'll need to re-adjust
#    the figure color limits; check the line plots at the top of the code to determine a useful color range. Does the new
#    starting value change your conclusions about the interaction between learning rate and training epochs?
#
# 2) In the same experiment, now change the starting location to be random (use code: np.random.choice(x,1)). How do these
#    results look? Are you surprised? Are the results of this experiment still interpretable and what does this tell you
#    about running experiments in DL?
#
