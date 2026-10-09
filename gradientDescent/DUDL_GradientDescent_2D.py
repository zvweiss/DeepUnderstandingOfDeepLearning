# This file was generated from a Jupyter notebook.

# |<h2>Course:</h2>|<h1><a href="https://udemy.com/course/deeplearning_x/?couponCode=202508" target="_blank">A deep understanding of deep learning</a></h1>|
# |-|:-:|
# |<h2>Section:</h2>|<h1>Gradient descent<h1>|
# |<h2>Lecture:</h2>|<h1><b>Gradient descent in 2D<b></h1>|
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
import sympy as sym  # sympy to compute the partial derivatives

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


# # Gradient descent in 2D


# %%
# the "peaks" function
def peaks(x, y):
    # expand to a 2D mesh
    x, y = np.meshgrid(x, y)

    z = (
        3 * (1 - x) ** 2 * np.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        - 1 / 3 * np.exp(-((x + 1) ** 2) - y**2)
    )
    return z


# %%
# create the landscape
x = np.linspace(-3, 3, 201)
y = np.linspace(-3, 3, 201)

Z = peaks(x, y)

# let's have a look!
plt.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]], vmin=-5, vmax=5, origin="lower")
plt.show()


# %%
# create derivative functions using sympy

sx, sy = sym.symbols("sx,sy")

sZ = (
    3 * (1 - sx) ** 2 * sym.exp(-(sx**2) - (sy + 1) ** 2)
    - 10 * (sx / 5 - sx**3 - sy**5) * sym.exp(-(sx**2) - sy**2)
    - 1 / 3 * sym.exp(-((sx + 1) ** 2) - sy**2)
)


# create functions from the sympy-computed derivatives
df_x = sym.lambdify((sx, sy), sym.diff(sZ, sx), "sympy")
df_y = sym.lambdify((sx, sy), sym.diff(sZ, sy), "sympy")

df_x(1, 1).evalf()


# %%
# random starting point (uniform between -2 and +2)
localmin = np.random.rand(2) * 4 - 2  # also try specifying coordinates
startpnt = localmin[:]  # make a copy, not re-assign

# learning parameters
learning_rate = 0.01
training_epochs = 1000

# run through training
trajectory = np.zeros((training_epochs, 2))
for i in range(training_epochs):
    grad = np.array(
        [df_x(localmin[0], localmin[1]).evalf(), df_y(localmin[0], localmin[1]).evalf()]
    )
    localmin = (
        localmin - learning_rate * grad
    )  # add _ or [:] to change a variable in-place
    trajectory[i, :] = localmin


print(localmin)
print(startpnt)


# %%
# let's have a look!
plt.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]], vmin=-5, vmax=5, origin="lower")
plt.plot(startpnt[0], startpnt[1], "bs")
plt.plot(localmin[0], localmin[1], "ro")
plt.plot(trajectory[:, 0], trajectory[:, 1], "r")
plt.legend(["rnd start", "local min"])
plt.colorbar()
plt.show()


# %%
# # Additional explorations

# %%
# 1) Modify the code to force the initial guess to be [0,1.4]. Does the model reach a reasonable local minimum?
#
# 2) Using the same starting point, change the number of training epochs to 10,000. Does the final solution differ from
#    using 1000 epochs?
#
# 3) (Again with the same starting location) Change the learning to .1 (1000 epochs). What do you notice about the trajectory?
#    Try again with the learning rate set to .5, and then to .00001.
#
