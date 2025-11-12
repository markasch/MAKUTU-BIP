"""
Like other Numpy or Scipy-based functions, the `scipy.integrate.odeint` function 
cannot be used directly in a PyMC model because PyMC needs to know the variable 
input and output types to compile.  
Therefore, we use a Pytensor wrapper to give the variable types to PyMC.  
Then the function can be used in PyMC in conjunction with gradient-free samplers.  
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from scipy.integrate import odeint

# set up and solve the Lotka-Volterra system
theta = np.array([0.52, 0.026, 0.84, 0.026, 34.0, 6.0]) 
time = np.arange(1900, 1921, 0.01)
def rhs(X, t, theta):
    # unpack
    x, y = X
    alpha, beta, gamma, delta, xt0, yt0 = theta
    # equations
    dx_dt = ...
    dy_dt = ...
    return [dx_dt, dy_dt]
# call Scipy's odeint function
x_y = odeint(func=rhs, y0=theta[-2:], t=time, args=(theta,))
# plot data and solution
...

# solve Least Squares deterministic inverse problem
...
results = least_squares(ode_model_resid, x0=theta)

# Use PYMC to solve the Bayesian inverse problem:
# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return odeint(func=rhs, y0=theta[-2:], t=data.year, args=(theta,))

theta = results.x  # use Least Squares solution to inform the priors
with pm.Model() as model:
    # Priors
    alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
    beta  = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
    gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
    delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
    xt0   = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
    yt0   = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
    sigma = pm.HalfNormal("sigma", 10)

    # ODE solution function
    ode_solution = pytensor_forward_model_matrix(
        pm.math.stack([alpha, beta, gamma, delta, xt0, yt0])
    )

    # Likelihood
    pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data[["hare", "lynx"]].values)

# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]

# sample the chain...
sampler = "DEMetropolis"
chains = 8
draws = 6000
with model:
    trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], draws=draws, chains=chains)
trace = trace_DEM
az.summary(trace)

# graphical output
az.plot_trace(trace, kind="rank_bars")
plt.suptitle(f"Trace Plot {sampler}");