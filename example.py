import numpy as np
from PI_module import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import skewnorm as sn


##############################################################################
# Data generation

# number of observations
n = 100 

# error terms, here skew normal with mean zero to demonstrate the effect of 
# skewed errors on the prediction intervals
scale = 1
shape = 6.3 # shape=6.3 <=> skewness=0.9, shape=2.62 <=> 0.6, shape=1.5 <=> 0.3
loc = -scale*np.sqrt(2/np.pi)*shape/np.sqrt(1+shape**2) # loc to satisfy E[e]=0
e = sn.rvs(shape, loc, scale, n).reshape((-1,1))

# true parameter vector
beta = np.array([0.1, 0.5]).reshape((-1,1))

# regressor matrix
X = np.random.normal(3, 1, n).reshape((-1,1))
X = sm.add_constant(X)

# dependent variable
Y = X @ beta + e


##############################################################################
# Estimation 

# create an OLS object and fit the model
mymod = OLS()
beta_OLS, stdbeta = mymod.fit(X, Y)

# evaluation points
X0 = sm.add_constant(np.linspace(0,6,100))

# significance level
alpha = 0.05

# model predictions
ypred = mymod.predict(X0)

# obtaining the confidence and prediction intervals
CI_low, CI_upp = mymod.CI(X0, alpha=alpha)
PI_low, PI_upp = mymod.PI(X0, alpha=alpha)


##############################################################################
# Plotting the results

plt.figure(figsize=(7,5))
plt.fill_between(X0[:,1], PI_low, PI_upp, 
                 alpha = 0.3, color='g', label='Prediction Interval')
plt.scatter(X[:,1], Y.flatten())
plt.plot(X0[:,1], CI_low, 'k', label = 'Confidence Interval')
plt.plot(X0[:,1], CI_upp, 'k')
plt.legend(loc='upper left')
plt.xlim(0,6)
plt.xlabel('Regressor')
plt.ylabel('Dependent variable')
 
