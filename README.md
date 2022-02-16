# Prediction-Intervals

The moduel PI_module implements valid prediction intervals for general error terms and is based on the paper "Semiparametric Prediction Intervals in Parametric Models with Non-normal Additive Error Terms", soon available on arXiv. The link will be added here.

PI_module contains the class OLS for fitting a linear model by OLS, computing confidence intervals and computing valid prediction intervals for non-normal error terms. 
    
Methods available to the user:
- fit: estimates the underlying linear model by OLS
- predict: predicts the dependent variable for provided regressors
- CI: computes confidence intervals for the predictions for the provided regressors
- PI: computes prediction intervals for the predictions for the provided regressors
    
The methods predict, CI and PI can be used only after fitting the model, i.e. running the method fit.

The file example.py shows how to use the module
