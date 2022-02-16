import numpy as np
from scipy.stats import norm
from numpy.linalg import inv
from scipy.optimize import minimize, bisect

class OLS():
    """
    Class for fitting a linear model by OLS, computing confidence intervals and 
    computing valid prediction intervals for non-normal error terms. 
    
    Methods available to the user:
    - fit: estimates the underlying linear model by OLS
    - predict: predicts the dependent variable for provided regressors
    - CI: computes confidence intervals for the predictions for the provided 
        regressors
    - PI: computes prediction intervals for the predictions for the provided 
        regressors
    
    The methods predict, CI and PI can be used only after fitting the model,
    i.e. running the method fit.
    """
    
    def __init__(self):
        # indicator, whether model has been fitted
        self.fitted = False
    
    def fit(self, X, Y):
        """
        fit(self, X, Y)
        
        Fit an underlying linear model by OLS.

        Parameters
        ----------
        X : numpy.ndarray
            (n,k) array, regressor matrix. Should contain a column of ones, if
            an intercept should be fitted.
            n - number of observations
            k - number of regressors (potentially including a constant)
        Y : numpy.ndarray
            (n,1) array of dependent variables.
            n - number of observations

        Returns
        -------
        beta : numpy.ndarray
            (k,1) array, OLS parameter estimator.
            k - number of regressors (potentially including a constant)
        stdbeta : numpy.ndarray
            (k) array, estimated standard deviations of the OLS estimator.
            k - number of regressors (potentially including a constant)
        """
        
        self.X = X
        self.Y = Y
        
        # OLS estimator
        beta = inv(X.T @ X) @ X.T @ Y
        self.beta = beta
        
        # residual vector
        self.resid = Y - X @ beta
        
        # covariance matrix and standard deviations of OLS estimator
        Vbeta = self.__compute_Vbeta__()
        stdbeta = np.sqrt(np.diag(Vbeta))
        self.Vbeta = Vbeta
        
        # set indicator, whether model has been fitted, to True
        self.fitted = True
        
        return beta, stdbeta
    
    
    def predict(self, X0):
        """
        predict(self, X0)
        
        Predicts the dependent variable based on the fitted model for the 
        evaluation points provided in X0.

        Parameters
        ----------
        X0 : numpy.ndarray
            (m,k) array of evaluation points.
            m - number of evaluation points
            k - number of regressors (potentially including a constant)

        Returns
        -------
        yhat : numpy.ndarray
            (m,1) array of predictions.
            m - number of evaluation points
        """
        
        if self.fitted == False:
            raise ValueError('fit the model before making predictions')
        
        # predict dependent variable for X0
        yhat = X0 @ self.beta
        
        return yhat
    
    
    def CI(self, X0, alpha=0.05):
        """
        CI(self, X0, alpha=0.05)
        
        Computes the confidence intervals for the fitted model and the 
        evaluation points provided in X0.

        Parameters
        ----------
        X0 : numpy.ndarray
            (m,k) array of evaluation points.
            m - number of evaluation points
            k - number of regressors (potentially including a constant)
        alpha : float, optional
            Significance level. The default is 0.05.

        Returns
        -------
        CI_low : numpy.ndarray
            (m) array, lower limits of the confidence intervals.
            m - number of evaluation points
        CI_upp : numpy.ndarray
            (m) array, upper limits of the confidence intervals.
            m - number of evaluation points
        """
        
        if self.fitted == False:
            raise ValueError('fit the model before computing CIs')
        
        m,k = X0.shape
        
        # compute the variances of the predictions yhat=X0@beta
        Vyhat = np.zeros(m)
        for i in range(m):
            xi = X0[i,:].reshape((-1,1))
            Vyhat[i] = xi.T @ self.Vbeta @ xi
        self.Vyhat = Vyhat
        
        # obtain the corresponding standard deviations
        stdyhat = np.sqrt(Vyhat).reshape((-1,1))
        
        # standard normal quantile
        q = norm.isf(alpha/2)
        
        # obtain the predictions
        ypred = self.predict(X0)
        
        # construct the confidence intervals
        CI_upp = (ypred + q * stdyhat).flatten()
        CI_low = (ypred - q * stdyhat).flatten()
        
        return CI_low, CI_upp
        
    
    def PI(self, X0, alpha=0.05, method='quantile', smooth=False):
        """
        PI(self, X0, alpha=0.05, method='quantile', smooth=False)
        
        Computes the prediction intervals for the fitted model and the 
        evaluation points provided in X0.

        Parameters
        ----------
        X0 : numpy.ndarray
            (m,k) array of evaluation points.
            m - number of evaluation points
            k - number of regressors (potentially including a constant)
        alpha : float, optional
            Significance level. The default is 0.05.
        method : str, optional
            Method for computing the prediction intervals. Should be one of:
            - 'quantile' (ses the alpha/2 and 1-alpha/2 quantiles as PI)
            - 'HD' (identifies the shortest PI with coverage 1-alpha)
            The method 'HD' (highest density) should be used in combination 
            with smooth=True for stability reasons. The default is 'quantile'.
        smooth : bool, optional
            Indicates, whether the pdf and cdf should be smoothed via kernel 
            density estimation for the residuals. The default is False.

        Returns
        -------
        PI_low : numpy.ndarray
            (m) array, lower limits of the prediction intervals.
            m - number of evaluation points
        PI_upp : numpy.ndarray
            (m) array, upper limits of the prediction intervals.
            m - number of evaluation points
        """
        
        if self.fitted == False:
            raise ValueError('fit the model before computing PIs')
        
        m,k = X0.shape
        
        # obtain predictions
        ypred = self.predict(X0).flatten()
        
        # compute confidence intervals to obtain Vyhat
        CI_low, CI_upp = self.CI(X0, alpha)
        
        # standard deviation of predictions
        stdyhat = np.sqrt(self.Vyhat)
        
        # obtain the standard deviation of yhat+error under normality
        stdy = np.sqrt(stdyhat**2 + self.sig2)
        
        # obtain the 1-alpha/2 quantile of the standard normal distribution for
        # the construction of the starting point for the root algorithm below
        q = norm.isf(alpha/2)
        
        # obtain the bandwidth for smoothing
        if smooth==True:
            n = self.X.shape[0]
            var_silverman = np.min([np.sqrt(self.sig2), 
                                    (np.quantile(self.resid.flatten(),0.75)-
                                     np.quantile(self.resid.flatten(),0.25))/1.349])
            h = 0.9 * var_silverman * n**(-1/5)
        
        # compute the prediction intervals
        PI_upp = np.zeros(m)
        PI_low = np.zeros(m)
        
        for i, (mu, sig, sigy) in enumerate(zip(ypred, stdyhat, stdy)):
            # add the bandwidth to the kernel width, if smooth==True
            if smooth==True:
                sig = np.sqrt(sig**2+h**2)
            
            # estimate of cdf and pdf for prediction intervals
            F = lambda z: np.mean(norm.cdf(z - self.resid, mu, sig))
            f = lambda z: np.mean(norm.pdf(z - self.resid, mu, sig))
            
            if method=='quantile':
                # obtain alpha/2 and 1-alpha/2 quantiles
                PI_low[i] = bisect(lambda z: F(z) - alpha/2, mu-4*sigy, mu)
                PI_upp[i] = bisect(lambda z: F(z) - (1-alpha/2), mu, mu+4*sigy)
            
            elif method=='HD':
                # identify the shortest interval with coverage 1-alpha by 
                # minimizing the length, conditional on the correct coverage.
                # the optimization parameter x contains two elements, x[0] is 
                # the lower limit of the interval, x[1] is the interval length.
                f2min = lambda x: x[1]
                
                # We use the lower interval limit and interval width under 
                # normality of the error terms as starting values
                x0 = np.array([mu-q*sigy, 2*q*sigy])
                
                # jacobian of the optimization function
                jac = lambda x: np.array([0,1])
                
                # equality constraint to guarantee correct coverage 1-alpha
                eq_cons = {'type': 'eq',
                            'fun' : lambda x: F(x[0]+x[1]) - F(x[0]) - (1-alpha),
                            'jac' : lambda x: np.array([f(x[0]+x[1]) - f(x[0]),
                                                        f(x[0]+x[1])])}
                
                # inequality constraint to guarantee positive interval width
                ineq_cons = {'type': 'ineq',
                             'fun' : lambda x: x[1],
                             'jac' : lambda x: np.array([0,1])}
                
                # obtainint the prediction intervals
                min_result = minimize(f2min, x0, method='SLSQP', jac=jac,
                                      constraints=[eq_cons,ineq_cons], 
                                      options={'ftol': 10e-4}).x
                PI_low[i] = min_result[0]
                PI_upp[i] = min_result[0]+min_result[1]
            
            else:
                raise ValueError("method should be either 'HD' or 'quantile'")
        
        return PI_low, PI_upp
    
    
    def __compute_Vbeta__(self):
        """
        __compute_Vbeta__(self)
        
        Computes the covariance matrix of the OLS estimator

        Returns
        -------
        Vbeta : numpy.ndarray
            (k,k) array, covariance matrix of the OLS estimator.
            k - number of regressors (potentially including a constant)
        """
        
        X = self.X
        resid = self.resid
        n,k = X.shape
        
        # estimating the variance of the error term
        sig2 = 1/(n-k) * np.sum(resid**2)
        self.sig2 = sig2
        
        # estimating the covariance matrix of the OLS estimator
        Vbeta = sig2 * inv(X.T @ X)
        
        return Vbeta

