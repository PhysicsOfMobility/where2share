import pandas as pd
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from scipy.optimize import curve_fit,minimize
import numpy as np

class helpers:
    def __init__(self):
        print("Not meant to be used like this")
    @staticmethod
    def fp_eq(q,r_0,qc,a=0.0,b=0.0):
        """
        This is one of the fixed points of the transcritical bifurcation form.
        (It's the nontrivial fixed point)
        """
        return 1-(qc/q)**(1-b)
    @staticmethod
    def potential(r,f,r_0):
        """
        This is the potential whose negative derivative will
        return the differential equation.
        with the most recent version of the model $r_0$ will always be 0,
        because its effect is absorbed into the diffusion statistics.
        (I think this has a factor 4 too much in it! This will result in a D which cancels this factor out)
        """
        return -r*(2*(f)*r - 4*(f)*r_0 - 4*r**2/3 + 2*r*r_0)
    @staticmethod
    def prob_r(r, f, r_0, D):
        """
        proportional to the probability distribution of r
        derived from potential and fokker planck
        """
        return np.exp(-helpers.potential(r,f,r_0)/D)
    @staticmethod
    def log_prob_r(r, f, r_0, D):
        """
        see above
        """
        return -helpers.potential(r,f,r_0)/D
    @staticmethod
    def log_sum_exp(log_x):
        """
        a safer way to sum over exponentials
        """
        max_log_x = np.max(log_x, axis=0, keepdims=True)
        return max_log_x + np.log(np.sum(np.exp(log_x - max_log_x), axis=0))
    
    @staticmethod
    def r_mean(q,r_0,qc,D, a=0.0, b=0.0):
        """
        expectation value of the above mentioned probability distribution
        gives the expected rejection rate.

        in the most recent version of the model r_0 will be 0 and D is calculated from r_0.
        """
        epsilon = np.ones_like(r_0)*1e-10
        epsilon[r_0>0] = 0.0
    
        r = np.linspace(epsilon, 1, 1000)[:, None]
        f = helpers.fp_eq(q,r_0,qc,a,b)
        log_p = helpers.log_prob_r(r, f, r_0, D)
    
        log_weighted_sum_r = helpers.log_sum_exp(np.log(r) + log_p)
        
    
        log_sum_p = helpers.log_sum_exp(log_p)
        
        r_mean_value = np.exp(log_weighted_sum_r - log_sum_p)
        return r_mean_value


class CustomModel(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Theoretical model that adheres to Sklearn API.

    Has q_c and beta as unknown parameters, and simple heuristic parametric models to 
    describe these.
    The parameters are fitted using huber loss minimization, which is more robust to outliers.

    features: ['constraint', 'default_rejections', 'load', 'network_diameter', 'network_edges',
       'number_of_transporters']
    """
    def __init__(self, loss="huber"):
        self.loss = loss
        self.params_ = None

    def _crit_load(self, X, p):
        """
        parametrized model of the critical load
        """
        B, Dia, c = X[:, 5], X[:, 3], X[:, 0]
        return p[0] * np.log(1 + c * B / Dia) + p[1]

    def _beta_mod(self, X, p):
        """
        parametrized model of the cherry picking exponent beta.
        Depends on the critical load.
        """
        B, Dia, c, nedges = X[:, 5], X[:, 3], X[:, 0], X[:, 4]
        pseudo_qc = p[0] * np.log(1 + c * B / Dia) + p[1]
        xbeta = pseudo_qc / np.sqrt(nedges)
        return p[2] * xbeta + p[3]

    def _model_function(self, X, p):
        """
        Expected rejection rate according to the transcritical model with noise.

        r_mean is the expected value obtained from the fokker planck formalism.
        The noise level is determined from the default rejection rate, by analytically (approximatively)
        calculating the expected rejection rate when q -> 0. By requiring that this 
        matches the default rejections r_0, on finds an expression for D, which needs 
        to scale very specifically with q to achieve the r_0 rejection rate.
        """
        q, B, r0 = X[:, 2], X[:, 5], X[:, 1]
        qc = self._crit_load(X, p)
        Beta = self._beta_mod(X, p)
        D = 2 * np.pi * r0**2 * (np.abs(qc / q))**(1 - Beta) + 1e-6
        return helpers.r_mean(q, np.zeros_like(r0), qc, D, b=Beta).reshape((len(X), -1))
    
    def _residuals(self, p, X, y, delta=0.2):
        yp = self._model_function(X, p).ravel()
        error = np.abs(y - yp)
        if self.loss == "huber":
            mask = error < delta
            residual = np.zeros_like(yp)
            residual[mask] = 0.5 * error[mask]**2
            residual[~mask] = delta * (error[~mask] - 0.5 * delta)
            return np.sum(residual)
        else:
            return np.sum((y - yp)**2)

    def fit(self, X, y):
        for col in X.columns:
            assert col in ['constraint', 'default_rejections', 'load', 'network_diameter', 'network_edges',
       'number_of_transporters']
        X = np.array(X)
        y = np.array(y).ravel()
        p0 = np.ones(4)
        res = minimize(self._residuals, p0, args=(X, y), bounds=[(1e-6, None)] * 4)
        self.params_ = res.x
        return self
        
    def transform(self, X):
        self.pred = self.predict(X)
        return X
        
    def predict(self, X):
        return self._model_function(np.array(X), self.params_).ravel()

class ResidualBoostingModel(BaseEstimator, RegressorMixin):
    """
    Boost the prediction of the theoretical model.

    Input is expected to have the following features:
    'constraint',
    'default_rejections',
    'load',
    'network_diameter',
    'network_distance',
    'network_edges',
    'network_nodes',
    'number_of_transporters'
    
    """
    def __init__(self, boosting_model, theory_model, ignore_columns = []):
        self.boosting_model = boosting_model
        self.theory_model = theory_model
        self.ignore_columns = ignore_columns
        self.boosting_columns = ['constraint',
                                 'default_rejections',
                                 'load',
                                 'network_diameter',
                                 'network_distance',
                                 'network_edges',
                                 'network_nodes',
                                 'number_of_transporters'
                                ]
        if self.ignore_columns:
            self.boosting_columns = [x for x in self.boosting_columns if not x in self.ignore_columns]
        self.theory_columns = ['constraint', 
                               'default_rejections', 
                               'load', 
                               'network_diameter',
                               'network_edges',
                               'number_of_transporters'
                                ]
        self._renaming_schema = {
            'average_distance':'network_distance',
            'number_of_nodes':'network_nodes',
            'number_of_edges':'network_edges',
            'diameter':'network_diameter'
        }
    def _rename_columns(self, X):
        X_new = X.copy()
        X_new.columns = [self._renaming_schema[x] if x in self._renaming_schema else x  for x in X_new.columns]
        return X_new
    def fit(self, X, y):
        # First take care of column names:
        X_new = self._rename_columns(X)
       #  for col in X_new.columns:
       #      assert col in ['constraint', 'default_rejections', 'load', 'network_diameter', 'network_edges',
       # 'number_of_transporters'], f"{col} unexpected"
            
        self.theory_model.fit(X_new.loc[:,self.theory_columns],y)
        theory_pred = self.theory_model.predict(X_new.loc[:,self.theory_columns])
        residuals = y - theory_pred
        self.boosting_model.fit(X_new.loc[:,self.boosting_columns], residuals)
        return self

    def predict(self, X):
        X_new = self._rename_columns(X)
        theory_pred = self.theory_model.predict(X_new.loc[:,self.theory_columns])
        residual_pred = self.boosting_model.predict(X_new.loc[:,self.boosting_columns])
        prediction = (theory_pred + residual_pred)
        return prediction*(prediction>=0)*(prediction<=1)+(prediction>1).astype(float)
        
    def _theory_predict(self, X):
        X_new = self._rename_columns(X)
        theory_pred = self.theory_model.predict(X_new.loc[:,self.theory_columns])
        # residual_pred = self.boosting_model.predict(X_new.loc[:,self.boosting_columns])
        return theory_pred
    def _residual_predict(self, X):
        X_new = self._rename_columns(X)
        theory_pred = self.theory_model.predict(X_new.loc[:,self.theory_columns])
        residual_pred = self.boosting_model.predict(X_new.loc[:,self.boosting_columns])
        return residual_pred

    def set_theory_model(self, model):
        self.theory_model = model