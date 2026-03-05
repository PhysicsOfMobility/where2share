import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from functools import cache, lru_cache

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from scipy.optimize import minimize

import pickle

# def fp_eq(q,r_0,qc, a=0.0):
#     return (1-qc/q-a*(q-qc)/q)

# def potential(r,f,r_0):
#     return -r*(2*(f)*r - 4*(f)*r_0 - 4*r**2/3 + 2*r*r_0)
# def prob_r(r, f,r_0,D):
#     return np.exp(-potential(r,f,r_0)/D)

# def r_mean(q,r_0,qc,D, a=0.0):
#     r = np.linspace(r_0, 1, 1000)[:, None]
#     f = fp_eq(q,r_0,qc,a)
#     p = prob_r(r, f, r_0, D)
#     return np.sum(r*p, axis=0)/np.sum(p,axis=0)


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
        """
        return -r*(2*(f)*r - 4*(f)*r_0 - 4*r**2/3 + 2*r*r_0)
    @staticmethod
    def prob_r(r, f, r_0, D):
        """
        proportional to the probability distribution of r
        derived from potential and fokker planck
        """
        return np.exp(-potential(r,f,r_0)/D)
    @staticmethod
    def log_prob_r(r, f, r_0, D):
        """
        see above
        """
        return -potential(r,f,r_0)/D
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
        f = fp_eq(q,r_0,qc,a,b)
        log_p = log_prob_r(r, f, r_0, D)
    
        log_weighted_sum_r = log_sum_exp(np.log(r) + log_p)
        
    
        log_sum_p = log_sum_exp(log_p)
        
        r_mean_value = np.exp(log_weighted_sum_r - log_sum_p)
        return r_mean_value


@lru_cache(maxsize=None)
def _make_r0_interpolator(index_tuple, columns_tuple, values_tuple):
    """
    Cached interpolator factory for default-rejection (r0) grids.
    Mirrors the make_interpolator() in testflask.py.

    Parameters
    ----------
    index_tuple   : tuple[float]  – fleet-size axis   (raw, NOT log2)
    columns_tuple : tuple[float]  – constraint axis    (raw)
    values_tuple  : tuple[float]  – flattened r0 grid  (row-major)
    """
    xs = np.log2(np.array(index_tuple, float))
    ys = np.array(columns_tuple, float)
    Z  = np.array(values_tuple, float).reshape(len(xs), len(ys))

    rgi = RegularGridInterpolator(
        (xs, ys), Z, method="linear", bounds_error=True
    )
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]

    def _interpolate(fleet_size, constraint):
        x = np.log2(fleet_size)
        if x > xmax or constraint > ymax:
            return 0.0
        if x < xmin or constraint < ymin:
            return 1.0
        return float(rgi((x, constraint)))

    return _interpolate


def _build_r0_interpolator(r0_values):
    """
    Convert a nested r0_values dict into a callable interpolator.
    Not itself cached, but the underlying RegularGridInterpolator is.

    Parameters
    ----------
    r0_values : dict
        Nested dict  r0_values[constraint_key][fleet_size_key] = r0_value
        (as stored in the GeoJSON / DataFrame row).

    Returns
    -------
    callable  (fleet_size, constraint) -> float
    """
    r0df = pd.DataFrame(r0_values)
    r0df.columns = [float(x) for x in r0df.columns]
    r0df.index   = [float(x) for x in r0df.index]
    r0df = r0df.sort_index().sort_index(axis=1)
    return _make_r0_interpolator(
        tuple(r0df.index),
        tuple(r0df.columns),
        tuple(r0df.to_numpy().ravel()),
    )




class CustomModel(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Theoretical model that adheres to Sklearn API.

    Has q_c and beta as unknown parameters, and simple heuristic parametric models to 
    describe these.
    The parameters are fitted using huber loss minimization, which is more robust to outliers.

    """
    def __init__(self, loss="huber"):
        self.loss = loss
        self.params_ = None

    def _crit_load(self, X, p):
        """
        parametrized model of the critical load
        """
        B, Dia, c = X[:, 15], X[:, 8], X[:, 0]
        return p[0] * np.log(1 + c * B / Dia) + p[1]

    def _beta_mod(self, X, p):
        """
        parametrized model of the cherry picking exponent beta.
        Depends on the critical load.
        """
        B, Dia, c, nedges = X[:, 15], X[:, 8], X[:, 0], X[:, 10]
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
        q, B, r0 = X[:, 2], X[:, 15], X[:, 1]
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
    Boost the predictionf of the theoretical model.
    """
    def __init__(self, boosting_model):
        self.boosting_model = boosting_model
        self.theory_model = None

    def fit(self, X, y):
        self.theory_model.fit(X,y)
        if self.theory_model is None:
            raise ValueError("Call set_theory_model(...) with a fitted model before fit().")
        self.theory_pred = self.theory_model.predict(X)
        residuals = y - self.theory_pred
        self.boosting_model.fit(X, residuals)
        return self

    def predict(self, X):
        theory_pred = self.theory_model.predict(X)
        residual_pred = self.boosting_model.predict(X)
        return theory_pred + residual_pred

    def set_theory_model(self, model):
        self.theory_model = model

class PerformanceOracle:
    """
    Performance oracle backed by the new feature-restricted boosting model.

    The new model expects an 8-column pandas DataFrame:
        constraint, default_rejections, load, network_diameter,
        network_distance, network_edges, network_nodes, number_of_transporters

    It replaces both the legacy TheoreticalModel path *and* the old
    JointModel / sklearn path of the original PerformanceOracle.
    """

    def __init__(self, model_file):
        """
        Parameters
        ----------
        model_file : str
            Path to the pickled new model (e.g. 'feature_restricted_model.pkl').
            The model object must expose .predict(X) and, optionally,
            ._theory_predict(X).
        """
        with open(model_file, "rb") as fh:
            self.model = pickle.load(fh)

    # ------------------------------------------------------------------ #
    #  Feature-vector construction (mirrors build_features in testflask)  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_feature_row(
        fleet_size, constraint, load, default_rejections,
        diameter, average_distance, number_of_edges, number_of_nodes,
    ):
        """Return a single feature dict ready for pd.DataFrame construction."""
        return {
            "constraint":             float(constraint),
            "default_rejections":     float(default_rejections),
            "load":                   float(load),
            "network_diameter":       float(diameter),
            "network_distance":       float(average_distance),
            "network_edges":          float(number_of_edges),
            "network_nodes":          float(number_of_nodes),
            "number_of_transporters": float(fleet_size),
        }

    # ------------------------------------------------------------------ #
    #  Main performance-curve method                                      #
    # ------------------------------------------------------------------ #
    def performance_curve(
        self,
        pop,
        diameter,
        average_distance,
        r0_values,
        adoption_fraction,
        constraint=10.0,
        number_of_edges=0,
        number_of_nodes=0,
    ):
        """
        Compute the performance curve for one region & constraint value.

        Parameters
        ----------
        pop               : float – region population
        diameter          : float – network diameter (minutes)
        average_distance  : float – average trip distance (minutes)
        r0_values         : dict  – nested dict of default-rejection grids
        adoption_fraction : float – fraction of population using ridepooling
        constraint        : float – maximum allowed delay
        number_of_edges   : int   – edges in the road network
        number_of_nodes   : int   – nodes in the road network

        Returns
        -------
        np.ndarray, shape (n_loads + 1, 6)
            Columns: [load, rejection, acceptance,
                      rel_driven_distance, fleet_size, is_taxi]
            Row 0 contains boundary values for load = 0.
        """
        # Request rate (identical to original)
        request_rate = pop * 3.2 * adoption_fraction / (17 * 60)

        # Build r0 interpolator from the new-style dict
        r0_func = _build_r0_interpolator(r0_values)

        # Load grid (identical to original)
        target_loads = np.concatenate([
            np.linspace(0.1, 2.0, 50),
            np.linspace(2.1, 5.0, 20),
            np.linspace(5.5, 15.0, 10),
        ])
        n = len(target_loads)

        rejections   = np.full(n, np.nan)
        actual_loads = np.full(n, np.nan)
        fleets       = np.full(n, np.nan)

        feature_rows        = []
        indices_to_predict  = []

        for i, qi in enumerate(target_loads):
            fleet_size = max(request_rate / qi * average_distance, 1.0)

            # New-style r0 interpolation (fleet_size, constraint)
            r0 = np.clip(r0_func(fleet_size, constraint), 0.0, 1.0)

            feature_rows.append(self._build_feature_row(
                fleet_size, constraint, qi, r0,
                diameter, average_distance, number_of_edges, number_of_nodes,
            ))
            indices_to_predict.append(i)
            actual_loads[i] = qi
            fleets[i]       = fleet_size

            # Once we're down to a single vehicle, freeze the curve
            if fleet_size <= 1.0:
                for j in range(i + 1, n):
                    actual_loads[j] = qi
                    fleets[j]       = fleet_size
                break

        # --- Batch prediction (much faster than one-by-one) ---
        X     = pd.DataFrame(feature_rows)
        preds = self.model.predict(X).ravel()

        for k, idx in enumerate(indices_to_predict):
            rejections[idx] = preds[k]

        # Fill any remaining slots (fleet_size <= 1 early-exit case)
        last_idx = indices_to_predict[-1]
        if last_idx < n - 1:
            rejections[last_idx + 1 :] = rejections[last_idx]

        # --- Relative driven distance (same logic as original) ---
        taxi_service_distance = 1.0 + fleets ** (-0.5)

        relative_driven_distance = (
            np.minimum(1.0 / actual_loads, taxi_service_distance)
            + rejections - 1.0
        )
        # 3-point moving-average smoothing (interior only)
        relative_driven_distance[1:-1] = (
            relative_driven_distance[:-2]
            + relative_driven_distance[1:-1]
            + relative_driven_distance[2:]
        ) / 3.0

        # --- Assemble output array ---
        plotting_data = np.zeros((n + 1, 6))

        plotting_data[1:, 0] = actual_loads
        plotting_data[1:, 1] = rejections
        plotting_data[1:, 2] = 1.0 - rejections
        plotting_data[1:, 3] = relative_driven_distance
        plotting_data[1:, 4] = fleets
        plotting_data[1:, 5] = (1.0 / actual_loads < taxi_service_distance).astype(int)

        # Boundary row at load = 0
        plotting_data[0, :] = [0, 0, 1, 0, pop / 2, 0]

        return plotting_data

    # ------------------------------------------------------------------ #
    #  Optional: expose theory-only predictions                          #
    # ------------------------------------------------------------------ #
    def theory_predict(self, X):
        """Delegate to the model's theory-only head, if available."""
        if hasattr(self.model, "_theory_predict"):
            return self.model._theory_predict(X)
        raise AttributeError("Loaded model has no _theory_predict method.")


class RidepoolingCosts:
    def __init__(self, modelfile: str, gamma_T=0.098, gamma_L=0.525):
        self.gamma_T = gamma_T
        self.gamma_L = gamma_L
        # ---- Use the new oracle ----
        self.Oracle = PerformanceOracle(model_file=modelfile)

    # linear_utility_function stays the same ...

    def create_surface(self, row, adoption_rate=0.01):
        """
        Now passes r0_values directly and includes network topology fields.
        No longer needs to flatten r0_values into (cs, lbs, vals) tuples.
        """
        curves = []
        constraints = np.arange(1, 61, 1)

        for c in constraints:
            curve = self.Oracle.performance_curve(
                pop=row["bev_21"],
                diameter=row["diameter"],
                average_distance=row["average_distance"],
                r0_values=row["r0_values"],          # pass dict directly
                adoption_fraction=adoption_rate,
                constraint=c,
                number_of_edges=row["number_of_edges"],  # new
                number_of_nodes=row["number_of_nodes"],  # new
            )
            curves.append(curve)

        # Rest of surface assembly is unchanged
        Xs    = np.zeros((60, 81))
        Ys    = np.zeros((60, 81))
        Zs    = np.zeros((60, 81))
        isRPs = np.zeros((60, 81))

        for i in range(len(curves)):
            Xs[i, :] = curves[i][:, 3]
            for j in range(len(curves[i])):
                Ys[:, j] = constraints
                Zs[i, j] = curves[i][j][1]
                isRPs[i, j] = curves[i][j][5]
        return Xs, Ys, Zs, isRPs
    @staticmethod
    def linear_utility_function(r, rel_driv_dist, c, ell, gamma_T, gamma_L):
        tau_r = c + ell
        vel = 1.0  # 50 km/h
        return gamma_T * c + gamma_L * ell * rel_driv_dist + gamma_T * tau_r * r

    def determine_optimal_parameters(self, row, adoption_rate=0.05):
        """
        row is a pandas Series containing ridepooling parameters. Necessary entries are:
        'bev_21' : The total population of the considered region
        'diameter': the diameter of the network in minutes
        'average_distance': the average distance in the network in minutes
        """
        Xs, Ys, Zs, isRPs = self.create_surface(row, adoption_rate=adoption_rate)
        C = RidepoolingCosts.linear_utility_function(
            Zs, Xs + 1, Ys, row["average_distance"], self.gamma_T, self.gamma_L
        )
        mask = np.any(~np.isnan(C), axis=0)
        C = C[:, mask]
        Xs = Xs[:, mask]
        Ys = Ys[:, mask]
        Zs = Zs[:, mask]
        isRPs = isRPs[:, mask]
        mask2 = C == np.min(C)
        best_load = Xs[mask2].reshape(-1)[0]
        best_constraint = Ys[mask2].reshape(-1)[0]
        expected_rejections = Zs[mask2].reshape(-1)[0]
        isRP = isRPs[mask2].reshape(-1)[0]
        return best_load, best_constraint, expected_rejections, isRP, np.min(C)

    def get_optimum_for_regions(self, region_df, adoption_rate=0.05):
        return zip(
            *(
                region_df.apply(
                    self.determine_optimal_parameters,
                    axis=1,
                    adoption_rate=adoption_rate,
                )
            )
        )
