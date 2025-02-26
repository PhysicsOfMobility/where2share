import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from functools import cache

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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


class TheoreticalModel:
    def __init__(
        self, p_D, p_q, epochs=1000, beta0=1e3, beta_change=2e-4, rand_factor=2
    ):
        """
        Columns of X should be (in order):
        fleet size, load, average distance, diameter, allowed delay, default rejection probability
        B, q, l, d, c, r0
        """
        self.D_params = p_D
        self.q_params = p_q
        self.epochs = epochs
        self.beta0 = beta0
        self.beta = beta0
        self.beta_change = beta_change
        self.rand_factor = rand_factor
        self.history = np.zeros(self.epochs)
        self.epochs_trained = 0
        pass

    def beta_step(self):
        self.beta = self.beta * (1 + self.beta_change)

    def random_step(self):
        D_change = np.random.normal(
            size=len(self.D_params),
            scale=np.sqrt(1 / len(self.D_params))
            * self.rand_factor
            * np.sqrt(1 / self.beta),
        )
        q_change = np.random.normal(
            size=len(self.q_params),
            scale=np.sqrt(1 / len(self.q_params))
            * self.rand_factor
            * np.sqrt(1 / self.beta),
        )
        new_D_params = self.D_params + D_change
        new_q_params = self.q_params + q_change
        return new_D_params, new_q_params

    def fit(self, X, y):
        self.best_MSE = 1e6
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            D = -np.ones(len(X))
            qc = -np.ones(len(X))
            new_D_params = np.zeros_like(self.D_params)
            new_q_params = np.zeros_like(self.q_params)
            while np.any(D <= 0) or np.any(qc < 0):
                new_D_params, new_q_params = self.random_step()
                D = self.D_model(X, new_D_params)
                qc = self.qc_model(X, new_q_params)

            r = TheoreticalModel.r_mean(X[:, 1], X[:, 5], qc, D)
            sqerr = (r - y) ** 2
            MSE = np.mean(sqerr[~np.isnan(sqerr)])
            if MSE < self.best_MSE:
                self.D_params = new_D_params
                self.q_params = new_q_params
                self.best_MSE = MSE
            elif np.random.random() < np.exp((self.best_MSE - MSE) * self.beta):
                self.D_params = new_D_params
                self.q_params = new_q_params
                self.best_MSE = MSE
            else:
                pass

            self.beta_step()
            self.epochs_trained += 1
            self.history[epoch] = self.best_MSE

    def predict(self, X, y=None):
        D = self.D_model(X)
        qc = self.qc_model(X)
        r = TheoreticalModel.r_mean(X[:, 1], X[:, 5], qc, D)
        return r

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X, y)

    def D_model(self, X, params=None):
        if params is None:
            return (
                self.D_params[0]
                + self.D_params[1] * (X[:, 0] * X[:, 1]) ** self.D_params[2]
            )
        else:
            return params[0] + params[1] * (X[:, 0] * X[:, 1]) ** params[2]

    def qc_model(self, X, params=None):
        """
        The critical load model should not depend on the load.
        However, given that the theory may not be accurate a load dependence may also account for
        slightly different shapes of the
        """
        if params is None:
            return (
                self.q_params[0]
                * X[:, 0] ** (self.q_params[1])
                * (X[:, 4] / X[:, 3]) ** self.q_params[2]
            )
        else:
            return params[0] * X[:, 0] ** (params[1]) * (X[:, 4] / X[:, 3]) ** params[2]

    @staticmethod
    def fp_eq(q, r_0, qc, a=0.0):
        return 1 - qc / q - a * (q - qc) / q

    @staticmethod
    def potential(r, f, r_0):
        return -r * (2 * (f) * r - 4 * (f) * r_0 - 4 * r**2 / 3 + 2 * r * r_0)

    @staticmethod
    def prob_r(r, f, r_0, D):
        return np.exp(-TheoreticalModel.potential(r, f, r_0) / D)

    @staticmethod
    def log_prob_r(r, f, r_0, D):
        return -TheoreticalModel.potential(r, f, r_0) / D

    @staticmethod
    def log_sum_exp(log_x):
        max_log_x = np.max(log_x, axis=0, keepdims=True)
        return max_log_x + np.log(np.sum(np.exp(log_x - max_log_x), axis=0))

    @staticmethod
    def r_mean(q, r_0, qc, D, a=0.0):

        epsilon = np.ones_like(r_0) * 1e-10
        epsilon[r_0 > 0] = 0.0

        r = np.linspace(r_0 + epsilon, 1, 1000)[:, None]
        f = TheoreticalModel.fp_eq(q, r_0, qc, a)
        log_p = TheoreticalModel.log_prob_r(r, f, r_0, D)
        # Compute the log of the weighted sum of r
        log_weighted_sum_r = TheoreticalModel.log_sum_exp(np.log(r) + log_p)

        # Compute the log of the sum of probabilities
        log_sum_p = TheoreticalModel.log_sum_exp(log_p)

        # Compute r_mean using the exponentials of the log sums
        r_mean_value = np.exp(log_weighted_sum_r - log_sum_p)
        return r_mean_value


class JointModel:
    def __init__(self, theory=None, ml_model=None):

        self.theory = (
            TheoreticalModel(
                np.array([0.059756, 0.05154776, -0.70109993]),
                np.array([0.56957613, 0.22253386, 0.26101479]),
            )
            if theory is None
            else theory
        )

        self.ml_model = (
            Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=0.1, epsilon=0.032))])
            if ml_model is None
            else ml_model
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.theory.fit(X, y)
        r1 = self.theory.predict(X).reshape(-1)
        y1 = y - r1
        Xc = X.copy()
        Xc[:, 0] = np.log2(Xc[:, 0])
        self.ml_model.fit(Xc, y1)

    def predict(self, X, y=None):
        r1 = self.theory.predict(X).reshape(-1)
        new_X = X.copy()
        new_X[:, 0] = np.log2(new_X[:, 0])
        dr1 = self.ml_model.predict(new_X).reshape(-1)
        return r1 + dr1

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def save(self, fname):
        self.__module__ = "where2share.models"
        pickle.dump(self, open(fname, "wb"))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, "rb"))


class PerformanceOracle:
    def __init__(self, pfile=None, use_sklearn=False):
        self.use_sklearn = use_sklearn
        if not use_sklearn:
            if pfile is not None:
                p_D, p_q, p_a = pickle.load(open(pfile, "rb"))
                self.diffusion_parameters = p_D
                self.crit_load_parameters = p_q
            else:
                self.crit_load_parameters = np.array(
                    [0.56974369, 0.21716436, 0.22031338]
                )
                self.diffusion_parameters = np.array(
                    [0.05129071, 0.05505149, -0.6594481]
                )
        else:
            self.model = JointModel()
            print(self.model.__module__)
            self.model = pickle.load(open(pfile, "rb"))

    def critical_load_model(self, fleet_size, constraint, diameter):
        qc = (
            self.crit_load_parameters[0]
            * fleet_size ** (self.crit_load_parameters[1])
            * (constraint / diameter) ** (self.crit_load_parameters[2])
        )
        return qc

    def noise_model(self, fleet_size, load):
        D = self.diffusion_parameters[0] + self.diffusion_parameters[1] * (
            fleet_size * load
        ) ** (self.diffusion_parameters[2])
        return D

    def upgrade_prediction(self):

        pass

    def interpolate_r0(self, cs, lbs, vals, average_distance, fleet_size, constraint):
        data = pd.DataFrame(
            data={"logB": lbs, "c": cs, "logr0": np.log10(np.array(vals) + 1e-10)}
        )
        data = (
            data.groupby(["logB", "c"])
            .mean()["logr0"]
            .unstack()
            .sort_index()
            .T.sort_index()
            .T
        )
        interpolator = RegularGridInterpolator(
            (data.index, data.columns), data.values, bounds_error=False, fill_value=None
        )
        r0_estimate = (
            10 ** interpolator(([np.log2(fleet_size)], [constraint / average_distance]))
        ) - 1e-10
        if r0_estimate[0] < 0:
            return 0
        elif r0_estimate[0] > vals[0]:
            return vals[0]
        else:
            return r0_estimate[0]

    @cache
    def performance_curve(
        self,
        pop,
        diameter,
        average_distance,
        cs,
        lbs,
        vals,
        adoption_fraction,
        constraint=10.0,
    ):
        number_of_trips = pop * 3.2 * adoption_fraction
        request_rate = number_of_trips / (17 * 60)

        target_loads = np.append(
            np.linspace(0.1, 2.0, 50),
            np.append(np.linspace(2.1, 5, 20), np.linspace(5.5, 15, 10)),
        )
        rejections = np.zeros_like(target_loads) * np.nan
        actual_loads = np.zeros_like(target_loads) * np.nan
        relative_driven_distances = np.zeros_like(target_loads) * np.nan
        fleets = np.zeros_like(target_loads) * np.nan
        for i, qi in enumerate(target_loads):
            actual_load = qi
            fleet_size = max([request_rate / qi * average_distance,1])
            r0 = np.min(
                [
                    np.max(
                        [
                            self.interpolate_r0(
                                cs, lbs, vals, average_distance, fleet_size, constraint
                            ),
                            0,
                        ]
                    ),
                    1.0,
                ]
            )
            rejection_rate = None
            if self.use_sklearn:
                rejection_rate = self.model.predict(
                    np.array(
                        [
                            [
                                fleet_size,
                                actual_load,
                                average_distance,
                                diameter,
                                constraint,
                                r0,
                            ]
                        ]
                    )
                ).reshape(-1)[0]
            else:
                qc = self.critical_load_model(fleet_size, constraint, diameter)
                D = self.noise_model(fleet_size, actual_load)
                rejection_rate = r_mean(actual_load, r0, qc, D, a=0.0)[0]
            rejections[i] = rejection_rate
            actual_loads[i] = actual_load
            fleets[i] = fleet_size
            if fleet_size <= 1.0:
                for j in range(i+1,len(target_loads)):
                    rejections[j] = rejection_rate
                    actual_loads[j] = actual_load
                    fleets[j] = fleet_size
                break
        """
        Notes on relative driven Distance: The naive formulation is 1/q + r. But at low loads, there is no sharing, and the 1/q part
        is not true anymore. Here the busses drive extra distance, given by the pickup distance. Hence we cut the value here with an estimate
        of the average distance to the nearest bus.
        """
        # taxi_service_distance = 1 + average_distance ** (-fleets+1)
        taxi_service_distance = 1 + fleets**(-1/2) #(average_distance ** (-(fleets)+1)
        taxi_indicator = int(
            np.arange(len(actual_loads))[1 / actual_loads < taxi_service_distance][0]
        ) if len(np.arange(len(actual_loads))[1 / actual_loads < taxi_service_distance])>0 else 0 # At what point does the system behave like a taxi?
        relative_driven_distance = (
            np.min(np.stack([1 / actual_loads, taxi_service_distance], axis=1), axis=1)
            + rejections
            - 1
        )
        relative_driven_distance[1:-1] = (
            relative_driven_distance[:-2]
            + relative_driven_distance[1:-1]
            + relative_driven_distance[2:]
        ) / 3

        plotting_data = np.zeros(shape=(len(target_loads) + 1, 6))
        plotting_data[1:, 0] = actual_loads
        plotting_data[1:, 1] = rejections
        plotting_data[1:, 2] = 1 - rejections
        plotting_data[1:, 3] = relative_driven_distance
        plotting_data[1:, 4] = fleets

        plotting_data[0, 0] = 0
        plotting_data[0, 1] = 0
        plotting_data[0, 2] = 1
        plotting_data[0, 3] = 0
        plotting_data[0, 4] = pop / 2
        plotting_data[1:, 5] = (1 / actual_loads < taxi_service_distance).astype(
            int
        )
        plotting_data[0, 5] = 0
        return plotting_data


class RidepoolingCosts:
    def __init__(self, modelfile: str, gamma_T=0.098, gamma_L=0.525):
        """
        Takes in the coefficients of the costs. Coefficients have the units cost per minute.
        Cost function has the shape:
        $$\tilde{C}_\mathrm{total} = \gamma_T \alpha c + \gamma_L \frac{B}{\lambda} + \gamma_T \tau_R r $$
        """
        self.Oracle = PerformanceOracle(pfile=modelfile, use_sklearn=True)
        pass

    @staticmethod
    def linear_utility_function(r, rel_driv_dist, c, ell, gamma_T, gamma_L):
        tau_r = c + ell
        vel = 1.0  # 50 km/h
        return gamma_T * c + gamma_L * ell * rel_driv_dist + gamma_T * tau_r * r

    def create_surface(self, row, adoption_rate=0.01):
        lbs = []
        cs = []
        vals = []
        d = row["r0_values"]
        for key1 in d:
            valrow = []
            for key2 in d[key1]:
                cs.append(float(key1) / row["average_distance"])
                lbs.append(np.log2(float(key2)))
                vals.append(float(d[key1][key2]))
        lbs = tuple(lbs)
        cs = tuple(cs)
        vals = tuple(vals)

        curves = []
        constraints = np.arange(1, 61, 1)
        for c in constraints:
            curve = self.Oracle.performance_curve(
                row["bev_21"],
                row["diameter"],
                row["average_distance"],
                cs,
                lbs,
                vals,
                adoption_rate,
                constraint=c,
            )

            curves.append(curve)
        Xs = np.zeros((60, 81))
        Ys = np.zeros((60, 81))
        Zs = np.zeros((60, 81))
        isRPs = np.zeros((60, 81))

        for i in range(len(curves)):
            Xs[i, :] = curves[i][:, 3]
            for j in range(len(curves[i])):
                Ys[:, j] = constraints
                Zs[i, j] = curves[i][j][1]
                isRPs[i, j] = curves[i][j][5]
        return Xs, Ys, Zs, isRPs

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
