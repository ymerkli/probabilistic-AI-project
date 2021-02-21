import math
import numpy as np
import GPy
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

import warnings
warnings.filterwarnings("ignore")

np.random.seed(7)
domain = np.array([[0, 5]])


""" Solution """

class BO_algo():
    def __init__(self, gpy_impl=False):
        """Initializes the algorithm with a parameter configuration. """
        # constants
        self.v_min = 1.2
        self.logv_min = math.log(self.v_min)
        self.gpy_impl = gpy_impl

        # data holders
        self.x_sample = np.array([]).reshape(-1, domain.shape[0])
        self.f_sample = np.array([]).reshape(-1, domain.shape[0])
        self.v_sample = np.array([]).reshape(-1, domain.shape[0])
        self.logv_sample = np.array([]).reshape(-1, domain.shape[0])
        self.gv_sample = np.array([]).reshape(-1, domain.shape[0])

        # incorporate prior beliefs about f() and v()
        self.f_sigma = 0.15
        self.f_variance = 0.5
        self.f_lengthscale = 0.5
        self.f_kernel = Matern(length_scale=0.5, nu=2.5)
        self.f_gpr = GaussianProcessRegressor(
            kernel=self.f_kernel,
            alpha=self.f_sigma**2
        )
        if self.gpy_impl:
            self.f_kernel = GPy.kern.Matern52(
                input_dim=domain.shape[0], variance=self.f_variance,
                lengthscale=self.f_lengthscale
            )
            self.f_gpr = None

        self.v_sigma = 0.0001
        self.v_variance = math.sqrt(2)
        self.v_lengthscale = 0.5
        self.v_const = 1.5
        self.v_kernel = ConstantKernel(self.v_const) + Matern(length_scale=self.v_lengthscale, nu=2.5)
        self.v_gpr = GaussianProcessRegressor(
            kernel=self.v_kernel,
            alpha=self.v_sigma**2
        )
        if self.gpy_impl:
            self.v_kernel = GPy.kern.Matern52(
                input_dim=domain.shape[0], variance=self.v_variance,
                lengthscale=self.v_lengthscale
            ) + GPy.kern.Bias(
                input_dim=domain.shape[0], variance=self.v_const
            )
            self.v_gpr = None

        self.gv_const = self.v_const - self.v_min
        self.gv_sigma = 0.0001
        self.gv_kernel = ConstantKernel(self.gv_const) + Matern(length_scale=0.5, nu=2.5)
        self.gv_gpr = GaussianProcessRegressor(
            kernel=self.gv_kernel,
            alpha=self.gv_sigma**2
        )

        self.logv_const = math.log(self.v_const)
        self.logv_sigma = 0.0001
        self.logv_kernel = ConstantKernel(self.logv_const) + Matern(length_scale=self.v_lengthscale, nu=2.5)
        self.logv_gpr = GaussianProcessRegressor(
            kernel=self.logv_kernel,
            alpha=self.logv_sigma**2
        )

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if self.x_sample.size == 0:
            # if no point has been sampled yet, we can't optimize the acquisition function yet
            # we instead sample a random starting point in the domain
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            next_x = np.array([x0]).reshape(-1, domain.shape[0])
        else:
            if len(self.f_sample) == 12 and np.all(self.f_sample < 0.4):
                # if after 10 iterations we have not found a point with a accuracy larger than 0.3
                # we take a random point in the domain as next point
                x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
                x0 = (self.x_sample[0] + (domain[:, 1] - domain[:, 0])/2) % domain[:, 1]
                next_x = np.array([x0]).reshape(-1, domain.shape[0])
            else:
                next_x = self.optimize_acquisition_function()

        assert next_x.shape == (1, domain.shape[0])
        return next_x

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(30):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(
                func=objective, x0=x0, bounds=domain, approx_grad=True
            )

            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.
        Constrained acquisition function as proposed by https://arxiv.org/abs/1403.5607

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        ei = self.expected_improvement(x, xi=0.015)
        constraint_weight = self.constraint_function(x)

        return float(ei * constraint_weight)

    def expected_improvement(self, x, xi=0.01):
        """
        Compute expected improvement at points x based on samples x_samples
        and y_samples using Gaussian process surrogate

        Args:
            x: Points at which EI should be computed
            xi: Exploitation-exploration trade-off parameter
        """
        if self.gpy_impl:
            mu, sigma = self.f_gpr.predict(x.reshape(-1, domain.shape[0]))
            mu_sample = self.f_gpr.predict(self.x_sample)
        else:
            mu, sigma = self.f_gpr.predict([x], return_std=True)
            mu_sample = self.f_gpr.predict(self.x_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def constraint_function(self, x):
        """
        Model constraint condition v(theta) > v_min as a real-valued latent constraint function
        g_k(x) with g_k(theta) = v(theta) - v_min > 0 and then infer PR(g_k > 0) from its posterior

        Following: https://arxiv.org/abs/1403.5607
        """

        # predict distribution of speed v
        if self.gpy_impl:
            mu, sigma = self.v_gpr.predict(x.reshape(-1, domain.shape[0]))
        else:
            mu, sigma = self.v_gpr.predict([x], return_std=True)

        # Gaussian CDF with params from GPR prediction
        if sigma != 0:
            pr = 1 - norm.cdf(self.v_min, loc=mu, scale=sigma)
        else:
            pr = 0.98 * (mu - self.v_min) if mu >= self.v_min else 0.02 * (self.v_min - mu)

        return pr

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # stack the newly obtained data point onto the existing data points
        self.x_sample    = np.vstack((self.x_sample, x))
        self.f_sample    = np.vstack((self.f_sample, f))
        self.v_sample    = np.vstack((self.v_sample, v))
        #self.logv_sample = np.vstack((self.logv_sample, math.log(v)))
        self.gv_sample   = np.vstack((self.gv_sample, v - self.v_min))

        # add new datapoint to GPs and retrain
        if self.gpy_impl:
            self.f_gpr = GPy.models.gp_regression.GPRegression(
                X=self.x_sample, Y=self.f_sample,
                kernel=self.f_kernel, noise_var=self.f_sigma**2
            )
            self.v_gpr = GPy.models.gp_regression.GPRegression(
                X=self.x_sample, Y=self.v_sample,
                kernel=self.v_kernel, noise_var=self.v_sigma**2
            )
            self.v_gpr.optimize()
            self.f_gpr.optimize()
        else:
            self.f_gpr.fit(self.x_sample, self.f_sample)
            self.v_gpr.fit(self.x_sample, self.v_sample)
            #self.gv_gpr.fit(self.x_sample, self.gv_sample)
            #self.logv_gpr.fit(self.x_sample, self.logv_sample)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # select the highest accuracy sample from all valid samples (i.e. samples above the speed threshold)
        valid_samples = self.f_sample
        valid_samples[self.v_sample < 1.2] = -1e6 # heuristically low number
        best_index = np.argmax(valid_samples)     # get the index of highest accuracy
        x_opt = self.x_sample[best_index]         # get the corresponding x value

        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()