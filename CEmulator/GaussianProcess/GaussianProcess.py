from ast import Add
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
import math
from scipy.special import gamma, kv
import warnings


GPR_CHOLESKY_LOWER = True

'''
kernel function used to calculate the covariance matrix.
'''
def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale

# adapted from scipy/optimize/optimize.py for functions with 2d output
def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad

class Kernel:
    '''
    Base class for kernel functions.
    '''
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Add(self, Constant(b))
        return Add(self, b)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Mul(self, Constant(b))
        return Mul(self, b)

class KernelOperator(Kernel):
    """
    Base class for all kernel operators.
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

class Add(KernelOperator):
    '''
    Add two kernels.
    '''
    def __call__(self, x1, x2=None):
        return self.k1(x1, x2) + self.k2(x1, x2)
class Mul(KernelOperator):
    '''
    Multiply two kernels.
    '''
    def __call__(self, x1, x2=None):
        return self.k1(x1, x2) * self.k2(x1, x2)

class Constant(Kernel):
    '''
    Constant kernel.
    '''
    def __init__(self, constant_value):
        self.constant_value = constant_value
    def __call__(self, x1, x2=None):
        return self.constant_value
    
class RBF(Kernel):
    '''
    Squared Exponential kernel
    '''
    def __init__(self, length_scale):
        self.length_scale = length_scale
    def __call__(self, x1, x2=None):
        ## check array shape
        x1 = np.atleast_2d(x1)
        length_scale     = _check_length_scale(x1,self.length_scale)
        if x2 is None:
            dists = pdist(x1 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(x1 / length_scale, x2 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
        return K

class Matern(RBF):
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)



    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                divide_result = np.zeros_like(D)
                np.divide(
                    D,
                    denominator,
                    out=divide_result,
                    where=denominator != 0,
                )
                K_gradient = K[..., np.newaxis] * divide_result
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

class GaussianProcessRegressor:
    '''
    Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    Modified from scikit-learn GaussianProcessRegressor.
    '''
    
    def __init__(self, X, y, kernel, alpha=1e-10, normalize_y=True):
        '''
        X: array-like, shape = (n_samples, n_features)
        Training data consisting of numeric features.
        y: array-like, shape = (n_samples,) here only support 1D y.
        kernel: kernel object to calculate the covariance matrix.
        alpha: float or array-like, shape = (n_samples,) 
        same as alpha in scikit-learn. Value added to the diagonal of the kernel matrix during fitting. 
        This can prevent a potential numerical issue during fitting, by ensuring that the calculated values form a positive definite matrix.
        In scikit-learn, the default value is 1e-10.
        normalize_y: bool, default=True
        '''
        # Initialize the GaussianProcess class
        if normalize_y:
            self._y_train_mean = np.mean(y)
            self._y_train_std  = np.std(y)
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = 0
            self._y_train_std = 1
        self.X_train = X
        self.y_train = y
        self.kernel = kernel
        self.alpha = alpha
        Kmatrix = self.kernel(X, X) # cov of training data
        Kmatrix[np.diag_indices_from(Kmatrix)] += self.alpha # K + sigma_n^2 I
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        try:
            self.L_ = cholesky(Kmatrix, 
                               lower=GPR_CHOLESKY_LOWER, 
                               check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train,
            check_finite=False,
        )
        
    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X is expected to have {self.X_train.shape[1]} features, "
                f"but has {X.shape[1]}."
            )
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        Kstar = self.kernel(X, self.X_train)
        y_mean = Kstar @ self.alpha_
        # Add the mean of the training data and scale back to the original scale
        y_mean = y_mean * self._y_train_std + self._y_train_mean
        if   return_cov:
            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, Kstar.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )
            # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
            y_cov = self.kernel_(X) - V.T @ V

            # undo normalisation
            y_cov = np.outer(y_cov, self._y_train_std**2).reshape(
                *y_cov.shape, -1
            )
            # if y_cov has shape (n_samples, n_samples, 1), reshape to
            # (n_samples, n_samples)
            if y_cov.shape[2] == 1:
                y_cov = np.squeeze(y_cov, axis=2)
            return y_mean, y_cov
        elif return_std:
            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, Kstar.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )
            # Compute variance of predictive distribution
            # Use einsum to avoid explicitly forming the large matrix
            # V^T @ V just to extract its diagonal afterward.
            y_var = self.kernel.diag(X).copy()
            y_var -= np.einsum("ij,ji->i", V.T, V)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. "
                    "Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0

            # undo normalisation
            y_var = np.outer(y_var, self._y_train_std**2).reshape(
                *y_var.shape, -1
            )

            # if y_var has shape (n_samples, 1), reshape to (n_samples,)
            if y_var.shape[1] == 1:
                y_var = np.squeeze(y_var, axis=1)
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean


