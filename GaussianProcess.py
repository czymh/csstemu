from ast import Add
from matplotlib.font_manager import X11FontDirectories
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
import warnings


GPR_CHOLESKY_LOWER = True

'''
kernel function used to calculate the covariance matrix.
'''
def _check_length_scale(length_scale):
    if length_scale.ndim > 1:
        raise ValueError('length_scale must be a 1D array')
    return length_scale

class Kernel:
    '''
    Base class for kernel functions.
    '''
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Add(self, czConstant(b))
        return Add(self, b)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Mul(self, czConstant(b))
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

class czConstant(Kernel):
    '''
    Constant kernel.
    '''
    def __init__(self, constant_value):
        self.constant_value = constant_value
    def __call__(self, x1, x2=None):
        return self.constant_value
    
class czRBF(Kernel):
    '''
    Squared Exponential kernel
    '''
    def __init__(self, length_scale):
        self.length_scale = length_scale
    def __call__(self, x1, x2=None):
        ## check array shape
        length_scale     = _check_length_scale(self.length_scale)
        x1 = np.atleast_2d(x1)
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


