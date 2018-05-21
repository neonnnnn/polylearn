# encoding: utf-8

# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.externals import six

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        pass

from lightning.impl.dataset_fast import get_dataset

from .base import _BasePoly, _PolyClassifierMixin, _PolyRegressorMixin
from .kernels import _poly_predict
from .cd_direct_fast_as import _cd_direct_as


class _BaseAllSubsets(six.with_metaclass(ABCMeta, _BasePoly)):

    @abstractmethod
    def __init__(self, loss='squared', n_components=2, beta=1,
                 mean=False, tol=1e-6, warm_start=False,
                 init_lambdas='ones', max_iter=10000, verbose=False,
                 callback=None, n_calls=100, random_state=None):
        self.loss = loss
        self.n_components = n_components
        self.beta = beta
        self.mean = mean
        self.tol = tol
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.verbose = verbose
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state

    def fit(self, X, y):
        """Fit all-subsets model to training data.

        Parameters
        ----------
        X : array-like or sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : Estimator
            Returns self.
        """

        X, y = self._check_X_y(X, y)
        n_features = X.shape[1]
        dataset = get_dataset(X, order="fortran")
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)

        if not (self.warm_start and hasattr(self, 'P_')):
            self.P_ = 0.01 * rng.randn(self.n_components, n_features)

        if not (self.warm_start and hasattr(self, 'lams_')):
            if self.init_lambdas == 'ones':
                self.lams_ = np.ones(self.n_components)
            elif self.init_lambdas == 'random_signs':
                self.lams_ = np.sign(rng.randn(self.n_components))
            else:
                raise ValueError("Lambdas must be initialized as ones "
                                 "(init_lambdas='ones') or as random "
                                 "+/- 1 (init_lambdas='random_signs').")

        y_pred = self._get_output(X)

        converged, self.n_iter_ = _cd_direct_as(
            self.P_, dataset, y, y_pred, self.lams_, self.beta,
            loss_obj, self.max_iter, self.mean, self.tol, self.verbose,
            self.callback, self.n_calls)
        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, X):
        y_pred = _poly_predict(X, self.P_[:, :], self.lams_, 'all-subsets')

        return y_pred

    def _predict(self, X):
        if not hasattr(self, "P_"):
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse='csc', dtype=np.double)
        return self._get_output(X)


class AllSubsetsRegressor(_BaseAllSubsets, _PolyRegressorMixin):
    """All-subsets model for regression (with squared loss).

    Parameters
    ----------
    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    beta : float, default: 1
        Regularization amount for higher-order weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_components, n_features]
        The learned basis functions.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Higher-Order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata.
    In: Proceedings of NIPS 2016.
    https://arxiv.org/abs/1607.07195
    """
    def __init__(self, n_components=2, beta=1, mean=False, tol=1e-6,
                 warm_start=False, init_lambdas='ones', max_iter=10000,
                 verbose=False, callback=None, n_calls=100, random_state=None):

        super(AllSubsetsRegressor, self).__init__(
            'squared', n_components, beta, mean, tol, warm_start,
            init_lambdas, max_iter, verbose, callback, n_calls, random_state)


class AllSubsetsClassifier(_BaseAllSubsets, _PolyClassifierMixin):
    """All-subsets model for classification.

    Parameters
    ----------

    loss : {'logistic'|'squared_hinge'|'squared'}, default: 'squared_hinge'
        Which loss function to use.

        - logistic: L(y, p) = log(1 + exp(-yp))

        - squared hinge: L(y, p) = max(1 - yp, 0)²

        - squared: L(y, p) = 0.5 * (y - p)²

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    beta : float, default: 1
        Regularization amount for feature combinations weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_components, n_features]
        The learned basis functions.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Higher-Order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata.
    In: Proceedings of NIPS 2016.
    https://arxiv.org/abs/1607.07195
    """

    def __init__(self, loss='squared_hinge', n_components=2, beta=1, mean=False,
                 tol=1e-6, warm_start=False, init_lambdas='ones',
                 max_iter=10000, verbose=False, callback=None, n_calls=100,
                 random_state=None):

        super(AllSubsetsClassifier, self).__init__(
            loss, n_components, beta, mean, tol, warm_start, init_lambdas,
            max_iter, verbose, callback, n_calls, random_state)
