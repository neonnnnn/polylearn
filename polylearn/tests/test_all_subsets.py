# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings

from nose.tools import assert_less_equal, assert_equal

import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_warns_message

from polylearn.kernels import _poly_predict
from polylearn import AllSubsetsRegressor
from polylearn import AllSubsetsClassifier


def cd_direct_slow(X, y, lams=None, n_components=5, beta=1., n_iter=10,
                   tol=1e-5, verbose=False, random_state=None, mean=False):
    from sklearn.utils import check_random_state
    from polylearn.kernels_fast import all_subsets_kernel

    n_samples, n_features = X.shape

    if mean:
        denominator = n_samples
    else:
        denominator = 1

    rng = check_random_state(random_state)
    P = 0.01 * rng.randn(n_components, n_features)
    if lams is None:
        lams = np.ones(n_components)

    K = all_subsets_kernel(X, P)
    pred = np.dot(lams, K.T)

    mu = 1  # squared loss
    converged = False

    for i in range(n_iter):
        sum_viol = 0
        for s in range(n_components):
            ps = P[s]
            for j in range(n_features):

                # trivial approach:
                # multilinearity allows us to isolate the term with ps_j * x_j
                x = X[:, j]

                grad_y = lams[s] * x * K[:, s] / (1 + ps[j] * x)
                l1_reg = 2 * beta * np.abs(lams[s])
                inv_step_size = mu * (grad_y ** 2).sum()/denominator + l1_reg

                dloss = pred - y  # squared loss
                step = (dloss * grad_y).sum()/denominator + l1_reg * ps[j]
                step /= inv_step_size

                P[s, j] -= step
                sum_viol += np.abs(step)

                # stupidly recompute all predictions. No rush yet.
                K = all_subsets_kernel(X, P)
                pred = np.dot(K, lams)

        reg_obj = beta * np.sum((P ** 2).sum(axis=1) * np.abs(lams))

        if verbose:
            print("Epoch", i, "violations", sum_viol, "obj",
                  0.5 * ((pred - y) ** 2).sum() + reg_obj)

        if sum_viol < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return P


n_components = 3
n_features = 7
n_samples = 20

rng = np.random.RandomState(1)

X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)

lams = rng.randn(n_components)


def check_fit():
    y = _poly_predict(X, P, lams, kernel="all-subsets")

    est = AllSubsetsRegressor(n_components=n_components, max_iter=150000,
                              beta=1e-10, tol=1e-6, random_state=0,
                              init_lambdas='random_signs')
    est.fit(X, y)
    y_pred = est.predict(X)
    err = mean_squared_error(y, y_pred)

    assert_less_equal(
        err,
        1e-6,
        msg="Error {} too big.".format(err))


def test_fit():
    yield check_fit


def check_improve():
    y = _poly_predict(X, P, lams, kernel="all-subsets")

    est = AllSubsetsRegressor(n_components=n_components, beta=0.0001,
                              max_iter=5, tol=0,
                              random_state=0, init_lambdas='random_signs')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_5 = est.fit(X, y).predict(X)
        est.set_params(max_iter=10)
        y_pred_10 = est.fit(X, y).predict(X)

    assert_less_equal(mean_squared_error(y, y_pred_10),
                      mean_squared_error(y, y_pred_5),
                      msg="More iterations do not improve fit.")


def test_improve():
    yield check_improve


def check_overfit():
    noisy_y = _poly_predict(X, P, lams, kernel="all-subsets")
    noisy_y += 5. * rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    # weak regularization, should overfit
    est = AllSubsetsRegressor(n_components=n_components, beta=1e-4, tol=0.01,
                              random_state=0, init_lambdas='random_signs')
    y_train_pred_weak = est.fit(X_train, y_train).predict(X_train)
    y_test_pred_weak = est.predict(X_test)

    est.set_params(beta=10)  # high value of beta -> strong regularization
    y_train_pred_strong = est.fit(X_train, y_train).predict(X_train)
    y_test_pred_strong = est.predict(X_test)

    assert_less_equal(mean_squared_error(y_train, y_train_pred_weak),
                      mean_squared_error(y_train, y_train_pred_strong),
                      msg="Training error does not get worse with regul.")

    assert_less_equal(mean_squared_error(y_test, y_test_pred_strong),
                      mean_squared_error(y_test, y_test_pred_weak),
                      msg="Test error does not get better with regul.")


def test_overfit():
    yield check_overfit


def test_convergence_warning():
    y = _poly_predict(X, P, lams, kernel="all-subsets")

    est = AllSubsetsRegressor(beta=1e-8, max_iter=1, random_state=0)
    assert_warns_message(UserWarning, "converge", est.fit, X, y)


def test_random_starts():
    noisy_y = _poly_predict(X, P, lams, kernel="all-subsets")
    noisy_y += 5. * rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    scores = []
    # init_lambdas='ones' is important to reduce variance here
    reg = AllSubsetsRegressor(n_components=n_components,
                              beta=5, max_iter=20000,
                              init_lambdas='ones', tol=0.001,)
    for k in range(10):
        reg.set_params(random_state=k)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))

    assert_less_equal(np.std(scores), 0.001)


def check_same_as_slow(mean):
    y = _poly_predict(X, P, lams, kernel="all-subsets")

    reg = AllSubsetsRegressor(n_components=n_components,
                              beta=1, warm_start=False, tol=1e-3,
                              max_iter=5, random_state=0, mean=mean,
                              init_lambdas='random_signs')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)

        P_fit_slow = cd_direct_slow(X, y, lams=reg.lams_,
                                    n_components=n_components, beta=1, n_iter=5,
                                    tol=1e-3, random_state=0, mean=mean)

    assert_array_almost_equal(reg.P_[:, :], P_fit_slow, decimal=4)


def test_same_as_slow():
    yield check_same_as_slow, True
    yield check_same_as_slow, False


def check_classification_losses(loss):
    y = np.sign(_poly_predict(X, P, lams, kernel="all-subsets"))
    clf = AllSubsetsClassifier(loss=loss, beta=1e-6, tol=1e-6,
                               random_state=0, init_lambdas='random_signs')
    clf.fit(X, y)
    assert_equal(1.0, clf.score(X, y))


def test_classification_losses():
    for loss in ('squared_hinge', 'logistic'):
        yield check_classification_losses, loss


def check_warm_start():
    y = _poly_predict(X, P, lams, kernel="all-subsets")
    # Result should be the same if:
    # (a) running 10 iterations
    clf_10 = AllSubsetsRegressor(n_components=n_components, max_iter=10,
                                 warm_start=False, random_state=0,
                                 init_lambdas='random_signs')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_10.fit(X, y)

    # (b) running 5 iterations and 5 more
    clf_5_5 = AllSubsetsRegressor(n_components=n_components,
                                  max_iter=5, warm_start=True,
                                  random_state=0, init_lambdas='random_signs')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5_5.fit(X, y)
        P_fit = clf_5_5.P_.copy()
        lams_fit = clf_5_5.lams_.copy()
        clf_5_5.fit(X, y)

    # (c) running 5 iterations when starting from previous point.
    clf_5 = AllSubsetsRegressor(n_components=n_components, max_iter=5,
                                warm_start=True, random_state=0,
                                init_lambdas='random_signs')
    clf_5.P_ = P_fit
    clf_5.lams_ = lams_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5.fit(X, y)

    assert_array_almost_equal(clf_10.P_, clf_5_5.P_)
    assert_array_almost_equal(clf_10.P_, clf_5.P_)

    # Prediction results should also be the same if:
    # (note: could not get this test to work for the exact P_.)

    noisy_y = _poly_predict(X, P, lams, kernel="all-subsets")
    noisy_y += rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    beta_low = 0.50
    beta = 0.50
    beta_hi = 0.50
    ref = AllSubsetsRegressor(n_components=n_components, beta=beta,
                              max_iter=20000, random_state=0,
                              init_lambdas='ones')
    ref.fit(X_train, y_train)
    y_pred_ref = ref.predict(X_test)

    # (a) starting from lower beta, increasing and refitting
    from_low = AllSubsetsRegressor(n_components=n_components, beta=beta_low,
                                   max_iter=20000, warm_start=True,
                                   random_state=0, init_lambdas='ones')
    from_low.fit(X_train, y_train)
    from_low.set_params(beta=beta)
    from_low.fit(X_train, y_train)
    y_pred_low = from_low.predict(X_test)

    # (b) starting from higher beta, decreasing and refitting
    from_hi = AllSubsetsRegressor(n_components=n_components, beta=beta_hi,
                                  max_iter=20000, warm_start=True,
                                  random_state=0, init_lambdas='ones')
    from_hi.fit(X_train, y_train)
    from_hi.set_params(beta=beta)
    from_hi.fit(X_train, y_train)
    y_pred_hi = from_hi.predict(X_test)

    assert_array_almost_equal(y_pred_low, y_pred_ref, decimal=3)
    assert_array_almost_equal(y_pred_hi, y_pred_ref, decimal=3)


def test_warm_start():
    yield check_warm_start
