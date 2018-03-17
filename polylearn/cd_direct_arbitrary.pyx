# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#

from cython.view cimport array
from lightning.impl.dataset_fast cimport ColumnDataset
from libc.math cimport fabs
from .cd_linear_fast cimport _cd_linear_epoch
from .loss_fast cimport LossFunction


cdef inline void _grad_anova(double[:] dA,
                             double[:, ::1] A,
                             double x_ij,
                             double p_js,
                             unsigned int degree,
                             Py_ssize_t i):
    cdef Py_ssize_t t

    dA[0] = x_ij
    for t in range(1, degree):
        dA[t] = x_ij * (A[t, i] - p_js * dA[t-1])


cdef void _precompute_A_all_degree(ColumnDataset X,
                                   double[:, :, ::1] P,
                                   Py_ssize_t order,
                                   double[:, ::1] A,
                                   Py_ssize_t s,
                                   unsigned int degree):
    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, j, ii, t
    cdef unsigned int deg

    for t in range(1, degree+1):
        for i in range(n_samples):
            A[t, i] = 0

    # calc {1, \ldots, degree}-order anova kernels for all data
    # A[m, i] = m-order anova kernel for i-th data
    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)
        for t in range(degree):
            for ii in range(n_nz):
                i = indices[ii]
                A[degree-t, i] += A[degree-t-1, i] * P[order, s, j] * data[ii]


cdef inline double _update(int* indices,
                           double* data,
                           int n_nz,
                           double p_js,
                           double[:] y,
                           double[:] y_pred,
                           LossFunction loss,
                           double lam,
                           unsigned int degree,
                           double beta,
                           double[:, ::1] A,
                           double[:] dA,
                           unsigned int denominator):
    cdef double l1_reg = 2 * beta * fabs(lam)
    cdef Py_ssize_t i, ii
    cdef double inv_step_size = 0
    cdef double update = 0

    for ii in range(n_nz):
        i = indices[ii]
        _grad_anova(dA, A, data[ii], p_js, degree, indices[ii])
        update += loss.dloss(y_pred[i], y[i]) * dA[degree-1]
        inv_step_size += dA[degree-1] ** 2

    inv_step_size *= loss.mu * (lam ** 2) / denominator
    inv_step_size += l1_reg

    update *= lam
    update /= denominator
    update += l1_reg * p_js
    update /= inv_step_size

    return update


cdef inline double _cd_epoch(double[:, :, ::1] P,
                             Py_ssize_t order,
                             ColumnDataset X,
                             double[:] y,
                             double[:] y_pred,
                             double[:] lams,
                             unsigned int degree,
                             double beta,
                             LossFunction loss,
                             double[:, ::1] A,
                             double[:] dA,
                             unsigned int denominator):
    cdef Py_ssize_t s, j, ii, i
    cdef double p_old, update, offset
    cdef double sum_viol = 0
    cdef Py_ssize_t n_components = P.shape[1]
    cdef Py_ssize_t n_features = P.shape[2]
    cdef unsigned int deg

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Update P_{s} \forall s \in [n_components] for A^{degree}
    # P_{s} for A^{degree} = P[order, s]
    for s in range(n_components):
        # initialize the cached ds for this s
        _precompute_A_all_degree(X, P, order, A, s, degree)

        for j in range(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)
            # Compute coordinate update
            p_old = P[order, s, j]
            update = _update(indices, data, n_nz, p_old, y, y_pred, loss,
                             lams[s], degree, beta, A, dA, denominator)
            sum_viol += fabs(update)
            P[order, s, j] -= update

            # Synchronize predictions and A
            for ii in range(n_nz):
                i = indices[ii]
                dA[0] = data[ii]
                for deg in range(1, degree):
                    dA[deg] = data[ii] * (A[deg, i] - p_old * dA[deg-1])
                    A[deg, i] -= update * dA[deg-1]

                A[degree, i] -= update * dA[degree-1]
                y_pred[i] -= lams[s] * update * dA[degree-1]

    return sum_viol


def _cd_direct_arbitrary(double[:, :, ::1] P not None,
                         double[:] w not None,
                         ColumnDataset X,
                         double[:] col_norm_sq not None,
                         double[:] y not None,
                         double[:] y_pred not None,
                         double[:] lams not None,
                         unsigned int degree,
                         double alpha,
                         double beta,
                         bint fit_linear,
                         bint fit_lower,
                         LossFunction loss,
                         Py_ssize_t max_iter,
                         double tol,
                         int verbose,
                         bint mean):

    cdef Py_ssize_t n_samples, i
    cdef unsigned int it, deg, denominator
    cdef double viol
    cdef bint converged = False
    n_samples = X.get_n_samples()

    # precomputed values
    # A[m, i] = A^{m}(p, X_i)
    cdef double[:, ::1] A = array((degree+1, n_samples), sizeof(double), 'd')
    cdef double[:] dA = array((degree, ), sizeof(double), 'd')

    # Which is loss function, mean or sum?
    if mean:
        denominator = n_samples
    else:
        denominator = 1

    # init anova kernels
    for i in range(n_samples):
        A[0, i] = 1

    it = 0
    for it in range(max_iter):
        viol = 0
        if fit_linear:
            viol += _cd_linear_epoch(w, X, y, y_pred, col_norm_sq,
                                     alpha, loss, denominator)

        if fit_lower:
            for deg in range(2, degree):
                viol += _cd_epoch(P, degree-deg, X, y, y_pred, lams, deg,
                                  beta, loss, A, dA, denominator)

        viol += _cd_epoch(P, 0, X, y, y_pred, lams, degree,
                          beta, loss, A, dA, denominator)

        if verbose:
            print("Iteration {} violation sum {}".format(it + 1, viol))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it + 1))
            converged = True
            break

    return converged, it
