# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Kyohei Atarashi
# License: BSD

from libc.math cimport fabs
from cython.view cimport array
from lightning.impl.dataset_fast cimport ColumnDataset
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np


cdef void _precompute(ColumnDataset X,
                      double[:, ::1] P,
                      double[:] out,
                      Py_ssize_t s):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = P.shape[1]

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, j, ii
    cdef unsigned int d
    cdef double tmp

    for i in range(n_samples):
        out[i] = 1

    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            out[i] *= (1. + data[ii] * P[s, j])


cdef inline double _update(int* indices,
                           double* data,
                           int n_nz,
                           double p_js,
                           double[:] y,
                           double[:] y_pred,
                           LossFunction loss,
                           double lam,
                           double beta,
                           double[:] D,
                           double[:] cache_kp):

    cdef double l1_reg = 2 * beta * fabs(lam)

    cdef Py_ssize_t i, ii

    cdef double inv_step_size = 0

    cdef double kp  # derivative of the all-subsets kernel
    cdef double update = 0

    for ii in range(n_nz):
        i = indices[ii]

        kp = D[i] * data[ii] / (1 + p_js * data[ii])
        cache_kp[ii] = kp
        kp *= lam
        update += loss.dloss(y_pred[i], y[i]) * kp
        inv_step_size += kp ** 2

    inv_step_size *= loss.mu
    inv_step_size += l1_reg
    update += l1_reg * p_js
    update /= inv_step_size

    return update


cdef inline double _cd_direct_epoch(double[:, ::1] P,
                                    ColumnDataset X,
                                    double[:] y,
                                    double[:] y_pred,
                                    double[:] lams,
                                    double beta,
                                    LossFunction loss,
                                    double[:] D,
                                    double[:] cache_kp,
                                    int[:] indices_features,
                                    bint shuffle,
                                    rng):
    cdef Py_ssize_t s, j, jj
    cdef double p_old, update, offset
    cdef double sum_viol = 0
    cdef Py_ssize_t n_components = P.shape[0]
    cdef Py_ssize_t n_features = P.shape[1]

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for s in range(n_components):

        # initialize the cached ds for this s
        _precompute(X, P, D, s)
        if shuffle:
            rng.shuffle(indices_features)
        for jj in range(n_features):
            j = indices_features[jj]
            X.get_column_ptr(j, &indices, &data, &n_nz)

            # compute coordinate update
            p_old = P[s, j]
            update = _update(indices, data, n_nz, p_old, y, y_pred,
                             loss, lams[s], beta, D, cache_kp)
            P[s, j] -= update
            sum_viol += fabs(update)

            # Synchronize predictions and ds
            for ii in range(n_nz):
                i = indices[ii]
                D[i] -= update * cache_kp[ii]
                y_pred[i] -= lams[s] * update * cache_kp[ii]
    return sum_viol


def _cd_direct_as(self,
                  double[:, ::1] P not None,
                  ColumnDataset X,
                  double[:] y not None,
                  double[:] y_pred not None,
                  double[:] lams not None,
                  double beta,
                  LossFunction loss,
                  unsigned int max_iter,
                  double tol,
                  int verbose,
                  callback,
                  unsigned int n_calls,
                  bint shuffle,
                  rng):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()
    cdef unsigned int it

    cdef double viol
    cdef bint converged = False
    cdef bint has_callback = callback is not None

    # precomputed values
    cdef double[:] D = array((n_samples, ), sizeof(double), 'd')
    cdef double[:] cache_kp = array((n_samples,), sizeof(double), 'd')
    
    # for random sampling of feature index
    cdef np.ndarray[int, ndim=1] indices_features
    indices_features = np.arange(n_features, dtype=np.int32)

    it = 0
    for it in range(max_iter):
        viol = 0

        viol += _cd_direct_epoch(P, X, y, y_pred, lams, beta, loss, D,
                                 cache_kp, indices_features, shuffle, rng)

        if has_callback and it % n_calls == 0:
            if callback(self) is not None:
                break

        if verbose:
            print("Iteration", it + 1, "violation sum", viol)

        if viol < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            converged = True
            break

    return converged, it