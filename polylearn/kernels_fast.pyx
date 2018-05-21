# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from lightning.impl.dataset_fast import get_dataset
from lightning.impl.dataset_fast cimport RowDataset, ColumnDataset
import numpy as np
cimport numpy as np


cdef void _all_subsets_fast(double[:, ::1] output,
                            RowDataset X,
                            ColumnDataset P):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t X_samples, P_samples, i1, ii2, i2, jj, j
    X_samples = X.get_n_samples()
    P_samples = P.get_n_samples()

    for i1 in range(X_samples):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                output[i1, i2] *= (1 + x[jj]*p[ii2])


def all_subsets_kernel(X, P):
    output = np.ones((X.shape[0], P.shape[0]))
    _all_subsets_fast(output,
                      get_dataset(X, order='c'),
                      get_dataset(P, order='fortran'))
    return output