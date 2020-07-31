# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def gt_bytes2vec(bytes gt_bytes):
    """
    Convert an ascii string of diploid GT fields into an integer numpy array.
    Assumes well-formed input. I.e. ``gt_bytes`` should look like:
        b"0/1\t0/0\t/0/1\t./.\n".
    '0' is converted to 0, '1' to 1, and '.' to 2.
    Returns a two-tuple (gt_vec, ac), where ``gt_vec`` is the genotype array,
    and ``ac`` is a list of the allele counts.
    """
    cdef unsigned int n = len(gt_bytes)
    assert n % 4 == 0
    cdef char * gt_str = gt_bytes
    cdef np.ndarray gt_vec = np.empty(n // 2, dtype=np.int8)
    cdef char [:] mem = gt_vec
    cdef int a
    cdef int ac[3]
    cdef int i
    ac[0] = ac[1] = ac[2] = 0
    for i in range(n // 2):
        # The ascii values of '.', '0', '1' are 46, 48, and 49 respectively.
        a = (gt_str[2*i] - 46) ^ 2
        ac[a] += 1
        mem[i] = a
    return gt_vec, ac
