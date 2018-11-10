# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
Supervised Learning to Aggregate Data with the Sugeno Integral
by Marek Gagolewski, Simon James, and Gleb Beliakov

Exact branch-refine-and-bound-type algorithm
for fitting Sugeno integrals with respect to symmetric capacities
minimizing L1 error, particularly suitable for the case of ordinal data.

Main function: fit_Sugeno_L1_brnb()


Cython code Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np
cimport numpy as np
import queue

cdef np.double_t EPS = 1e-12



cdef inline np.double_t minf(np.double_t x, np.double_t y):
    if x <= y:   return x
    else:        return y



cdef inline np.double_t maxf(np.double_t x, np.double_t y):
    if x >= y:   return x
    else:        return y



cdef inline np.double_t absf(np.double_t x):
    if x >= 0.0: return  x
    else:        return -x



cpdef inline np.double_t Sugeno(np.double_t[:] x, np.double_t[:] h):
    """
    Compute the (symmetric) Sugeno integral S_h(x).
    It holds Sugeno(x, h) == np.max(np.minimum(x, h)).

    Arguments
    =========

    x - a decreasingly (weakly) ordered input sample

    h - an increasing (weakly) coefficients vector

    Both x and h should be of identical lengths.


    Returns
    =======

    a single numeric value
    """
    cdef np.uint_t n = x.shape[0], i
    assert h.shape[0] == n
    cdef np.double_t ret_val

    ret_val = minf(x[0], h[0])
    for i in range(1, n):
        # @TODO@ binsearch for large n
        assert h[i-1] <= h[i]
        assert x[i-1] >= x[i]
        if x[i] > h[i]:           # x[i]  > h[i]  =>  min(x[i], h[i]) = h[i]
            if ret_val < h[i]:
                ret_val = h[i]
        else:                     # x[i] <= h[i]  =>  min(x[i], h[i]) = x[i]
            if ret_val < x[i]:
                ret_val = x[i]
            # early stop now
            # x[i+1] <= x[i] <= h[i] <= h[i+1]
            # => min(x[i], h[i]) = x[i] >= min(x[i+1], h[i+1]) = x[i+1]
            break

    return ret_val


cpdef np.ndarray[np.double_t] Sugenos(np.double_t[:,:] x, np.double_t[:] h):
    """
    Given many inputs, compute the (symmetric) Sugeno integral S_h(x[k,:])
    for all k.

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    h - an increasing (weakly) coefficients vector of length n


    Returns
    =======

    a numeric vector of length m
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t i, k
    assert h.shape[0] == n
    cdef np.ndarray[np.double_t] y = np.empty(m)

    for k in range(m):
        y[k] = Sugeno(x[k,:], h)

    return y


cpdef np.double_t fitness_L1(np.double_t[:,:] x, np.double_t[:] y, np.double_t[:] h):
    """
    Computes the absolute error between the Sugeno integrals of given inputs w.r.t. a given
    coefficients vector and reference values, i.e., np.sum(np.abs(Sugenos(x, h)-y)).

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    y - a vector of m reference outputs

    h - an increasing (weakly) coefficients vector of length n


    Returns
    =======

    a numeric vector of length m
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t i, k
    assert h.shape[0] == n
    assert y.shape[0] == m
    cdef np.double_t err = 0.0
    cdef np.double_t val

    for k in range(m):
        val = Sugeno(x[k,:], h)
        err += absf(val-y[k])

    return err


cpdef void fit_Sugeno_L1_improve_lower_bound(np.double_t[:] h, np.double_t[:] h_stop, np.double_t[:,:] x, np.double_t[:] y, np.double_t[:] I):
    """
    Tries to improve h -- finds h_new such that h <= h_new <= h_stop

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    y - a vector of m reference outputs

    h, h_stop - an increasing (weakly) coefficients vector of length n, h[i]<=h_stop[i]

    I - result of np.unique(np.r_[x.ravel(),y.ravel()]) **assert: sorted increasing**


    Returns
    =======

    nothing; h is modified in-place
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t i, k, I1, I0, Imin
    cdef np.double_t mine, cure, h_lower
    assert h.shape[0] == n
    assert h_stop.shape[0] == n
    assert y.shape[0] == m

    I1 = I.shape[0]-1
    for i in reversed(range(n)): # improve h[n-1], h[n-2], ..., h[0]
        assert h[i] <= h_stop[i]
        while I[I1] > h_stop[i]:  # @TODO@ binsearch?
            I1 -= 1
            assert I1>=0
        assert I[I1] == minf(h_stop[i], h[i+1] if i<n-1 else np.inf)

        h_lower = h[i]
        h[i] = I[I1]
        mine = fitness_L1(x, y, h)
        I0 = I1
        minI = I1

        while I[I0] > h_lower:
            I0 -= 1
            assert I0>=0
            h[i] = I[I0]
            cure = fitness_L1(x, y, h)
            if cure < mine-EPS:  # Upper argmin
                mine = cure
                minI = I0
        assert I[I0] == h_lower

        h[i] = I[minI]
        I1 = minI



cpdef void fit_Sugeno_L1_improve_upper_bound(np.double_t[:] h, np.double_t[:] h_stop, np.double_t[:,:] x, np.double_t[:] y, np.double_t[:] I):
    """
    Tries to improve h -- finds h_new such that h_stop <= h_new <= h

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    y - a vector of m reference outputs

    h, h_stop - an increasing (weakly) coefficients vector of length n, h_stop[i]<=h[i]

    I - result of np.unique(np.r_[x.ravel(),y.ravel()]) **assert: sorted increasing**


    Returns
    =======

    nothing; h is modified in-place
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t i, k, I1, I0, Imin
    cdef np.double_t mine, cure, h_lower
    assert h.shape[0] == n
    assert h_stop.shape[0] == n
    assert y.shape[0] == m

    I1 = 0
    for i in range(n): # improve h[0], h[1], ..., h[n-1]
        assert h_stop[i] <= h[i]
        while I[I1] < h_stop[i]:  # @TODO@ binsearch?
            I1 += 1
            assert I1<len(I)
        assert I[I1] == maxf(h_stop[i], h[i-1] if i>0 else -np.inf)

        h_lower = h[i]
        h[i] = I[I1]
        mine = fitness_L1(x, y, h)
        I0 = I1
        minI = I1

        while I[I0] < h_lower:
            I0 += 1
            assert I0<len(I)
            h[i] = I[I0]
            cure = fitness_L1(x, y, h)
            if cure < mine-EPS:  # Lower argmin
                mine = cure
                minI = I0
        assert I[I0] == h_lower

        h[i] = I[minI]
        I1 = minI




cdef struct brnb_node_res:
    np.double_t fl # error at hl
    np.double_t fu # error at hu
    np.double_t f  # min(fl, fu)
    np.double_t e  # lower error bound estimate
    np.double_t maxd # max(hu-hl)
    np.double_t sumd # sum(hu-hl)




cpdef brnb_node_res fit_Sugeno_L1_brnb_node(np.double_t[:] hl, np.double_t[:] hu, np.double_t[:,:] x, np.double_t[:] y):
    """
    Tries to imply something about fitness_L1(x, y, h) for hl<=h<=hu

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    y - a vector of m reference outputs

    hl, hu - an increasing (weakly) coefficients vector of length n, hl[i]<=hu[i]


    Returns
    =======

    an instance of struct brnb_node_res, with fields:
        np.double_t fl # error at hl
        np.double_t fu # error at hu
        np.double_t f  # min(fl, fu)
        np.double_t e  # lower error bound estimate
        np.double_t maxd # max(hu-hl)
        np.double_t sumd # sum(hu-hl)
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t k, i
    cdef brnb_node_res res
    cdef np.double_t sl, su

    assert hl.shape[0] == n
    assert hu.shape[0] == n
    assert y.shape[0] == m

    res.maxd = 0.0
    res.sumd = 0.0
    for i in range(n):
        assert hl[i] <= hu[i]
        res.sumd += hu[i]-hl[i]
        if hu[i]-hl[i] > res.maxd:
            res.maxd = hu[i]-hl[i]


    res.fl = 0.0
    res.fu = 0.0
    res.e  = 0.0

    for k in range(m):
        sl = Sugeno(x[k,:], hl)
        su = Sugeno(x[k,:], hu)

        res.fl += absf(y[k]-sl)
        res.fu += absf(y[k]-su)

        if   y[k] > su: res.e += y[k]-su
        elif y[k] < sl: res.e += sl-y[k]
        else:           res.e += 0.0

    res.f = minf(res.fl, res.fu)
    assert res.e <= res.f+EPS
    return res




def fit_Sugeno_L1_brnb(np.ndarray[np.double_t,ndim=2] x, np.ndarray[np.double_t] y, np.uint_t maxiter=1_000_000):
    """
    Finds an L1 best fit for the Sugeno integral

    Arguments
    =========

    x - an m*n matrix representing m decreasingly (weakly) ordered input samples of length n

    y - a vector of m reference outputs

    maxiter - maximal number of iterations to perform

    Returns
    =======

    Dictionary with keys:
    * f - fitness at the solution
    * h - a vector of n increasing coefficients  with h[n-1]=1
    * niter - number of iterations performed
    * success - False if maxiter reached
    """
    cdef np.uint_t n = x.shape[1]
    cdef np.uint_t m = x.shape[0]
    cdef np.uint_t i, I0, I1, j, k, niter
    assert y.shape[0] == m

    cdef np.double_t[:] I = np.unique(np.r_[0.0, x.ravel(),y.ravel(), 1.0])
    cdef np.ndarray[np.double_t] hl = np.r_[[0.0]*(n-1), 1.0]
    cdef np.ndarray[np.double_t] hu = np.r_[[1.0]*(n-1), 1.0]
    cdef np.ndarray[np.double_t] hl1, hu1, hl2, hu2
    fit_Sugeno_L1_improve_upper_bound(hu, hl, x, y, I) # modifies h_upper in-place
    fit_Sugeno_L1_improve_lower_bound(hl, hu, x, y, I) # modifies h_lower in-place

    cdef np.double_t result_f = np.inf
    cdef np.ndarray[np.double_t] result_h = hl.copy() # placeholder
    cdef np.uint_t result_niter = 0

    cdef brnb_node_res brnb_node

    cdef object q = queue.Queue() #FIFO; for LIFO, use queue.LifoQueue()
    q.put((hl, hu))
    while not q.empty():
        result_niter += 1
        if result_niter % 1000 == 1: print("[%12d]%s"%(result_niter, "\u001B[D"*14),end="")
        if result_niter > maxiter: break #raise Exception("maxiter reached.")

        hl, hu = q.get()
        brnb_node = fit_Sugeno_L1_brnb_node(hl, hu, x, y)

        if brnb_node.fl <  result_f:
            result_f = brnb_node.fl
            result_h = hl.copy()

        if brnb_node.fu <  result_f:
            result_f = brnb_node.fu
            result_h = hu.copy()

        if brnb_node.e >=  result_f:
            continue # bound

        if brnb_node.maxd == 0.0:
            continue # singleton


        if False: # randomize
            i = np.random.choice(np.flatnonzero(hl != hu)) # [single value is returned]
        else:    # halving
            #idx = np.flatnonzero(hl != hu); i = idx[len(idx)//2]
            j = n-1
            i = 0
            while True:
                while hl[j] == hu[j]: j -= 1
                if i == j: break
                while hl[i] == hu[i]: i += 1
                if i >= j: break
                j -= 1
                i += 1
        assert i >= 0 and i < n and hl[i] != hu[i]

        I0 = 0
        while I[I0] < hl[i]: I0 += 1
        assert I[I0] == hl[i]

        I1 = I.shape[0]-1
        while I[I1] > hu[i]: I1 -= 1
        assert I[I1] == hu[i]
        assert I0 < I1

        if False: # randomize [single value is returned]
            j = np.random.randint(I0, I1, 1) # [single value is returned]
        else:    # halving
            j = (I0+I1)//2 # halving
        assert I0 <= j < I1

        hl1 = hl.copy()
        hu1 = hu.copy()
        for k in range(i+1): hu1[k] = minf(I[j], hu1[k])
        fit_Sugeno_L1_improve_upper_bound(hu1, hl1, x, y, I) # modifies h_upper in-place
        fit_Sugeno_L1_improve_lower_bound(hl1, hu1, x, y, I) # modifies h_lower in-place


        hu2 = hu.copy()
        hl2 = hl.copy()
        for k in range(i,n): hl2[k] = maxf(I[j+1], hl2[k])
        fit_Sugeno_L1_improve_upper_bound(hu2, hl2, x, y, I) # modifies h_upper in-place
        fit_Sugeno_L1_improve_lower_bound(hl2, hu2, x, y, I) # modifies h_lower in-place



        #if np.random.rand() < 0.5:
        # order doesn't make much difference for a FIFO queue:
        q.put((hl1, hu1))
        q.put((hl2, hu2))
        #else:
        #    q.put((hl2, hu2))
        #    q.put((hl1, hu1))

    return dict(f=result_f, h=result_h, niter=result_niter, success=(result_niter <= maxiter))



#   # Sugeno -- unit tests
#   assert Sugeno(np.r_[5.0, 5.0, 5.0, 5.0, 5.0], np.arange(1.0, 6.0)) == 5.0
#   assert Sugeno(np.r_[1.0, 1.0, 1.0, 1.0, 1.0], np.arange(1.0, 6.0)) == 1.0
#   assert Sugeno(np.r_[5.0, 4.0, 3.0, 2.0, 1.0], np.arange(1.0, 6.0)) == 3.0
#   assert Sugeno(np.r_[3.0, 3.0, 3.0, 0.0, 0.0], np.arange(1.0, 6.0)) == 3.0
#   assert Sugeno(np.r_[5.0, 5.0, 5.0, 3.0, 3.0], np.arange(1.0, 6.0)) == 3.0
#   for i in range(10000):
#       n = np.random.randint(10, 100)
#       h = np.sort(np.random.beta(np.random.uniform(0,10), np.random.uniform(0,10), n))
#       x = np.sort(np.random.beta(np.random.uniform(0,10), np.random.uniform(0,10), n))[::-1]
#       assert Sugeno(x, h) == np.max(np.minimum(x, h))
#
#   # Sugenos and fitness_L1 -- unit tests
#   assert np.all(Sugenos(np.array([
#       [5.0, 5.0, 5.0, 5.0, 5.0],
#       [1.0, 1.0, 1.0, 1.0, 1.0],
#       [5.0, 4.0, 3.0, 2.0, 1.0],
#       [3.0, 3.0, 3.0, 0.0, 0.0],
#       [5.0, 5.0, 5.0, 3.0, 3.0]
#   ]), np.arange(1.0, 6.0)) == np.r_[5.0,1.0,3.0,3.0,3.0])
#   for i in range(100):
#       n = np.random.randint(10, 100)
#       m = np.random.randint(10, 100)
#       h = np.sort(np.random.beta(np.random.uniform(0,10), np.random.uniform(0,10), n))
#       x = np.sort(np.random.beta(np.random.uniform(0,10), np.random.uniform(0,10), (m, n)), axis=1)[:,::-1]
#       y = Sugenos(x, h)
#       assert np.all(y == np.max(np.minimum(x, h), axis=1))
#       assert fitness_L1(x, y, h) == 0.0
#       y += np.random.normal(0, 0.05, m)
#       assert np.allclose(fitness_L1(x, y, h), np.sum(np.abs(Sugenos(x, h)-y)))
#       y = np.random.beta(np.random.uniform(0,10), np.random.uniform(0,10), m)
#       assert np.allclose(fitness_L1(x, y, h), np.sum(np.abs(Sugenos(x, h)-y)))
#
#   #%timeit fitness_L1(x, y, h)
#   #%timeit np.sum(np.abs(np.max(np.minimum(x, h), axis=1)-y))

