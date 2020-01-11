Fitting Sugeno Integrals to Data
================================

Gagolewski M., James S., Beliakov G., Supervised learning to aggregate data with the Sugeno integral, *IEEE Transactions on Fuzzy Systems* 27(4), 2019, pp. 810-815. doi:10.1109/TFUZZ.2019.2895565

A Cython implementation of the exact branch-refine-and-bound-type algorithm
for fitting Sugeno integrals with respect to symmetric capacities
minimizing mean absolute (L1) error,
particularly suitable for the case of ordinal data.

------------------------------------------------------------------------------

Main function: `fit_Sugeno_L1_brnb()` in `fit_Sugeno_L1_brnb.pyx`
(requires Cython and a C compiler) --
finds a best MAE fit for the Sugeno integral:

```
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


Example
=======

import pyximport
import numpy as np
pyximport.install(reload_support=True, setup_args={"include_dirs": np.get_include()})
import fit_Sugeno_L1_brnb as fit

x = ...inputs...
y = ...reference outputs...
result = fit.fit_Sugeno_L1_brnb(x, y)
```

------------------------------------------------------------------------------

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
