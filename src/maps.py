#############################################################################################
#############################################################################################
#                                                                                           #
# import the libraries                                                                      #
#                                                                                           #
#############################################################################################
#############################################################################################



import numpy as np



#############################################################################################
#############################################################################################
#                                                                                           #
# create the triangular map for 2D transport                                                #
#                                                                                           #
#############################################################################################
#############################################################################################



class TriangularTransport2D:
    def __init__(self, degree, n_quad = 20):
        """main definition of the object"""

        # we fix the polynomial degree and the quadrature infos
        self.d = degree
        self.xq, self.wq = np.polynomial.legendre.leggauss(n_quad)

        # we set the indices for the alpha_2 parameter (the others are trivial)
        self.alpha2_idx = [
            (i, j) for i in range(degree + 1)
                   for j in range(degree + 1 - i)
        ]

        # n_params gives the total number of parameters (#a1 + #alpha1 + #a2 + #alpha2)
        self.n_params = 1 + (degree + 1) + (degree + 1) + len(self.alpha2_idx)

    # ------------------------------------------------------------------

    def initial_theta(self, scale=0.01, seed=None):
        """ initialize each parameter with a random small number (using gaussian noise) """

        if seed is not None:
            np.random.seed(seed)

        theta = np.zeros(self.n_params)

        # n_params long cycle
        for i in range(self.n_params):
            theta[i] = scale * np.random.randn()

        return theta

    # ------------------------------------------------------------------

    def unpack(self, theta):
        """ starting from theta, we want to unpack the parameters a1, alpha1, a2, alpha2 """

        # degree of the polynomial
        d1 = self.d + 1

        #a1
        a1 = theta[0]

        #alpha1
        start = 1
        alpha1 = theta[start : start + d1]

        #a2
        start += d1
        a2 = theta[start : start + d1]

        #alpha2
        start += d1
        alpha2 = theta[start:]

        # output
        return a1, alpha1, a2, alpha2

    # ------------------------------------------------------------------

    def p1(self, x, alpha1):
        """ first polynomial """

        val = 0.0
        for k in range(len(alpha1)):

            # add alpha1[k] * x^k term
            val += alpha1[k] * (x ** k)
        return val


    def p2(self, x1, x2, alpha2):
        """ second polynomial """

        val = 0.0
        for k in range(len(alpha2)):

            # get polynomial exponents
            i, j = self.alpha2_idx[k]

            # add alpha2[k] * x1^i * x2^j term
            val += alpha2[k] * (x1 ** i) * (x2 ** j)
        return val

    # ------------------------------------------------------------------

    def T1(self, x1, a1, alpha1):
        """ T^1 """

        if x1 == 0.0:
            return a1

        # map interval [0, x1] to [-1, 1]
        mid = 0.5 * x1
        half = 0.5 * x1

        s = 0.0
        for xi, wi in zip(self.xq, self.wq):

            # quadrature evaluation point
            t = mid + half * xi

            # accumulate exp(p1(t))
            # s += wi * np.exp(self.p1(t, alpha1))
            s += wi * self.safe_exp(self.p1(t, alpha1))

        return a1 + half * s


    def T2(self, x1, x2, a2, alpha2):
        """ T^2 """

        # polynomial part depending on x1
        poly = 0.0
        for k in range(len(a2)):
            poly += a2[k] * (x1 ** k)

        if x2 == 0.0:
            return poly

        # map interval [0, x2] to [-1, 1]
        mid = 0.5 * x2
        half = 0.5 * x2

        s = 0.0
        for xi, wi in zip(self.xq, self.wq):

            # quadrature evaluation point
            t = mid + half * xi

            # accumulate exp(p2(x1, t))
            # s += wi * np.exp(self.p2(x1, t, alpha2))
            s += wi * self.safe_exp(self.p2(x1, t, alpha2))


        return poly + half * s

    # ------------------------------------------------------------------

    def forward(self, x, theta):
        """ push forward the samples x """

        # get parameters
        a1, alpha1, a2, alpha2 = self.unpack(theta)

        n = len(x)
        y = np.zeros((n, 2))

        # push the sampls
        for i in range(n):
            x1 = x[i][0]
            x2 = x[i][1]

            y[i][0] = self.T1(x1, a1, alpha1)
            y[i][1] = self.T2(x1, x2, a2, alpha2)

        return y

    # ------------------------------------------------------------------

    def log_det_jacobian(self, x, theta):
        """ log |det J| of the transport map """

        _, alpha1, _, alpha2 = self.unpack(theta)

        return np.array([
            self.p1(x1, alpha1) + self.p2(x1, x2, alpha2)
            for x1, x2 in x
        ])
    
    # ------------------------------------------------------------------

    def inverse(self, y, theta, max_iter = 100):
        """ finding the inverse """

        a1, alpha1, a2, alpha2 = self.unpack(theta)

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, 2)

        n = y.shape[0]
        x = np.zeros((n, 2))

        for k in range(n):
            y1, y2 = y[k]

            # invert T1 by bisection
            lo, hi = -10.0, 10.0
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self.T1(mid, a1, alpha1) < y1:
                    lo = mid
                else:
                    hi = mid
            x1 = 0.5 * (lo + hi)

            # invert T2 by bisection (x1 is now fixed)
            lo, hi = -10.0, 10.0
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self.T2(x1, mid, a2, alpha2) < y2:
                    lo = mid
                else:
                    hi = mid
            x2 = 0.5 * (lo + hi)

            x[k, 0] = x1
            x[k, 1] = x2

        return x if n > 1 else x[0]
     
    # ------------------------------------------------------------------

    def safe_exp(self, u, clip=2000.0):
        """ required techincality to avoid overflow issues during training of the maps """

        return np.exp(np.clip(u, -clip, clip))
