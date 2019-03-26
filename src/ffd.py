import numpy as np

def createCylinder(radius, height, imax=20, jmax=20):

    phi = np.linspace(0 , 2 *np.pi ,imax)
    z = np.linspace(0, height, jmax, endpoint=True)

    X = np.zeros((imax ,jmax, 3))

    for j in range(jmax):
        X[: ,j, 0] = radius * np.cos(phi)
        X[: ,j, 1] = radius * np.sin(phi)
        X[: ,j, 2] = z[j]

    return X


class FFD():
    def __init__(self):

        self.lx = 0.0
        self.ly = 0.4
        self.lz = 0.5

        self.offx = 0.0
        self.offy = 0.0
        self.offz = 0.0

        self.l = 3
        self.m = 3
        self.n = 3

        self.Px = None
        self.Py = None
        self.Pz = None

        self.X0 = 0.0
        self.Y0 = 0.0
        self.Z0 = 0.0

        self.s = None
        self.t = None
        self.u = None

    def createLattice(self):
        """
        Creates a simple reclinear block lattice
        :return:
        """
        self.Px = np.zeros((self.l,self.m,self.n))
        self.Py = np.zeros((self.l,self.m,self.n))
        self.Pz = np.zeros((self.l,self.m,self.n))

        for i in range(self.l):
            for j in range(self.m):
                for k in range(self.n):
                    self.Px[i, j, k] = self.offx + self.lx * i / (self.l - 1)
                    self.Py[i, j, k] = self.offy + self.ly * j / (self.m - 1)
                    self.Pz[i, j, k] = self.offz + self.lz * k / (self.n - 1)

        self.X0 = self.Px[0, 0, 0]
        self.Y0 = self.Py[0, 0, 0]
        self.Z0 = self.Pz[0, 0, 0]


    def calcSTU(self, xg, yg, zg):
        """
        Calc STU coordinates
        :param xg:
        :param yg:
        :param zg:
        :return:
        """

        self.s = (xg - self.X0)/self.lx
        self.t = (yg - self.Y0)/self.ly
        self.u = (zg - self.Z0)/self.lz


    def calcDeformation(self):

        Xdef = np.zeros((self.s.shape[0], 3))

        for p in range(self.s.shape[0]):
            for i in range(self.l):
                for j in range(self.m):
                    for k in range(self.n):
                        Xdef[p,:] += FFD.binomial(self.l-1,i)*np.power(1-self.s[p], self.l-1-i)*np.power(self.s[p],i) * \
                                     FFD.binomial(self.m-1,j)*np.power(1-self.t[p], self.m-1-j)*np.power(self.t[p],j) * \
                                     FFD.binomial(self.n-1,k)*np.power(1-self.u[p], self.n-1-k)*np.power(self.u[p],k) * \
                                     np.asarray([self.Px[i,j,k], self.Py[i,j,k], self.Pz[i,j,k]])

        return Xdef

    @staticmethod
    def binomial(n, k):
        """
        A fast way to calculate binomial coefficients by Andrew Dalke.
        See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0


    def plotLattic(self, ax):
        ax.scatter(self.Px, self.Py, self.Pz, c='r')