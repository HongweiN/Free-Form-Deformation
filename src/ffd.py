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
    def __init__(self, geometry=None):

        self.lx = 0.0
        self.ly = 0.4
        self.lz = 0.5

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

        self.geometry = geometry
        self.originalShape = None

    def createLattice(self, offx=0.1, offy=0.1, offz=0.1):
        """
        Creates a simple reclinear block lattice
        :return:
        """

        self.Px = np.zeros((self.l,self.m,self.n))
        self.Py = np.zeros((self.l,self.m,self.n))
        self.Pz = np.zeros((self.l,self.m,self.n))

        X,Y,Z = self.geometry[0], self.geometry[1], self.geometry[2]
        self.originalShape = X.shape

        dx = (X.max() - X.min())
        dy = (Y.max() - Y.min())
        dz = (Z.max() - Z.min())

        self.lx = (2 * offx + 1.0) * dx
        self.ly = (2 * offy + 1.0) * dy
        self.lz = (2 * offz + 1.0) * dz


        for i in range(self.l):
            for j in range(self.m):
                for k in range(self.n):
                    self.Px[i, j, k] = X.min() -dx*offx + self.lx * i / (self.l - 1)
                    self.Py[i, j, k] = Y.min() -dy*offy + self.ly * j / (self.m - 1)
                    self.Pz[i, j, k] = Z.min() -dz*offz + self.lz * k / (self.n - 1)

        self.X0 = self.Px[0, 0, 0]
        self.Y0 = self.Py[0, 0, 0]
        self.Z0 = self.Pz[0, 0, 0]


    def calcSTU(self):
        """
        Calc STU coordinates
        :param xg:
        :param yg:
        :param zg:
        :return:
        """

        self.s = (self.geometry[0].flatten() - self.X0)/self.lx
        self.t = (self.geometry[1].flatten() - self.Y0)/self.ly
        self.u = (self.geometry[2].flatten() - self.Z0)/self.lz


    def calcDeformation(self):
        """
        Calculate the deformed geometry
        :return:
        """

        Xdef = np.zeros((self.s.shape[0], 3))

        for p in range(self.s.shape[0]):
            for i in range(self.l):
                for j in range(self.m):
                    for k in range(self.n):
                        Xdef[p,:] += FFD.binomial(self.l-1,i)*np.power(1-self.s[p], self.l-1-i)*np.power(self.s[p],i) * \
                                     FFD.binomial(self.m-1,j)*np.power(1-self.t[p], self.m-1-j)*np.power(self.t[p],j) * \
                                     FFD.binomial(self.n-1,k)*np.power(1-self.u[p], self.n-1-k)*np.power(self.u[p],k) * \
                                     np.asarray([self.Px[i,j,k], self.Py[i,j,k], self.Pz[i,j,k]])

        return Xdef[:,0].reshape(self.originalShape), Xdef[:,1].reshape(self.originalShape), Xdef[:,2].reshape(self.originalShape)

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
        """
        Plot the lattice
        :param ax:
        :return:
        """
        for i in range(self.l):
            for j in range(self.m):
                for k in range(self.n):
                    try:
                        _vx = [self.Px[i + 1, j, k], self.Px[i, j, k]]
                        _vy = [self.Py[i + 1, j, k], self.Py[i, j, k]]
                        _vz = [self.Pz[i + 1, j, k], self.Pz[i, j, k]]
                        ax.plot(_vx,_vy,_vz, 'k-')
                    except:
                        pass

                    try:
                        _vx = [self.Px[i, j + 1, k], self.Px[i, j, k]]
                        _vy = [self.Py[i, j + 1, k], self.Py[i, j, k]]
                        _vz = [self.Pz[i, j + 1, k], self.Pz[i, j, k]]
                        ax.plot(_vx,_vy,_vz, 'k-')
                    except:
                        pass

                    try:
                        _vx = [self.Px[i, j, k + 1], self.Px[i, j, k]]
                        _vy = [self.Py[i, j, k + 1], self.Py[i, j, k]]
                        _vz = [self.Pz[i, j, k + 1], self.Pz[i, j, k]]
                        ax.plot(_vx,_vy,_vz, 'k-')
                    except:
                        pass

        ax.scatter(self.Px, self.Py, self.Pz, c='r')