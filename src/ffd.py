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


class FFD(object):
    def __init__(self, geometries=None):

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

        self.s = len(geometries)*[None]
        self.t = len(geometries)*[None]
        self.u = len(geometries)*[None]

        self.geometries = geometries
        self.originalShapes = [geom[0].shape for geom in geometries]

    def createLattice(self, offx=0.1, offy=0.1, offz=0.1):
        """
        Creates a simple reclinear block lattice
        :return:
        """

        self.Px = np.zeros((self.l,self.m,self.n))
        self.Py = np.zeros((self.l,self.m,self.n))
        self.Pz = np.zeros((self.l,self.m,self.n))

        Xmin, Xmax = np.zeros(3), np.zeros(3)
        for i in range(3):
            Xmin[i] = min([np.asarray(geom)[i, :, :].min() for geom in self.geometries])
            Xmax[i] = max([np.asarray(geom)[i, :, :].max() for geom in self.geometries])

        dx = Xmax[0] - Xmin[0]
        dy = Xmax[1] - Xmin[1]
        dz = Xmax[2] - Xmin[2]

        self.lx = (2 * offx + 1.0) * dx
        self.ly = (2 * offy + 1.0) * dy
        self.lz = (2 * offz + 1.0) * dz


        for i in range(self.l):
            for j in range(self.m):
                for k in range(self.n):
                    self.Px[i, j, k] = Xmin[0] -dx*offx + self.lx * i / (self.l - 1)
                    self.Py[i, j, k] = Xmin[1] -dy*offy + self.ly * j / (self.m - 1)
                    self.Pz[i, j, k] = Xmin[2] -dz*offz + self.lz * k / (self.n - 1)

        self.X0 = self.Px[0, 0, 0]
        self.Y0 = self.Py[0, 0, 0]
        self.Z0 = self.Pz[0, 0, 0]


    def rotateLattice(self, layer, deg=0.0, scale=1.0, offx=0.0, offy=0.0):
        """

        :param layer:
        :param deg:
        :return:
        """
        center = np.zeros(3)
        center[0] = self.Px[:, :, layer].mean()
        center[1] = self.Py[:, :, layer].mean()
        center[2] = self.Pz[:, :, layer].mean()

        rad = np.deg2rad(deg)

        for i in range(self.l):
            for j in range(self.m):
                x = center[0] + offx + scale*((self.Px[i, j, layer] - center[0]) * np.cos( rad) + (self.Py[i, j, layer] - center[1]) * np.sin(rad))
                y = center[1] + offy+ scale*((self.Px[i, j, layer] - center[0]) * np.sin(-rad) + (self.Py[i, j, layer] - center[1]) * np.cos(rad))
                self.Px[i, j, layer], self.Py[i, j, layer] = x, y


    def calcSTU(self):
        """
        Calc STU coordinates
        :param xg:
        :param yg:
        :param zg:
        :return:
        """
        for n, geometry in enumerate(self.geometries):
            self.s[n] = (geometry[0].flatten() - self.X0)/self.lx
            self.t[n] = (geometry[1].flatten() - self.Y0)/self.ly
            self.u[n] = (geometry[2].flatten() - self.Z0)/self.lz


    def calcDeformation(self):
        """
        Calculate the deformed geometry
        :return:
        """
        geometriesDef = list()

        for n, geometry in enumerate(self.geometries):
            Xdef = np.zeros((self.s[n].shape[0], 3))

            for p in range(self.s[n].shape[0]):
                for i in range(self.l):
                    for j in range(self.m):
                        for k in range(self.n):
                            Xdef[p,:] += FFD.binomial(self.l-1,i)*np.power(1-self.s[n][p], self.l-1-i)*np.power(self.s[n][p],i) * \
                                         FFD.binomial(self.m-1,j)*np.power(1-self.t[n][p], self.m-1-j)*np.power(self.t[n][p],j) * \
                                         FFD.binomial(self.n-1,k)*np.power(1-self.u[n][p], self.n-1-k)*np.power(self.u[n][p],k) * \
                                         np.asarray([self.Px[i,j,k], self.Py[i,j,k], self.Pz[i,j,k]])

            geometriesDef.append([Xdef[:,0].reshape(self.originalShapes[n]),
                                  Xdef[:,1].reshape(self.originalShapes[n]),
                                  Xdef[:,2].reshape(self.originalShapes[n])])

        return geometriesDef

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
                        ax.plot(_vx, _vy, _vz, '--', color='gray')
                    except:
                        pass

                    try:
                        _vx = [self.Px[i, j + 1, k], self.Px[i, j, k]]
                        _vy = [self.Py[i, j + 1, k], self.Py[i, j, k]]
                        _vz = [self.Pz[i, j + 1, k], self.Pz[i, j, k]]
                        ax.plot(_vx, _vy, _vz, '--', color='gray')
                    except:
                        pass

                    try:
                        _vx = [self.Px[i, j, k + 1], self.Px[i, j, k]]
                        _vy = [self.Py[i, j, k + 1], self.Py[i, j, k]]
                        _vz = [self.Pz[i, j, k + 1], self.Pz[i, j, k]]
                        ax.plot(_vx,_vy,_vz, '--', color='gray')
                    except:
                        pass

        ax.scatter(self.Px, self.Py, self.Pz, c='r')