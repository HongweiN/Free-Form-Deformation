import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.ffd import createCylinder
from src.ffd import FFD
plt.style.use('ggplot')


if __name__ == "__main__":

    X = createCylinder(radius=0.4, height=1.0, imax=40, jmax=20)

    ### Control lattice ###
    ffd = FFD(geometry=[X[:, :, 0], X[:, :, 1], X[:, :, 2]])
    ffd.l = 3
    ffd.n = 3
    ffd.m = 3

    ffd.createLattice()
    ffd.calcSTU()

    ffd.Px[0,-1,1] = ffd.Px[0,-1,1]-0.8

    Xdef,Ydef,Zdef = ffd.calcDeformation()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    #ax.plot_wireframe(X[:, :, 0], X[:, :, 1], X[:, :, 2])
    ax.plot_surface(Xdef,Ydef,Zdef, antialiased=False)

    ffd.plotLattic(ax)

    plt.axis('equal')
    plt.show()