import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.ffd import createCylinder
from src.ffd import FFD

if __name__ == "__main__":

    X = createCylinder(radius=0.4, height=1.0)

    ### Control lattice ###
    ffd = FFD()
    ffd.lx = 1.2
    ffd.ly = 1.2
    ffd.lz = 1.2

    ffd.offx = -0.6
    ffd.offy = -0.6
    ffd.offz = -0.1

    ffd.l = 3
    ffd.n = 2
    ffd.m = 2


    ffd.createLattice()
    ffd.calcSTU(X[:, :, 0].flatten(), X[:, :, 1].flatten(), X[:, :, 2].flatten())
    ffd.Px[0,-1,1] = ffd.Px[0,-1,1]-2.5

    Xdef = ffd.calcDeformation()



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ffd.plotLattic(ax)

    ax.plot_wireframe(X[:, :, 0], X[:, :, 1], X[:, :, 2])
    ax.plot_surface(Xdef[:,0].reshape(20,20), Xdef[:,1].reshape(20,20), Xdef[:,2].reshape(20,20))

    plt.axis('equal')
    plt.show()