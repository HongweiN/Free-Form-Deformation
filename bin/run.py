import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.ffd import createCylinder
from src.ffd import FFD
from src.geomturbo import GeomTurbo

plt.style.use('ggplot')


gtfile = "/share/proj-mlhw01/lgt/4000F10/peere00c/190506_aeromechOpti/GAT3003_SID_newExtract/1dir_grid/dir_s3/autogrid.geomTurbo"
gt = GeomTurbo(gtfile=gtfile, rotate=True)
gt.readGeomturbofile()

XSS, XPS = gt.equalizeGeometry()

### Control lattice ###
geom1 = [XSS[:, :, 0], XSS[:, :, 1], XSS[:, :, 2]]
geom2 = [XPS[:, :, 0], XPS[:, :, 1], XPS[:, :, 2]]
geoms = [geom1,geom2]

ffd = FFD(geometries=geoms)
ffd.l = 3
ffd.n = 3
ffd.m = 3

ffd.createLattice()
ffd.calcSTU()

#ffd.Px[0, -1, 1] = ffd.Px[0, -1, 1] - 0.3

ffd.rotateLattice(1, deg=-30, scale=0.8, offx=0.01, offy=-0.05)


geomsDef = ffd.calcDeformation()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

for geomDef in geomsDef:
# ax.plot_wireframe(X[:, :, 0], X[:, :, 1], X[:, :, 2])
    ax.plot_surface(geomDef[0], geomDef[1], geomDef[2], antialiased=False)

ffd.plotLattic(ax)

plt.axis('equal')
plt.show()

#plt.plot(gt.geometry[0][0][0],gt.geometry[0][0][1])
#plt.plot(gt.geometry[0][1][0],gt.geometry[0][1][1])
#plt.show()
