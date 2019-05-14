#!/swd/SPost/virtualenv/conda/envs/SPost_Py2/bin/python


# /usw/spgvenvs/py_core/2.0.0/rhel7/bin/python
import sys
import os

sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt
from src.ffd import FFD
from src.geomturbo import GeomTurbo
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')


gtfile = sys.argv[1]

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

ffd.rotateLattice(0, deg=-40, scalex=0.9, scaley=1.0, offx=0.00, offy=-0.00)
ffd.rotateLattice(1, deg=40, scalex=0.5, scaley=1.0, offx=0.00, offy=-0.16)
ffd.rotateLattice(2, deg=-40, scalex=0.9, scaley=1.0, offx=0.00, offy=-0.06)

geomsDef = ffd.calcDeformation()

gt.processEqualizedGeom(geomsDef[0],geomsDef[1])
gt.writeGeomturbofile()

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
