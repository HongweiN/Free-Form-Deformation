import re
import numpy as np
import os
import sys
import logging
from scipy import interpolate
import scipy.interpolate as si

logger = logging.getLogger('MISESFOIL')

class GeomTurbo(object):

    def __init__(self, gtfile, rotate = False):
        self.gtfile = gtfile
        self.gtfiletemp = self.gtfile + "_TEMP"
        self.rotate = rotate
        self.nblades = 0
        self.nsections = 0
        self.phirot = 0.0
        self.geometry = None
        self.turniningdirection = 1
        self.invert = False

    def readGeomturbofile(self):
        """
        Takes a geomturbo file and creates a dictionary out of it
        :param geomturbofile: Path to geomturbofile
        :param rotate: Rotate to 12o'clock position
        :return: geometry dictionary
        """
        # ============================================
        logger.info("Reading geomturbofile")

        keywords_start = [re.compile(r"\s*NI_BEGIN_NOTUSED\s*zrcurve\s*"),
                          re.compile(r"#?\s*NI_BEGIN\s*nibladegeometry\s*")]
        keywords_end = [re.compile(r"\s*NI_END_NOTUSED\s*zrcurve\s*"), re.compile(r"#?\s*NI_END\s*nibladegeometry\s*")]

        numpattern = re.compile(r'\s*((-?\d+\.\d+[Ee]?[+-]?\d*\s*)+)')
        # ============================================
        rowdict = {}

        hnsdict = {}
        f1 = open(self.gtfile, "r")
        f2 = open(self.gtfile + "_TEMP", "w")
        keyactivated = False
        activatedkeyword = ""
        keywordctr = 0

        # ============================================
        for line in f1:
            # ---------------------------------------
            # HNS - read hns data
            if keyactivated and activatedkeyword == 0:
                # Read ZR curve and set keyactivated False afterwards
                m = numpattern.match(line[:-1])
                if m:
                    aux = np.zeros((1, 2))
                    aux[0, :] = np.asarray(list(map(float, m.group(1).split())))
                    Xhns = np.append(Xhns[:], aux, axis=0)
                    hnsdict["flowpath_" + str(keywordctr)] = Xhns[:]
                    if not writtenonce:
                        f2.write("PARAM_hns_" + str(keywordctr) + "\n")
                        writtenonce = True
                else:
                    f2.write(line)
            # ---------------------------------------
            # Profile Section
            elif keyactivated and activatedkeyword == 1:
                # Read section and set keyactivated False afterwards
                # ---------------------------------------

                if "number_of_blades" in line[:-1]:
                    pattern = re.compile(r'\s*number_of_blades\s*(\d+)\s*')
                    m = pattern.match(line[:-1])
                    nblades = int(m.group(1))

                if "suction" in line[:-1]:
                    side = "ss"
                    sectionctr = -1
                elif "pressure" in line[:-1]:
                    side = "ps"
                    sectionctr = -1

                if "XYZ" in line[:-1]:
                    X = np.zeros((0, 3))  # x,y,z values of one section side!
                    sectionctr += 1

                # ---------------------------------------
                m = numpattern.match(line[:-1])
                if m:
                    aux = np.zeros((1, 3))
                    aux[0, :3] = np.asarray(list(map(float, m.group(1).split())))
                    X = np.append(X[:], aux, axis=0)
                    rowdict["points_" + side + "_" + str(sectionctr)] = X[:]

            else:
                # do nothing
                pass
            # ---------------------------------------
            # Keywordchecker
            # ---------------------------------------
            if not keyactivated:
                f2.write(line)
                for n, keyword in enumerate(keywords_start):
                    if keyword.match(line[:-1]):

                        keyactivated = True
                        activatedkeyword = n
                        # ---------------
                        Xhns = np.zeros((0, 2))
                        writtenonce = False
                        # ---------------
                        keywordctr += 1
                        # ---------------
                        if n == 1:
                            f2.write("PARAM_BLADEGEOM" + "\n")
                        break

            elif keyactivated:
                for n, keyword in enumerate(keywords_end):
                    if keyword.match(line[:-1]):
                        keyactivated = False
                        if n == 1:
                            f2.write(line)
                        break
        # ---------------------------------------
        f1.close()
        f2.close()

        # ---------------------------------------
        # ASSEMBLE AND RESTRUCTURE TO BLADEDICT OBJECT
        # ---------------------------------------
        phirot = 0.0
        if self.rotate:
            for nsec in range(sectionctr + 1):
                Xss = rowdict["points_" + "ss" + "_" + str(nsec)]
                Xps = rowdict["points_" + "ps" + "_" + str(nsec)]

                phirot += 0.5 / (sectionctr + 1) * (
                            np.mean(np.arctan2(Xss[:, 1], Xss[:, 0])) + np.mean(np.arctan2(Xps[:, 1], Xps[:, 0])))

            for nsec in range(sectionctr + 1):
                Xss = rowdict["points_" + "ss" + "_" + str(nsec)]
                Xps = rowdict["points_" + "ps" + "_" + str(nsec)]

                rss = (Xss[:, 1] ** 2 + Xss[:, 0] ** 2) ** 0.5
                rps = (Xps[:, 1] ** 2 + Xps[:, 0] ** 2) ** 0.5
                phiss = np.arctan2(Xss[:, 1], Xss[:, 0])
                phips = np.arctan2(Xps[:, 1], Xps[:, 0])

                Xss[:, 1], Xss[:, 0] = rss * np.sin(phiss - phirot), rss * np.cos(phiss - phirot)
                Xps[:, 1], Xps[:, 0] = rps * np.sin(phips - phirot), rps * np.cos(phips - phirot)

                rowdict["points_" + "ss" + "_" + str(nsec)] = Xss
                rowdict["points_" + "ps" + "_" + str(nsec)] = Xps

            logger.info("Rotate geomturbo by %s", np.rad2deg(phirot))
        # ---------------------------------------
        # ASSEMBLE AND RESTRUCTURE TO BLADEDICT OBJECT
        # ---------------------------------------
        self.nblades = nblades
        self.nsections = sectionctr + 1
        self.phirot = phirot

        self.geometry = dict()
        for nsec in range(sectionctr + 1):
            Xss = rowdict["points_" + "ss" + "_" + str(nsec)]
            Xps = rowdict["points_" + "ps" + "_" + str(nsec)]
            xss, yss, zss = Xss[:, 2], Xss[:, 1], Xss[:, 0]
            xps, yps, zps = Xps[:, 2], Xps[:, 1], Xps[:, 0]

            self.geometry[nsec] = [[xss, yss, zss], [xps, yps, zps]]

        if np.mean(yss) < np.mean(yps):
            self.turniningdirection = -1
        else:
            self.turniningdirection = 1
        # ---------------------------------------
        logger.info("Geomturbo file read")

    def equalizeGeometry(self,res=100):
        """
        Creates same resolution for each section
        :return: SS and PS geometry
        """
        import matplotlib.pyplot as plt
        import numpy as np

        sinter = GeomTurbo.resolution(res)
        X = np.zeros((sinter.shape[0],self.nsections,3,2))

        for nsec in range(self.nsections):

            for s,side in zip((0,1),("ss","ps")):
                x, y, z = self.geometry[nsec][s]

                ### sectionfit ###
                tck, _ = interpolate.splprep(self.geometry[nsec][s], s=0, k=2)


                geomEqual = interpolate.splev(sinter, tck)
                X[:,nsec, :, s]=np.asarray(geomEqual).T

        return X[:, :, :, 0], X[:, :, :, 1]

    def processEqualizedGeom(self, XSS, XPS):
        """

        :param XSS:
        :param XPS:
        :return:
        """
        X = [XSS, XPS]
        for nsec in range(self.nsections):
            for s,side in zip((0,1),("ss","ps")):

                self.geometry[nsec][s] = [X[s][0][:,nsec],X[s][1][:,nsec],X[s][2][:,nsec]]


    @staticmethod
    def resolution(res, letrans=0.2, tetrans=0.2):
        """
        :param res:
        :param letrans:
        :param tetrans:
        :return:
        """
        # =============
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # =============
        logpart = 2*sigmoid(np.linspace(-1e+1,0.0, int(res/3)))

        sle = 0.0 + letrans * logpart
        smid = np.linspace(letrans,(1.0-tetrans), int(res/3))
        ste = (1.0-tetrans) + tetrans * (1-logpart[::-1])

        sinter = np.hstack((sle,smid[1:],ste[1:]))

        return sinter

    @staticmethod
    def bspline(cv, n=100, degree=3):
        """ Calculate n samples on a bspline

            cv :      Array ov control vertices
            n  :      Number of samples to return
            degree:   Curve degree
        """
        cv = np.asarray(cv)
        count = cv.shape[0]

        # Prevent degree from exceeding count-1, otherwise splev will crash
        degree = np.clip(degree, 1, count - 1)

        # Calculate knot vector
        kv = np.array([0] * degree + range(count - degree + 1) + [count - degree] * degree, dtype='int')

        # Calculate query range
        u = np.linspace(0, (count - degree), n)

        # Calculate result
        return np.array(si.splev(u, (kv, cv.T, degree))).T

    def writeGeomturbofile(self, name="mygeomturbo.geomTurbo"):
        """
        Write bladedict to geometurbofile
        :param bladedict:
        :param name:
        :param rotate:
        :param invert:
        :return:
        """
        # ============================================
        logger.info("Writing new geomturbofile")

        # ==========================================
        if not os.path.isfile(self.gtfiletemp):
            logger.critical("Serious Error! Template geomturbo %s does not exist", self.gtfiletemp)
            sys.exit()

        os.system("cp " + self.gtfiletemp + " " + name)

        # ==========================================
        line = ""
        line = line + "TYPE                             GEOMTURBO\n"
        line = line + "GEOMETRY_MODIFIED                0\n"
        line = line + "GEOMETRY TURBO VERSION           5\n"
        line = line + "blade_expansion_factor_hub       0.01\n"
        line = line + "blade_expansion_factor_shroud    0.01\n"
        line = line + "intersection_parasolid\n"
        line = line + "intersection_npts    10\n"
        line = line + "intersection_control 0\n"
        line = line + "data_reduction                        1\n"
        line = line + "data_reduction_spacing_tolerance      1e-06\n"
        line = line + "data_reduction_angle_tolerance        90\n"
        line = line + "units		                      1\n"
        line = line + "number_of_blades		          " + str(self.nblades) + "\n"

        for s, side in enumerate(["suction", "pressure"]):
            line = line + side + "\n" + "SECTIONAL\n" + " " + str(self.nsections) + "\n"
            for nsec in range(self.nsections):

                # ----------------------------------------
                # ROtate Blade back to original orientation
                # ----------------------------------------
                if self.rotate:
                    [x, yrot, zrot] = self.geometry[nsec][s]
                    r = (yrot ** 2 + zrot ** 2) ** 0.5
                    phi = np.arctan2(yrot, zrot)
                    y, z = r * np.sin(phi + self.phirot), r * np.cos(phi + self.phirot)

                else:
                    [x, y, z] = self.geometry[nsec][s]

                if self.invert:
                    y = -y[:]

                # ----------------------------------------
                line = line + "# SECTION " + str(nsec + 1) + "\nXYZ\n" + str(x.shape[0]) + "\n"

                for i in range(x.shape[0]):
                    line = line + "   " + str(z[i]) + "   " + str(y[i]) + "   " + str(x[i]) + "\n"

        # ==========================================
        f1 = open(name, 'r')
        contents = f1.read()
        contents_updated = contents.replace("PARAM_BLADEGEOM", line)
        f1.close()
        f1 = open(name, 'w')
        f1.write(contents_updated)
        f1.close()
        # ---------------------------------------
        logger.info("Geomturbo file written!")