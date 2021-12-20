import cython
cdef extern from "math.h":
    double M_PI

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "math.h":
    double exp(double)

cdef extern from "math.h":
    double log(double)

cdef extern from "math.h":
    double sin(double)

cdef extern from "math.h":
    double cos(double)

cdef extern from "math.h":
    double acos(double)

cdef extern from "math.h":
    double asin(double)

cdef extern from "math.h":
    double atan2(double, double)

cdef extern from "math.h":
    double pow(double, double)


@cython.cdivision(True)
def arome(double mean_anomaly, double sma, double inc, double lda, ldc, double beta0, double Vsini, double sigma0, double zeta, double Rp, int Kmax, str units):
    """
    AROME: high-level API
    Computes and return analytical Rossiter-McLaughlin signals
    adapted the CCF or the iodine cell technique.

    Inputs:
    -------
    mean_anomaly: float
    Orbital mean anomaly in radian or degree

    sma: float
    Semi-major axis in stellar radii

    inc: float
    Orbital inclination in radian or degree

    lda: float
    Spin-orbit angle (lambda) in radian or degree

    ldc: array of 2 or 4 elements
    Limb darkening coefficients (2) for a linear law, (4) for a non-linear law

    beta0: float
    width of the non-rotating star in km/s

    Vsini: float
    Stellar projected rotational velocity in km/s

    sigma0: float
    Width of the best Gaussian fit in km/s

    zeta: float
    Macro-turbulence parameter in km/s

    Rp: float
    Radius of the planet in solar radius

    Kmax: float
    Order of expansion for the Iodine cell technique

    units: string
    units of all the input angles (mean_anomaly, inc, lda)
    Possible values: 'degree' or 'radian' (default)


    Outputs:
    --------

    vccf: float
    Value of the RM effect measured by the CCF technique in km/s



    Credits:
    --------
    Author  G. Boue  EXOEarths, Centro de Astrofisica, Universidade do Porto.
    Python translation  A. Santerne EXOEarths, Centro de Astrofisica, Universidade do Porto.

    Copyright (C) 2012, CAUP
    email of the author : gwenael.boue@astro.up.pt
    email of the Python translator : alexandre.santerne@astro.up.pt


    This work has been supported by the European Research Council/European
    Community under the FP7 through Starting Grant agreement number 239953, as
    well as from Fundacao para a Ciencia e a Tecnologia (FCT) through program
    Ciencia 2007 funded by FCT/MCTES (Portugal) and POPH/FSE (EC), and in the
    form of grants reference PTDC/CTE-AST/098528/2008, SFRH/BPD/71230/2010, and
    SFRH/BPD/81084/2011.


    License of the file :
    ---------------------
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """
    #import numpy as n
    if units == 'degree':
        # inclination
        inc *= M_PI/180.0           # radian */
        # spin-orbit angle
        lda *= M_PI/180.0           # radian */
        # Mean anomaly
        mean_anomaly *= M_PI/180.0

    # planet's coordinates */
    x0 = sma*(-cos(lda)*cos(mean_anomaly)+sin(lda)*sin(mean_anomaly)*cos(inc))
    y0 = sma*(-sin(lda)*cos(mean_anomaly)-cos(lda)*sin(mean_anomaly)*cos(inc))
    z0 = sma*sin(mean_anomaly)*sin(inc)
    # print x0, y0, z0

    EPSILON = 1e-20
    rho = sqrt(x0*x0+y0*y0)
    dmin = rho-Rp
    dmax = rho+Rp
    #import numpy as np

    if z0 < 0:
        return 0.
    elif dmin >= 1.0-EPSILON:
        return 0.
    else:

        parome = {'beta0': beta0, 'Vsini': Vsini,
            'sigma0': sigma0, 'zeta': zeta, 'Kmax': Kmax, 'Rp': Rp}

        # limb darkening
        limb_coef, limb_pow, kern_coef = arome_alloc_LDc(ldc)
        parome['limb_coef'] = limb_coef
        parome['limb_pow'] = limb_pow
        parome['kern_coef'] = kern_coef
        parome['nlimb'] = len(limb_coef)
        parome['Gaussfit_a0'] = setGaussfit_a0(parome)

        #Iodine_den = setIodine_den(parome)
        parome = arome_calc_fvpbetap(x0, y0, z0, parome)

        return arome_get_RM_CCF(parome)

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef arome_alloc_LDc(ldc1):
    """
    Define limb darkening coefficient for arome

    inputs:
    -------

    ldc: array of 2 or 4 floats
    Linear (2) or non-linear (4) limb darkening coefficients

    output:
    -------

    limb_coef: array
    limb darkening coefficients for arome

    limb_pow: array
    limb darkening power for arome
    """
    cdef double[:] limb_coef
    cdef double[:] limb_pow
    cdef double[:] ldc=ldc1
    cdef double denom
    import numpy as n
    if len(ldc) == 2:
        limb_coef = n.zeros(3, float)
        limb_pow = n.zeros(3, float)
        denom = M_PI*(1.0-ldc[0]/3.0-ldc[1]/6.0)
        limb_coef[0] = (1.0-ldc[0]-ldc[1])/denom
        limb_coef[1] = (ldc[0]+2.0*ldc[1])/denom
        limb_coef[2] = -1.*ldc[1]/denom
        limb_pow[0] = 0
        limb_pow[1] = 2
        limb_pow[2] = 4
    elif len(ldc) == 4:
        limb_coef = n.zeros(5, float)
        limb_pow = n.zeros(5, float)
        denom = M_PI*(1.0-ldc[0]/5.0-ldc[1]/3.0-3.0*ldc[2]/7.0-ldc[3]/2.0)
        limb_coef[0] = (1.0-ldc[0]-ldc[1]-ldc[2]-ldc[3])/denom
        limb_coef[1] = ldc[0]/denom
        limb_coef[2] = ldc[1]/denom
        limb_coef[3] = ldc[2]/denom
        limb_coef[4] = ldc[3]/denom
        limb_pow[0] = 0
        limb_pow[1] = 1
        limb_pow[2] = 2
        limb_pow[3] = 3
        limb_pow[4] = 4
    else:
        raise IOError(
            "arome_alloc_LDc argument must be a two or four parameters in an array")

    kern_coef = setrotkernel(limb_coef, limb_pow)
    # print len(kern_coef)
    return limb_coef, limb_pow, kern_coef


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
cdef setrotkernel(limb_coef1, limb_pow1):
    import numpy as n
    cdef double Im2 = M_PI  # int[-1,1] 1/sqrt(1-x**2) dx
    cdef double Im1 = 2.3962804694711844  # int[-1,1] 1/(1-x**2)**(1/4) dx
    cdef double Im0 = 2.0  # int[-1,1] dx
    cdef double  Ip1 = 1.7480383695280799  # int[-1,1] (1-x**2)**(1/4) dx

    cdef double  ntabI = 4
    cdef double[:] tabI
    cdef double[:] limb_pow=limb_pow1
    cdef double[:] limb_coef=limb_coef1
    cdef int lps = len(limb_pow)
    for k in range(lps):
        ntabI = max(ntabI, limb_pow[k]+4)
    # ntabI float -> ntabI int
    tabI = n.zeros(int(ntabI))

    tabI[-2] = Im2
    tabI[-1] = Im1
    tabI[0] = Im0
    tabI[1] = Ip1

    for k in range(2, int(ntabI)-2):
        tabI[k] = float(k)/float(k+2)*tabI[k-4]  # int[-1,1] (1-x**2)**(k/4) dx
    # print tabI
    cdef double alpha
    cdef double[:] kern_coef = n.zeros(len(limb_pow))
    for k in range(lps):
        alpha = limb_pow[k]
        # Alpha floar -> Alpha int
        kern_coef[k] = limb_coef[k]*tabI[int(alpha)]
    # print kern_coef
    return kern_coef

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double setGaussfit_a0(parome):
    # cdef double a=parome['sigma0']
    cdef double res1
    cdef double integral

    cdef double a = parome['sigma0']
    cdef double b = parome['beta0']
    cdef double c = parome['zeta']
    cdef double p1 = parome['Vsini']
    cdef int p2 = parome['nlimb']
    cdef double[:] p3 = parome['kern_coef']
    cdef double[:] p4 = parome['limb_pow']

    from scipy import integrate
    integral = integrate.quad(funcAmp_a0, 0.0, 1.0, (a,b,c,p1,p2,p3,p4), 20)[0]
    res1 = 4.0*a*sqrt(M_PI)*integral
    return res1


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double funcAmp_a0(double x, double a, double b, double c, double p1, int p2, p3,p4):
    sig2 = pow(a,2.)+pow(b,2.)+pow(c,2.)/2.0
    mu = sqrt(1.0-x*x)
    smu = sqrt(mu)

    # Rotation kernel
    cdef double Rx = 0.0
    cdef double Gx
    cdef double[:] m = p3
    cdef double[:] n = p4
    for k in range(p2):
        Rx += m[k]*pow(smu, n[k])
    Rx *= mu
    # Gaussian
    c1 = -pow(x*p1, 2.)*0.5*(1./sig2)
    c2 = sqrt(2.0*M_PI*sig2)
    Gx = exp(c1)/c2
    return Rx*Gx


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef arome_calc_fvpbetap(double x, double y, double z,parome):
    """
    Computes the flux f, the subplanet velocity vp, and dispersions betapR, betapT

    x coordinate of the planet in stellar radius
    y coordinate of the planet in stellar radius
    z coordinate of the planet in stellar radius

    r        :     largest radius of the planet                                 
    rx       :     x radius of the rotated projected planet                     
    ry       :     y radius of the rotated projected planet                     
    phi0     :     atan2(x0, y0)                                                
    rho      :     sqrt(x0**2+y0**2)                                            
    psi      :     angle OPA (O=star, P=planet, A=intersection star/planet)     
    phi1     :     limit inf of an arc                                          
    phi2     :     limit sup of an arc                                          
    xbar     :     averaged x coordinate of the planet                          
    ybar     :     averaged y coordinate of the planet                          
    II       :     II = I(xbar, ybar) (mean subplanet intensity)                
    Hxx      :     partial_x partial_x I(x,y)                                   
    Hyy      :     partial_y partial_y I(x,y)                                   
    Hxy      :     partial_x partial_y I(x,y)                                   
    Hxx2     :     partial_x partial_x I(x,y)                                   
    Hyy2     :     partial_y partial_y I(x,y)                                   
    Hxy2     :     partial_x partial_y I(x,y)                                   
    a00      :     area covered by the planet                                   
    axx      :     averaged of (x-xbar)**2                                      
    ayy      :     averaged of (y-ybar)**2                                      
    axy      :     averaged of (x-xbar)(y-ybar)                                 
    ff       :     fraction of flux oculted by the planet                       
    vv       :     subplanet velocity                                           
    v2       :     square of the subplanet velocity                             
    dmin     :     minimal distance between the planet ellipse and the origin   
    dmax     :     maximal distance between the planet ellipse and the origin   
    dbetaR   :     dispersion in the radial direction                           
    dbetaT   :     dispersion in the radial direction                           
    zetaR2   :     radial dispersion due to macro-turbulence                    
    zetaT2   :     tangential dispersion due to macro-turbulence                
    """
    cdef double xbar
    cdef double ybar
    cdef double a00
    cdef double axx
    cdef double ayy
    cdef double axy

    import numpy as n
    #import math
    EPSILON = 1e-20

    if z <= 0.0:  # the planet is behind the star
        parome['flux'] = 0.0
        parome['vp'] = 0.0
        parome['betapR'] = parome['beta0']
        parome['betapT'] = parome['beta0']
        return parome

    # parameters of the planet
    #rx, ry, r = parome['Rp'],parome['Rp'],parome['Rp']
    cdef double rx = parome['Rp']
    cdef double ry = parome['Rp']
    cdef double r = parome['Rp']
    cdef double phi0 = atan2(y, x)
    cdef double rho = sqrt(pow(x,2.)+pow(y,2.))
    cdef double dmin = rho-r
    cdef double dmax = rho+r

    if dmin >= 1.0-EPSILON:  # the planet doesn't overlap the stellar disk, that's life !
        parome['flux'] = 0.0
        parome['vp'] = 0.0
        parome['betapR'] = parome['beta0']
        parome['betapT'] = parome['beta0']
        return parome

    elif dmax <= 1.0:
        xbar = x  # int x dxdy / int dxdy
        ybar = y  # int y dxdy / int dxdy
        a00 = M_PI*rx*ry  # int dxdy
        axx = pow(rx,2.)/4.0  # int (x-x0)**2 dxdy / int dxdy
        ayy = pow(ry,2.)/4.0  # int (y-y0)**2 dxdy / int dxdy
        axy = 0.0        # int (x-x0)*(y-y0) dxdy / int dxdy

    else :  # during ingress and egress

        # stellar boundary
        psi = acos((1.0+pow(rho,2.)-pow(r,2.))/(2.0*rho))  # angle BSP (see Fig. 1)
        phi1 = phi0-psi  # angle xSB (see Fig. 1)
        phi2 = phi0+psi  # angle xSA (see Fig. 1)

        # print phi0, phi1, phi2

        a00  = funcF00(0.0, 0.0, 1.0,1.0,phi2)-funcF00(0.0,0.0,1.0,1.0,phi1)
        xbar = funcF10(0.0, 0.0, 1.0,1.0,phi2)-funcF10(0.0,0.0,1.0,1.0,phi1)
        ybar = funcF01(0.0, 0.0, 1.0,1.0,phi2)-funcF01(0.0,0.0,1.0,1.0,phi1)
        axx  = funcF20(0.0, 0.0, 1.0,1.0,phi2)-funcF20(0.0,0.0,1.0,1.0,phi1)
        ayy  = funcF02(0.0, 0.0, 1.0,1.0,phi2)-funcF02(0.0,0.0,1.0,1.0,phi1)
        axy  = funcF11(0.0, 0.0, 1.0,1.0,phi2)-funcF11(0.0,0.0,1.0,1.0,phi1)

        # planet boundary
        psi = acos(-1.*(1.0-pow(rho,2.)-pow(r,2.))/(2.0*r*rho))  # angle APS (see Fig. 1) in [0,pi]
        phi1 = phi0+M_PI-psi  # angle xPA (see Fig. 1)
        phi2 = phi0+M_PI+psi  # angle xPB (see Fig. 1)

        a00  += (funcF00(x, y, rx,ry,phi2)-funcF00(x,y,rx,ry,phi1))
        xbar += (funcF10(x, y, rx,ry,phi2)-funcF10(x,y,rx,ry,phi1))
        ybar += (funcF01(x, y, rx,ry,phi2)-funcF01(x,y,rx,ry,phi1))
        axx  += (funcF20(x, y, rx,ry,phi2)-funcF20(x,y,rx,ry,phi1))
        ayy  += (funcF02(x, y, rx,ry,phi2)-funcF02(x,y,rx,ry,phi1))
        axy  += (funcF11(x, y, rx,ry,phi2)-funcF11(x,y,rx,ry,phi1))

        # print a00, xbar, ybar, axx, ayy, axy

        xbar /= a00
        ybar /= a00
        axx = axx/a00 - pow(xbar,2.)
        ayy = ayy/a00 - pow(ybar,2.)
        axy = axy/a00 - xbar*ybar

        # print xbar, ybar, axx, ayy, axy
    cdef double Hxx0, Hyy0,Hxy0,II,ff,Hxx1, Hyy1,Hxy1,Hxx2, Hyy2,Hxy2, dbetaR,dbetaT

    II = funcIxn(xbar, ybar, parome,0)
    Hxx0, Hyy0, Hxy0 = HessIxn(xbar,ybar,parome,0)
    ff = a00*(II+0.5*(Hxx0*axx+Hyy0*ayy+2.0*Hxy0*axy))


    Hxx1, Hyy1, Hxy1 = HessIxn(xbar,ybar,parome,1)
    Hxx1 -= xbar*Hxx0
    Hyy1 -= xbar*Hyy0
    Hxy1 -= xbar*Hxy0
    vv = xbar + 0.5/II*(Hxx1*axx+Hyy1*ayy+2.0*Hxy1*axy)

    # print II, Hxx1*axx, Hyy1*ayy, Hxy1*axy, vv

    Hxx2, Hyy2, Hxy2 = HessIxn(xbar,ybar,parome,2)

    Hxx2 -= pow(xbar,2.)*Hxx0
    Hyy2 -= pow(xbar,2.)*Hyy0
    Hxy2 -= pow(xbar,2.)*Hxy0
    v2 = pow(xbar,2.) + 0.5/II*(Hxx2*axx+Hyy2*ayy+2.0*Hxy2*axy)

    # print II, axx, ayy, axy, v2
    # print v2 - vv**2

    # results

    parome['flux'] = ff
    parome['vp'] = vv
    dbetaR = sqrt(v2-pow(vv,2.))
    dbetaT = dbetaR
    cdef double mvsini=parome['Vsini']
    cdef double mzeta=parome['zeta']
    cdef double[:] limb_pow

    # print ff, vv, dbetaR, dbetaT

    # set the units

    parome['vp'] *= mvsini
    dbetaR *= mvsini
    dbetaT *= mvsini

    if mzeta > 0.0 : #take into account macro turbulence
        limb_pow = n.zeros(parome['nlimb'], float)
        mu2bar = 1.0-pow(xbar,2.)-pow(ybar,2.)

        # multiply I(x,y) by cos**2(theta)
        for k in range(parome['nlimb']):
            limb_pow[k] = parome['limb_pow'][k]
            parome['limb_pow'][k] += 4

        Hxx2, Hyy2, Hxy2 = HessIxn(xbar,ybar,parome,0)
        Hxx2 -= mu2bar*Hxx0
        Hyy2 -= mu2bar*Hyy0
        Hxy2 -= mu2bar*Hxy0
        zetaR2 = mu2bar + 0.5/II*(Hxx2*axx+Hyy2*ayy+2.0*Hxy2*axy)
        zetaT2 = 1.0-zetaR2

        zetaR2 *= pow(mzeta,2.)
        zetaT2 *= pow(mzeta,2.)

        dbetaR = sqrt(pow(dbetaR,2.)+zetaR2)
        dbetaT = sqrt(pow(dbetaT,2.)+zetaT2)

        # retrieve the initial limb-darkening law
        for k in range(parome['nlimb']):
            parome['limb_pow'][k] = limb_pow[k]
    cdef double mbeta0=parome['beta0']
    # add to the width of the non-rotating star
    parome['betapR'] = sqrt(pow(dbetaR,2.)+pow(mbeta0,2.))
    parome['betapT'] = sqrt(pow(dbetaT,2.)+pow(mbeta0,2.))

    return parome


@cython.cdivision(True)
cdef double funcF00(double x0, double y0, double rx, double ry, double phi):
    """
    Computes F_00(phi) (Boue et al., 2012, Tab1) use for the covered surface
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """
    return (0.5*(rx*ry*phi+x0*ry*sin(phi)-y0*rx*cos(phi)))


@cython.cdivision(True)
cdef double funcF10(double x0, double  y0, double rx,double ry,double phi):
    """
    Computes F_10(phi) (Boue et al., 2012, Tab1) use for the averaged velocity
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """
    return (-x0*y0*rx*cos(phi)-0.5*y0*pow(rx, 2.)*pow(cos(phi), 2.)+0.25*x0*rx*ry*(2.0*phi-sin(2.0*phi))+1.0/12.0*pow(rx,2.)*ry*(3.0*sin(phi)-sin(3.0*phi)))

cdef double funcF01(double x0, double  y0, double rx,double ry,double phi):
    """
    Computes F_01(phi) (Boue et al., 2012, Tab1) use for the averaged velocity
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """
    return  x0*y0*ry*sin(phi) + 0.5*x0*pow(ry,2.)*pow(sin(phi),2.) + 0.25*y0*rx*ry*(2.0*phi+sin(2.0*phi))-1.0/12.0*rx*pow(ry,2.)*(3.0*cos(phi)+cos(3.0*phi))


@cython.cdivision(True)
cdef double funcF20(double x0, double  y0, double rx,double ry,double phi):
    """
    Computes F_20(phi) (Boue et al., 2012, Tab1) use for the velocity dispersion
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """

    return -1.*pow(x0, 2)*y0*rx*cos(phi)\
            - x0*y0*pow(rx, 2.)*pow(cos(phi),2.)\
            + 0.25*pow(x0, 2.)*rx*ry*(2.0*phi-sin(2.0*phi))\
            - 1.0/12.0*y0*pow(rx, 3.)*(3.0*cos(phi)+cos(3.0*phi))\
            + 1.0/6.0*x0*pow(rx, 2.)*ry*(3.0*sin(phi)-sin(3.0*phi))\
            + 1.0/32.0*pow(rx, 3.)*ry*(4.0*phi-sin(4.0*phi))


@cython.cdivision(True)
cdef double funcF02(double x0, double  y0, double rx,double ry,double phi):
    """
    Computes F_02(phi) (Boue et al., 2012, Tab1) use for the velocity dispersion
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """
    return x0*pow(y0, 2.)*ry*sin(phi)\
            + x0*y0*pow(ry, 2.)*pow(sin(phi),2.)\
            + 0.25*pow(y0, 2.)*rx*ry*(2.0*phi+sin(2.0*phi))\
            + 1.0/12.0*x0*pow(ry, 3.)*(3.0*sin(phi)-sin(3.0*phi))\
            - 1.0/6.0*y0*rx*pow(ry, 2.)*(3.0*cos(phi)+cos(3.0*phi))\
            + 1.0/32.0*pow(ry, 3.)*rx*(4.0*phi-sin(4.0*phi))


@cython.cdivision(True)
cdef double funcF11(double x0, double  y0, double rx,double ry,double phi):
    """
    Computes F_11(phi) (Boue et al., 2012, Tab1) use for the velocity dispersion
    x_0 : x coordinates of the planet (or of the star)
    y_0 : y coordinates of the planet (or of the star)
    rx  : x radius of the planet (or of the star)
    ry  : y radius of the planet (or of the star)
    phi : limit of the arc 
    """
    return 0.25*x0*y0*(2.0*rx*ry*phi+x0*ry*sin(phi)-y0*rx*cos(phi))+0.125*pow(x0*ry*sin(phi), 2)\
            - 0.125*(pow(y0, 2.)+pow(ry,2.))*pow(rx*cos(phi),2.)\
            + 1.0/48.0*y0*pow(rx, 2.)*ry*(15.0*sin(phi)-sin(3*phi))\
            - 1.0/48.0*x0*rx*pow(ry, 2.)*(15.0*cos(phi)+cos(3*phi))


@cython.cdivision(True)
cdef funcIxn(double x0, double y0, parome, double n):
    """
    Computes x**n*B(x,y)*I(x,y) at the position (x0,y0)
    x0    : x coordinates of the planet
    y0    : y coordinates of the planet
    pdata : limb-darkening coefficient
    n     : power of x
    """
    return pow(x0, n)*funcLimb(x0, y0,parome)


@cython.cdivision(True)
cdef HessIxn(double x0, double y0, parome, double n):
    """
    Computes the Hessian of x**n*I(x,y) at the position (x0,y0)

    inputs:
    -------
    x0  :   x coordinates of the planet
    y0  :   y coordinates of the planet
    c1  :   limb-darkening coefficient
    c2  :   limb-darkening coefficient
    n   :   power of x

    outputs:
    --------
    Hxx :  partial_x partial_x (x^nI(x,y))
    Hyy :  partial_y partial_y (x^nI(x,y))
    Hxy :  partial_x partial_y (x^nI(x,y))
    """
    cdef double xn = pow(x0, n)
    cdef double L, Lx, Ly, Lxx, Lyy, Lxy, Hxx, Hyy,Hxy

    L = funcLimb(x0, y0, parome)
    Lx, Ly = dfuncLimb(x0, y0, parome)
    Lxx, Lyy, Lxy = ddfuncLimb(x0, y0,parome)

    # if n in [1,2] : print Lxx, Lyy, Lxy

    Hxx = xn*Lxx
    if n >0 :
        Hxx += 2.0*float(n)*x0**(n-1)*Lx
    if n>1 :
        Hxx += L*float(n)*float(n-1)*x0**(n-2)
    Hyy = xn*Lyy
    Hxy = xn*Lxy
    if n >0 :
        Hxy += Ly*float(n)*x0**(n-1)

    return Hxx, Hyy, Hxy


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef funcLimb(double x0, double  y0, parome):
    """
    Computes I(x,y)=sum cn*mu**(n/2) at the position (x0,y0)
    x0    : x coordinates of the planet
    y0    : y coordinates of the planet
    pdata : limb-darkening coefficients
    """
    cdef double[:] p4 = parome['limb_pow']
    cdef double[:] p5 = parome['limb_coef']
    cdef int p1 = parome['nlimb']
    cdef double mus, res, R
    R   = pow(x0, 2.)+pow(y0, 2.)
    mus = pow(1.0-R, 0.25)
    res = 0.0
    for k in range(p1):
        res += p5[k]*pow(mus, p4[k])
    return res


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef dfuncLimb(double x0, double y0, parome):
    """
    Computes the first derivatives of I(x,y)=sum cn*mu**(n/2) at the position (x0,y0)

    inputs:
    -------
    x0    :  x coordinates of the planet
    y0    :  y coordinates of the planet
    pdata :  limb-darkening coefficients

    outputs:
    --------
    Jx    : partial_x I(x,y)
    Jy    : partial_y I(x,y)
    """
    cdef double[:] p4 = parome['limb_pow']
    cdef double[:] p5 = parome['limb_coef']
    cdef int p1 = parome['nlimb']
    cdef double mus, mu2, R,dIdR
    R   = pow(x0, 2.)+pow(y0, 2.)
    mu2 = 1.0-R
    mus = pow(mu2, 0.25)
    dIdR = 0.0
    # print R, mu2, mus, dIdR
    for k in range(p1):
        dIdR -= 0.25*p4[k]*p5[k]*pow(mus, p4[k])/mu2
    Jx = 2.0*x0*dIdR
    Jy = 2.0*y0*dIdR

    return Jx, Jy


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef ddfuncLimb(double x0, double y0, parome):
    """
    Computes the second derivatives of I(x,y)=sum cn*mu**(n/2) at the position (x0,y0)

    inputs:
    -------
    x0    :   x coordinates of the planet
    y0    :   y coordinates of the planet
    pdata :   limb-darkening coefficient

    outputs:
    --------
    Hxx   :  partial_x partial_x (I(x,y))
    Hyy   :  partial_y partial_y (I(x,y))
    Hxy   :  partial_x partial_y (I(x,y))
    """
    cdef double[:] p4 = parome['limb_pow']
    cdef double[:] p5 = parome['limb_coef']
    cdef int p1 = parome['nlimb']
    cdef double mus, mu2, R,IR, IRR, var,Hxx, Hyy, Hxy
    R   = pow(x0, 2.)+pow(y0, 2.)
    mu2 = 1.0-R
    mus = pow(mu2, 0.25)
    IR = 0.0
    IRR = 0.0

    # print R, mu2, mus, IR, IRR

    for k in range(p1):
        var = 0.25*p4[k]*p5[k]*pow(mus, p4[k])/mu2
        IR -= var
        IRR += var*(0.25*p4[k]-1.0)/mu2

    Hxx = 2.0*IR+4.0*pow(x0,2.)*IRR
    Hyy = 2.0*IR+4.0*pow(y0,2.)*IRR
    Hxy = 4.0*x0*y0*IRR

    return Hxx, Hyy, Hxy


@cython.cdivision(True)
cdef double arome_get_RM_CCF(parome):
    """
    Computes the RM effect measured by the CCF technique.
    v = 1/den * (2*sig0**2/(sig0**2+betap**2))**(3/2)*f*vp*exp(-vp**2/(2*(sig0**2+betap**2)))

    input:
    ------
    parome :  simulation structure

    output:
    -------
    res    : result
    """
    #import numpy as n
    cdef double den = parome['Gaussfit_a0']
    cdef double f = parome['flux']
    cdef double vp = parome['vp']
    cdef double bs2 = pow(parome['sigma0'], 2.)
    cdef double bpR2 = pow(parome['betapR'], 2.)
    cdef double bpT2 = pow(parome['betapT'], 2.)
    if f < 0.0 :
        return 0
    #print(den, f, vp, bs2, bpR2, bpT2)
    if parome['zeta'] > 0.0:  # with macro-turbulence
        return -0.5/den*pow((2.0*bs2/(bs2+bpR2)), 1.5)*f*vp*exp(-pow(vp,2.)/(2.0*(bs2+bpR2))) - 0.5/den*pow(2.0*bs2/(bs2+bpT2),1.5)*f*vp*exp(-pow(vp,2.)/(2.0*(bs2+bpT2)))
    else:  # without macro-turbulence
        return -1.0/den*pow((2.0*bs2/(bs2+bpR2)), 1.5)*f*vp*exp(-pow(vp, 2.)/(2.0*(bs2+bpR2)))
