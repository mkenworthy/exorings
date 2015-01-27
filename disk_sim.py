import sys, getopt, os
sys.path.append('/Users/kenworthy/Dropbox/python_workbooks/lib')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyfits

import exorings
import j1407

from scipy.optimize import fmin
#from scipy.ndimage import convolve
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

from matplotlib.patches import PathPatch

mpl.interactive(True)
# set sensible imshow defaults
mpl.rc('image', interpolation='nearest', origin='lower', cmap='gray')
mpl.rc('axes.formatter', limits=(-7, 7))

G = 6.6738480e-11 # m3 kg-1 s-2
yr = 365.242189669 * 86400  # sec
msol = 1.98855e30 # kg
rsol = 6.5500e8 # m
mjup = 1.8986e27 # kg
rjup = 6.9911e7 # m
mearth = 5.97219e24 # kg
mmoon = 7.3476e22 # kg
au = 1.49597870700e11 # m
pc = 3.0856e16 # m

# switch - implemented from http://code.activestate.com/recipes/410692/

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def Ptoa(P,m1,m2):
    """calculate orbital radius from period

    P period (years)
    m1 mass of primary (M_sol)
    m2 mass of secondary (M_jup)

    returns
    a semi-major axis (AU)
    """

    # a^3/P^2 = (G/4pipi) (m1 + m2)

    c = G / (4. * np.pi * np.pi)

    mu = (m1 * msol) + (m2 * mjup)

    a3 = np.power(P * yr, 2.) * (c * mu)

    return(np.power(a3,1./3.) / au)



def vcirc(m1,m2,a):
    """ Circular orbital velocity of m2 about m1 at distance a
        m1 in Solar masses
        m2 in Jupiter masses
        a in AU
        returns v in m/s
    """

    # http://en.wikipedia.org/wiki/Circular_orbit

    mu = G * ((m1*msol) + (m2*mjup))
    vcirc = np.power((mu / (a*au)), 0.5)
    return(vcirc)




def ringfunc(taun, *args):
    'cost function for ring fit to photometric data'
    (t, f, f_err, rad_ri, re, k, dst) = args
    # convolve and make smoothed ring photometry
    strip, dummy, g = exorings.ellipse_strip(rad_ri, \
        exorings.y_to_tau(taun), re[0], re[1], re[2], re[3], k, dst)

    # interpolate the smoothed curve....
    ring_model = interp1d(g[0], g[1], kind='linear')

    # ... to the times of the photometry
    ring_model_phot = ring_model(t)

    # calculate the residuals and the chi squared
    diff = f - ring_model_phot
    chisq = np.sum(np.power(diff/f_err, 2))
    red_chisq = chisq / diff.size

    return red_chisq

def calc_ring_stats(taun, t, f, f_err, rad_ri, re, k, dst, tmin, tmax):
    'full statistics function for ring fit to photometric data'
    # convolve and make smoothed ring photometry
    strip, dummy, g = exorings.ellipse_strip(rad_ri, \
        exorings.y_to_tau(taun), re[0], re[1], re[2], re[3], k, dst)

    # select points within the rings_tmin/tmax range
    mask = (t > tmin) * (t < tmax)
    t_sel = t[mask]
    f_sel = f[mask]
    f_err_sel = f_err[mask]

    print '%d points in time range %.2f to %.2f' % (t_sel.size, tmin, tmax)

    # interpolate the smoothed curve....
    ring_model = interp1d(g[0], g[1], kind='linear')

    # ... to the times of the photometry
    ring_model_phot = ring_model(t_sel)

    # calculate the residuals and the chi squared
    diff = f_sel - ring_model_phot

    chisq = np.sum(np.power(diff/f_err_sel, 2))
    # degrees of freedom = number of photometry points - number of ring edges - 1
    dof = diff.size - taun.size - 1
    red_chisq = chisq / dof
    print 'number of photometric = %d ' % diff.size
    print 'number of ring edges  = %d ' % taun.size
    print 'number of DOF         = %d ' % dof
    print 'chi squared           = %.2f' % chisq
    print ' reduced chi squared  = %.2f' % red_chisq

    # http://en.wikipedia.org/wiki/Bayesian_information_criterion
    # n - number of points in data
    # k - number of free parameters
    # BIC = chisquared + k . ln(n) + C
    # C is a constant which does not change between candidate models but is
    # dependent on the data points

    BIC = chisq + (taun.size) * np.log(diff.size)
    print ' BIC                  = %.2f' % BIC
    return red_chisq

nn = 1
def costfunc(x, *args):
    (y, dt, i_deg, phi_deg) = x
    (grad_t, grad_mag_n, t0) = args
    # grad_time
    global nn

    (tmp, grad_disk_fit) = exorings.ring_grad_line(grad_t, y, dt, i_deg, phi_deg)

    # lazy way of calculating the time midpoint of the light curve
    rmintime = np.arange(np.min(grad_t), np.max(grad_t), 0.01)
    (tmp, rminline) = exorings.ring_grad_line(rmintime, y, dt, i_deg, phi_deg)
    rmint = rmintime[np.argmin(rminline)]

    # make a cost function
    delta = grad_disk_fit - grad_mag_n

    # if delta is positive, keep it
    # if delta is negative, make it positive and multiply by 50
    delta[np.where(delta < 0)] = -delta[np.where(delta < 0)] * 50.

    # dean is penalty to clamp rmint
    dean = np.abs(rmint - t0)

    cost = np.sum(delta) + (dean * 20)
    nn += 1
    return cost

def ind_ring(ring, r):
    'returns index for closest ring r'
    rdiff = ring - r
    # find index of smallest positive rdiff
    return np.argmin(np.abs(rdiff))

def ind_ring_big(ring, r):
    'returns index for closest bigger ring r'
    rdiff = ring - r
    rdiff[(rdiff < 0)] += 99999.
    # find index of smallest positive rdiff
    return np.argmin(rdiff)

################################################################################
# BEGIN main program
################################################################################

(time, flux, flux_err) = j1407.j1407_photom_binned('j1407_bincc.dat', 54160.0, 54300.0)

# range of days that ring statistics should be considered
rings_tmin = 54220. - 30.
rings_tmax = 54220. + 30.

print 'restricting statistics of KIC to HJD range %.1f to %.1f' % (rings_tmin, rings_tmax)
goodp_rings = (time > rings_tmin) * (time < rings_tmax)
good_rings_npoints = goodp_rings.size
print 'number of points for statistics of KIC is %d' % (good_rings_npoints)

# get gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

# parse input options

fitsin = 'ring002.fits'
fitsout = 'ring003.fits'

read_in_ring_parameters = False
read_in_disk_parameters = False

tx = 54221.15
vstar = -1

def print_help():
    print 'disk_sim.py -r <ringfile> -d <diskfile> -t [time of min HJD] -o <outputfile>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:r:o:t:s:", \
        ["dfile=", "rfile=", "ofile=", "tx=", "vstar="])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt in ("-d", "--dfile"):
        fitsin_disk = arg
        read_in_disk_parameters = True
    elif opt in ("-r", "--rfile"):
        fitsin_ring = arg
        read_in_ring_parameters = True
    elif opt in ("-o", "--ofile"):
        fitsout = arg
    elif opt in ("-t", "--tx"):
        tx = np.array(float(arg))
        print 'tx = time of central eclipse forced to be = ', tx
    elif opt in ("-s", "--vstar"):
        vstar = np.array(float(arg))
print 'Output file is ', fitsout

# read in or create the ring system tau and radii

phi_deg = 165.2         # guess tilt disk in degrees
i_deg = 72.1            # guess inclination in degrees
dt = 54225             # guess at date of central eclipse (dr/dt)=0
y = 3.81               # guess at impact parameter b (units of days)

Rstar = 1.13 # solar radii \pm 0.14 van Werkhoven 14
Mstar = 0.9  # msol \pm 0.1 van Werkhoven 14
Rstar = 0.99 # solar radii \pm 0.11 Kenworthy 15a
Rstar = 0.93 # for the smallest possible radius from equatorial rotation Kenworthy 15a
Mb = 18.5 # secondary in Mjup
Pb = 5.5 # secondary period in Mjup

t_ecl = 56. # length of eclipse in days

# convert to an orbital velocity
a = Ptoa(Pb, Mstar, Mb)
print 'Primary mass     = %5.2f  msol' % Mstar
print 'Primary radius   = %5.2f  rsol' % Rstar
print 'Secondary mass   = %5.2f  mjup' % Mb
print 'Orbital radius   = %5.2f  AU' % a
v = vcirc(Mstar, Mb, a)

if vstar > 0:
    v = vstar
    print 'manual velocity of star is %.1f km.s-1' % v

print 'Orbital velocity = %5.2f  km/s (use option -s to set new velocity)' % (v/1.e3)

dstar = (Rstar * rsol * 2 / v) / 86400.

print 'Primary diameter = %5.2f  days' % dstar

if read_in_ring_parameters:
    print 'Reading in rings from %s' % fitsin_ring
    (resxx, taun_rings, rad_rings, xxxdstar) = exorings.read_ring_fits(fitsin_ring)
else:
    print "Starting with new rings...."
    rad_rings = np.array([59.0])
    taun_rings = np.array([0.0])
    rad_rings = np.append(rad_rings, (100.))
    taun_rings = np.append(taun_rings, (1000.))

exorings.print_ring_tau(rad_rings, exorings.y_to_tau(taun_rings))

if read_in_disk_parameters:
    print 'Reading in disk parameters from %s' % fitsin_disk
    (res, taun_ringsxx, rad_ringsxx, dstarxx) = exorings.read_ring_fits(fitsin_disk)
else:
    # run minimizer to find best guess values
    print 'No disk gradient parameters read in - refitting new ones....'
    res = fmin(costfunc, np.array([y, dt, i_deg, phi_deg]), maxiter=5000, \
        args=(grad_time, grad_mag_norm, tx))

# set up stellar disk
kern = exorings.make_star_limbd(21, 0.8)

# make the radius and projected gradient for the measured gradient points
(ring_disk_fit, grad_disk_fit) = exorings.ring_grad_line(grad_time, res[0], res[1], res[2], res[3])
# produce fine grained gradient and ring values
samp_t = np.arange(-100, 100, 0.001) + 54222.
(samp_r, samp_g) = exorings.ring_grad_line(samp_t, res[0], res[1], res[2], res[3])
hjd_minr = samp_t[np.argmin(samp_g)]

hjd_to_ring = interp1d(samp_t, samp_r, kind='linear')

(rstart, rend) = exorings.ring_mask_no_photometry(kern, dstar, time, res[0], res[1], res[2], res[3])

### RESULTS of fitting routine

print ''
print 'Disk parameters fitting to gradients'
print '------------------------------------'
print ''
print ' impact parameter b   = %8.2f days' % res[0]
print ' HJD min approach t_b = %8.2f days' % res[1]
print ' disk inclination i   = %7.1f  deg' % res[2]
print '        disk tilt phi = %7.1f  deg' % res[3]
print ' HJD min gradient     = %8.2f days' % hjd_minr
print '             rmin     = %8.2f days' % np.min(samp_r)

# http://en.wikipedia.org/wiki/Bayesian_information_criterion
# n - number of points in data
# k - number of free parameters
# BIC = chisquared + k . ln(n) + C
# C is a constant which does not change between candidate models but is
# dependent on the data points

# plot folded light curve
time0 = np.abs(time-hjd_minr)
time0_grad = np.abs(grad_time-hjd_minr)

# flux_color and flux_col
# hold the color of the points for ingress and egress
flux_color = np.chararray((time.shape))
flux_color[:] = 'b'
flux_color[(time > hjd_minr)] = 'r'

# probably a better pythonic way to do this, but this works.
flux_col = ''
for b in flux_color.tolist():
    flux_col = str.join('', (flux_col, b))

def plot_folded_phot(f):
    'plot folded J1407 light curve'

    # j1407 photometry
    h1.scatter(time0, flux, c=flux_col, s=20, edgecolors='none', zorder=-20)
    h1.errorbar(time0, flux, flux_err, zorder=-30, ls='none')

    # gradient measurements
#    h1.scatter(time0_grad,np.ones_like(time0_grad)*0.8)

fig_fold = plt.figure(figsize=(16, 6))

h1 = fig_fold.add_subplot(111)
plot_folded_phot(fig_fold)

strip, dummy, g = exorings.ellipse_strip(rad_rings, exorings.y_to_tau(taun_rings), \
    res[0], res[1], res[2], res[3], kern, dstar)

# g[0] = time
# g[1] = stellar convolved tau
# g[2] = stellar tau no convolution
# g[3] = gradient
gt_abs = np.abs(g[0]-hjd_minr)
g1 = g[1]
gt_ingr = gt_abs[(g[0] <= hjd_minr)]
gt_egr = gt_abs[(g[0] > hjd_minr)]
g1_ingr = g1[(g[0] <= hjd_minr)]
g1_egr = g1[(g[0] > hjd_minr)]

h1.plot(np.abs(g[0]-hjd_minr), g[1])
h1.plot(gt_ingr, g1_ingr, color='blue')
h1.plot(gt_egr, g1_egr, color='red')
h1.plot(np.abs(g[0]-hjd_minr), g[2], color='orange')
h1.set_xlabel('Time from eclipse midpoint [days]')
h1.set_ylabel('Transmission')

print "Menu"
print "a - add a ring"
print "d - delete a ring boundary to the right"
print "o - run Amoeba optimizer"
print "v - display rings in vector format"
print "r - display rings in pixel format (slow)"
print ""
badring = 1

def onclick(event):
    global rad_rings  # naughty, I know, I know...
    global taun_rings
    global badring

    for case in switch(event.key):

        newt = event.xdata
        newtau = event.ydata
        print 'newtau is %f' % newtau
        if newtau > 1.0:
            newtau = 1.0
        if newtau < 0.0:
            newtau = 0.000001
        newr = hjd_to_ring(newt + hjd_minr)

        if case('r'):
            exorings.plot_tilt_faceon(rad_rings, taun_rings, res, hjd_minr, dstar)

            break

        if case('v'):
            exorings.plot_tilt_faceon_vect(rad_rings, taun_rings, res, \
                hjd_minr, rstart, rend, dstar)

            break

        if case('a'):
            print 'a pressed, adding a new ring'
            # assume that r is ordered

            print ' inserting new ring date %d and radius %d' % (newt, newr)
            bigr = np.append(rad_rings, newr)
            bigtau = np.append(taun_rings, exorings.tau_to_y(newtau))

            # index sort r, rearrange both r and tau to this
            sortedr = np.argsort(bigr)
            rad_rings = bigr[sortedr]
            taun_rings = bigtau[sortedr]
            break

        if case('d'):
            if rad_rings.size > 1:
                # find closest ring distance
                rsel_idx = ind_ring(rad_rings, newr)
                rad_rings = np.delete(rad_rings, rsel_idx)
                taun_rings = np.delete(taun_rings, rsel_idx)
            break

        if case('m'):
            rsel_idx = ind_ring(rad_rings, newr)

            rad_rings[rsel_idx] = newr
            taun_rings[rsel_idx] = exorings.tau_to_y(newtau)

            break

        if case('b'):
            badring *= -1
            print badring

            break

        if case('o'):
            taun_rings = fmin(ringfunc, taun_rings, maxiter=1000, \
                args=(time, flux, flux_err, rad_rings, res, kern, dstar))

            break

        if case('q'):
            print 'q is for quitters'
            break
        if case():
            print 'not a recognised keypress'


    # print stats of fit
    calc_ring_stats(taun_rings, time, flux, flux_err, rad_rings, res, \
        kern, dstar, rings_tmin, rings_tmax)

    # regenerate drawing
    strip, dummy, g = exorings.ellipse_strip(rad_rings, exorings.y_to_tau(taun_rings), \
        res[0], res[1], res[2], res[3], kern, dstar)
    gt_abs = np.abs(g[0]-hjd_minr)
    g1 = g[1]
    gt_ingr = gt_abs[(g[0] <= hjd_minr)]
    gt_egr = gt_abs[(g[0] > hjd_minr)]
    g1_ingr = g1[(g[0] <= hjd_minr)]
    g1_egr = g1[(g[0] > hjd_minr)]

    # save the current axis ranges
    x1, x2 = h1.get_xlim()
    y1, y2 = h1.get_ylim()

    h1.cla()
    plot_folded_phot(fig_fold)
    h1.plot(event.xdata, event.ydata, color='green')
    h1.plot(np.abs(g[0]-hjd_minr), g[1])
    h1.plot(gt_ingr, g1_ingr, color='blue')
    h1.plot(gt_egr, g1_egr, color='red')
    h1.plot(np.abs(g[0]-hjd_minr), g[2], color='orange')
    h1.set_xlabel('Time from eclipse midpoint [days]')
    h1.set_ylabel('Transmission')

    # zoom back the the last view
    h1.set_xlim([x1, x2])
    h1.set_ylim([y1, y2])
    plt.draw()

    # save tmp version
    # erase old version otherwise write_ring_fits throws a fit
    if os.path.isfile('tmp.fits'):
        os.remove('tmp.fits')
    exorings.write_ring_fits('tmp.fits', res, taun_rings, rad_rings, dstar)

cid = fig_fold.canvas.mpl_connect('key_press_event', onclick)
plt.show()

raw_input('press return to finish')
exorings.write_ring_fits(fitsout, res, taun_rings, rad_rings, dstar)

