import sys, getopt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import exorings
import j1407

mearth = 5.97219e24 # kg
mmoon = 7.3476e22 # kg
mjup = 1.8986e27 # kg
au = 1.49597870700e11 # m
Mb = 23.8 # secondary in Mjup

# no scientific notation for numbers on plots
mpl.rc('axes.formatter', limits=(-7, 7))

# read in j1407 photometry
(time, flux, flux_err) = j1407.j1407_photom_binned('j1407_bincc.dat', 54160.0, 54300.0)
print 'number of photometric points: %d' % time.size

# get j1407 gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

try:
    opts, args = getopt.getopt(sys.argv[1:], "hr:o:s:", ["rfile=", "ofile=", "vstar="])
except getopt.GetoptError:
    print '%s -r <inputfile> -s <velocity in metres per second> -o <outputfile>' % sys.argv[0]
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print help
        sys.exit()
    elif opt in ("-r", "--rfile"):
        fitsin = arg
    elif opt in ("-o", "--ofile"):
        plotout = arg
    elif opt in ("-s", "--vstar"):
        v = np.array(float(arg))

print 'Reading in ring and disk parameters from %s' % fitsin
(res, taun_rings, rad_rings, dstar) = exorings.read_ring_fits(fitsin)

exorings.print_ring_tau(rad_rings, exorings.y_to_tau(taun_rings))

# set up stellar disk
kern = exorings.make_star_limbd(21, 0.8)

# make the radius and projected gradient for the measured gradient points
(ring_disk_fit, grad_disk_fit) = exorings.ring_grad_line(grad_time, res[0], res[1], res[2], res[3])

# produce fine grained gradient and ring values
samp_t = np.arange(-100, 100, 0.001) + 54222.
(samp_r, samp_g) = exorings.ring_grad_line(samp_t, res[0], res[1], res[2], res[3])
hjd_minr = samp_t[np.argmin(samp_g)]

(rstart, rend) = exorings.ring_mask_no_photometry(kern, dstar, time, res[0], res[1], res[2], res[3])

rmask = (rstart < 40)
rstart = rstart[rmask]
rend = rend[rmask]

# print disk parameters
exorings.print_disk_parameters(res, hjd_minr, samp_r)

# days to meters
dtm = rad_rings * 86400. * v

ring_radius = rad_rings * 86400. * v # m

rstartm = rstart * 86400. * v
rendm = rend * 86400. * v

exorings.print_ring_tau_latex(ring_radius / 1.e9, exorings.y_to_tau(taun_rings))

# radius on top, period below
fig_rad_per = plt.figure(figsize=(11, 5))

pr = fig_rad_per.add_subplot(111)
pr2 = pr.twiny()
pr3 = pr.twinx()

# min and max of plot window in million km
pr_min = 20
pr_max = 95
pr.axis((pr_min, pr_max, -0.2, 4.8))

# kappa is mass per unit area for unity transmission blocking
ring_radius_Mkm = ring_radius / 1.e9 # radius in million Km
ring_tau = -np.log(exorings.y_to_tau(taun_rings)) # natural log

# now let's sanity check by plotting red points at mid of rings
#  a  b  c  d
# r1 r2 r3 r4

# 0  r1 r2 r3
# r1 r2 r3 r4
# a  b  c  d

rlow = np.zeros_like(ring_radius_Mkm)
rlow[1:-1] = ring_radius_Mkm[0:-2]
rupper = ring_radius_Mkm
rmid = (rlow + rupper) / 2.

ring_area = np.pi * ((rupper*rupper) - (rlow*rlow))

kappa = 0.02 # cm2 g-1
surf_dens = 1./kappa # g cm-2

#              kg    cm2/m2 m2/km2  km2/Mkm2
k2 = surf_dens * 1e-3 * 1e4    * 1e6  * 1.e12 / mmoon

print 'surf_dens %f g.cm-2 == kappa %f Mmoon/Mkm2' % (surf_dens, k2)

# mass is kappa * (1 - exp(-tau)) / (1 - exp (-1.))

mass_ring = ring_area * k2 * (1- np.exp(-ring_tau)) / (1 - np.exp(-1.))

mr = np.copy(mass_ring)
mr[(mr < 1e-6)] = 1e-6
log_mr = np.log10(mr)

print 'kappa is %.3f cm2 g-1' % kappa
print 'total mass of all rings is %.3f Lunar masses' % np.sum(mass_ring)

# optical depth of order unity and kappa
pr.step(ring_radius / 1.e9, -np.log(exorings.y_to_tau(taun_rings)), \
    color='black', linewidth=2, zorder=-20)

# no photometry patches
for (bs, be) in zip(rstartm/1e9, rendm/1e9):
    pnew = mpl.patches.Rectangle((bs, -1), (be-bs), 6, color='0.8', zorder=-20)
    pr.add_patch(pnew)

# full intesnsity line
pr.hlines(0.0, 0, 1000, colors='black', linestyles='dashed')

pr.set_xlabel('Orbital radius (million km)', fontsize=16)
pr.set_ylabel('Tau', fontsize=16)

# now size of Hill sphere

# r0 is the centre of the ring gap in million km
# dr is the width of the gap
r0 = 60.2
dr = 4.0
mhill = np.power(((dr/2)/r0), 3.) * 3. * Mb
print 'Mass of satellite due to Hill gap is %5.2f M_Jup' % mhill
print 'Mass of satellite due to Hill gap is %5.2f M_earth' % (mhill * mjup / mearth)


# AU axis
# AU axis
# AU axis
# to set top axis scale, get min and max of old xaxis
(rlo, rhi) = pr.get_xlim()
#print "xlim is %f %f" % (rlo, rhi)

au_lo = rlo * 1e9 / au
au_hi = rhi * 1e9 / au

#print "au is %f %f" % (au_lo, au_hi)

pr2.set_xticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
pr2.set_xlim((au_lo, au_hi))
pr2.set_xticklabels((['0.2', '0.4', '0.6', '0.8', '1.0']))
pr2.xaxis.tick_top()
pr2.set_xlabel('Orbital radius (AU)', fontsize=16)

# RING MASS axis
# to set right axis scale, get min and max of old xaxis
(ylo, yhi) = pr.get_ylim()
#print "ylim is %f %f" % (ylo, yhi)

mass_lo = -2
mass_hi = 1.5

#print "mass range is %f %f" % (mass_lo, mass_hi)

pr3.set_yticks(np.array([1, 0, -1, -2]))
pr3.set_ylim((mass_lo, mass_hi))
pr3.scatter(rmid, log_mr, s=20, linewidth=2, facecolors='white', edgecolors='r')
pr3.tick_params(axis='both', which='major', labelsize=16)
pr3.set_yticklabels((['$10^{1}$', '$10^{0}$', '$10^{-1}$', '$10^{-2}$']))
pr3.yaxis.tick_right()
pr3.set_ylabel('Ring mass (Lunar masses)', fontsize=16)

# location of exomoon
pr.vlines(np.array(r0), -5., 5, zorder=1, lw=2, color='blue')

# adjust text size on the axes
pr2.tick_params(axis='both', which='major', labelsize=14)
pr.tick_params(axis='both', which='major', labelsize=14)

print 'writing out to %s' % plotout

plt.savefig(plotout)
