import sys, getopt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import exorings
import j1407

# set sensible imshow defaults
mpl.rc('axes.formatter', limits=(-7, 7))

def plot_gradient_fit(t, f, fn, xt, yt, p):
    # f = gradient of fit at points of measurement
    p.plot(xt, yt, lw=3.0, color='black', zorder=1)
    p.scatter(t, f, facecolor='1.0', s=60, color='black', zorder=2, lw=1)
    p.scatter(t, f, facecolor='None', s=60, color='black', zorder=3, lw=1)
    p.scatter(t, fn, facecolor='0.0', s=60, zorder=4, lw=1)
    p.vlines(t, f, fn, zorder=1, lw=2, color='0.5', linestyles='dotted')
    p.set_xlabel('HJD - 2450000 [Days]')
    p.set_ylabel('Light curve gradient [$L_\star/day$]')

# parse command line options
try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:o:", ["dfile=", "ofile="])
except getopt.GetoptError:
    print 'plot_disk_fit.py -d <disk input FITS> -o <output plot file>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        help()
    elif opt in ("-d", "--dfile"):
        fitsin_disk = arg
        read_in_disk_parameters = True
    elif opt in ("-o", "--ofile"):
        plotfileout = arg

# get j1407 gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

# read in the ring system tau and radii
print 'Reading in disk parameters from %s' % fitsin_disk
(res, taun_ringsxx, rad_ringsxx, dstar) = exorings.read_ring_fits(fitsin_disk)

# make the radius and projected gradient for the measured gradient points
(ring_disk_fit, grad_disk_fit) = \
    exorings.make_ring_grad_line(grad_time, res[0], res[1], res[2], res[3])

# produce fine grained gradient and ring values
samp_t = np.arange(-100, 100, 0.001) + 54222.
(samp_r, samp_g) = exorings.make_ring_grad_line(samp_t, res[0], res[1], res[2], res[3])
hjd_minr = samp_t[np.argmin(samp_g)]

exorings.print_disk_parameters(res, hjd_minr, samp_r)

# plotting fit of gradients from ellipse curve to J1407 gradients
plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

figfit = plt.figure(figsize=(10, 6))
f2 = figfit.add_subplot(111)

f2.set_ylim([0, 1.1*np.max(grad_mag)])
f2.set_xlim([np.min(samp_t), np.max(samp_t)])

plot_gradient_fit(grad_time, grad_disk_fit * np.max(grad_mag), grad_mag, \
    samp_t, samp_g*np.max(grad_mag), f2)

# make ticks thicker
for ax in figfit.axes: # go over all the subplots in the figure fig
    for i in ax.spines.itervalues(): # ... and go over all the axes too...
        i.set_linewidth(2)
    ax.minorticks_on() # switch on the minor ticks
    # set the tick lengths and tick widths
    ax.tick_params('both', length=15, width=2, which='major')
    ax.tick_params('both', length=6, width=1, which='minor')

# adjust text size on the axes
f2.tick_params(axis='both', which='major', labelsize=14)

print 'writing plot out to file %s' % plotfileout
plt.savefig(plotfileout)

