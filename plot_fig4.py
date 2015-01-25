import sys, getopt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from astropy.io import ascii
from scipy.interpolate import interp1d

import exorings

# set sensible imshow defaults
mpl.rc('image', interpolation='nearest', origin='lower', cmap='gray')

# no scientific notation for numbers on plots
mpl.rc('axes.formatter', limits=(-7, 7))

# use latex for labelling
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

# load in J1407 binned photometry curve
tin = ascii.read("j1407_bincc.dat")
time = tin['time']
flux = tin['flux']
flux_err = tin['flux_rms']

# 54160 to 54300
goodp = (time > 54160) * (time < 54300)

flux_err = flux_err[goodp]
flux = flux[goodp]
time = time[goodp]

print 'number of photometric points: %d' % time.size

vstar = -1.
try:
    opts, args = getopt.getopt(sys.argv[1:], "hr:o:s:", ["rfile=", "ofile=", "vstar="])
except getopt.GetoptError:
    print '%s -s <velocity> -r <inputfile> -o <outputfile>' % sys.argv[0]
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
        vstar = np.array(float(arg))

print 'ring file in  is %s' % fitsin
print 'plot file out is %s' % plotout

(res, taun_rings, rad_rings, dstar) = exorings.read_ring_fits(fitsin)

exorings.print_ring_tau(rad_rings, exorings.y_to_tau(taun_rings))

# set up stellar disk
kern = exorings.make_star_limbd(21, 0.8)

# produce fine grained gradient and ring values
samp_t = np.arange(-100, 100, 0.001) + 54222.
(samp_r, samp_g) = exorings.ring_grad_line(samp_t, res[0], res[1], res[2], res[3])
hjd_minr = samp_t[np.argmin(samp_g)]

hjd_to_ring = interp1d(samp_t, samp_r, kind='linear')

sst = exorings.print_disk_parameters(res, hjd_minr, samp_r)

## Calculate the best model fit given the rings and disk parameters
strip, convo, g = exorings.ellipse_strip(rad_rings, exorings.y_to_tau(taun_rings), \
    res[0], res[1], res[2], res[3], kern, dstar)

fit_time = g[0]
fit_flux = g[1]

### BEGIN THE PLOT ##################################################

datacolor = 'red'
modelcolor = 'green'

eb = dict(fmt='.', color=datacolor, ecolor=datacolor, capsize=0.0, \
marker='o', mfc=datacolor, mec=datacolor, ms=3, mew=0.001, \
elinewidth=0.5)

smalleb = dict(fmt='o', color='white', ecolor=datacolor, capsize=0.0, \
marker='o', mfc='white', mec=datacolor, ms=4, mew=1, elinewidth=2.0)

mdict = dict(color=modelcolor, zorder=-5)

ty = dict(color='black', fontsize=10, fontweight='bold', va='top', ha='right')

# set up plot area
fig = plt.figure(figsize=(10, 12))

# split into two panels - the top with the light curve and model
# fit and the bottom with the zoomed in plots

gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], wspace=0.0, hspace=0.05)

ax1 = plt.subplot(gs[0, :])

# the J1407 photometry
ax1.errorbar(time, flux, flux_err, zorder=-4, **eb)

# the ring model
ax1.plot(fit_time, fit_flux, **mdict)

ax1.axis((54180., 54260, 0., 1.19))
ax1.ticklabel_format(style='plain', useOffset=False, axis='x', scilimits=(-5, 10))
ax1.set_xlabel("Time [days]")
ax1.xaxis.set_label_position('top')
ax1.xaxis.tick_top()

# the vertical line marking t_tangential
ax1.vlines(hjd_minr, -1., 2., colors='k', linestyle='dashed')

# array of days that we want a zoom into
dt = np.array((-23, -22, -17, -16, -15, -14, -11, -10, -9, -8, -7, -6, \
+3, +5, +9, +10, +11, +24))
dt += 1.

# disk parameters as a latex table
ax1.text(0.17, 0.60, sst, transform=ax1.transAxes, **ty)

# xdet and ydet are the sizes of the zoomed boxes
ep_zoom = 0.5
y_zoom = 0.4
fiddle_time = 0.3

# number of plots in the grid
nx = 6
ny = dt.size/nx

og = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs[1], wspace=0.0, hspace=0.0)

for i in xrange(dt.size):
    print "image %d " % i
    ep_center = hjd_minr + dt[i] + fiddle_time
    ax = plt.subplot(og[i])

#    ax.errorbar(time,flux, flux_err, zorder=-4, **eb)

    # first select all the pixels in that day range
    # then centroid on that subset of pixels with the zoomed box
    ep_day = (time < (ep_center+0.5)) * (time > (ep_center-0.5))
    time_day = time[ep_day]
    flux_day = flux[ep_day]
    flux_err_day = flux_err[ep_day]

    ax.errorbar(time_day, flux_day, flux_err_day, zorder=-3, **smalleb)
    # the ring model
    ax.plot(fit_time, fit_flux, linewidth=3, **mdict)

    # get the center of the box from the median values of the selected
    # day
    day_center = np.median(time_day)
    y_center = (np.max(flux_day) + np.min(flux_day))/2.

    # label the top plot with a marker
    ax1.scatter(day_center, 1.05, marker='v', color='k')

    # corners of the zoomed box
    ep_low = day_center - (ep_zoom/2.)
    ep_hig = day_center + (ep_zoom/2.)
    flux_low = y_center - (y_zoom/2.)
    flux_hig = y_center + (y_zoom/2.)

    #ax1.add_patch(Rectangle((ep_low, flux_low), ep_zoom, y_zoom, facecolor="grey",zorder=-10,linewidth=0))

    if i == 0:
        ax1.add_patch(Rectangle((ep_low, flux_low), ep_zoom, y_zoom, facecolor="none", zorder=-10, linewidth=1))
        ax.text(0.1, 0.1, r'$\rm{width}=%4.2f$\ \rm{d}' % ep_zoom, transform=ax.transAxes)
        ax.text(0.1, 0.22, r'$\rm{height}=%4.2f$\ \rm{T}' % y_zoom, transform=ax.transAxes)

    ax.axis((ep_low, ep_hig, flux_low, flux_hig))

    # label the delta day
    ax.text(0.95, 0.95, dt[i], transform=ax.transAxes, **ty)

    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(plotout)

