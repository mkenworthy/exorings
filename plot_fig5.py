import sys, getopt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator

import exorings
import j1407

# no scientific notation for numbers on plots
mpl.rc('axes.formatter', limits=(-7, 7))

# read in j1407 photometry
(time, flux, flux_err) = j1407.j1407_photom_binned('j1407_bincc.dat', 54160.0, 54300.0)
print ('number of photometric points: %d' % time.size)

# get j1407 gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

try:
    opts, args = getopt.getopt(sys.argv[1:], "hr:o:s:", ["rfile=", "ofile=", "vstar="])
except getopt.GetoptError:
    print ('%s -r <inputfile> -s <velocity in metres per second> -o <outputfile>' % sys.argv[0])
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print (help)
        sys.exit()
    elif opt in ("-r", "--rfile"):
        fitsin = arg
    elif opt in ("-o", "--ofile"):
        plotout = arg
    elif opt in ("-s", "--vstar"):
        v = np.array(float(arg))

print ('Reading in ring and disk parameters from %s' % fitsin)
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

# times when there is no photometry
(rstart, rend) = exorings.ring_mask_no_photometry(kern, dstar, time, res[0], res[1], res[2], res[3])
rmask = (rstart < 40)
rstart = rstart[rmask]
rend = rend[rmask]

# print disk parameters
exorings.print_disk_parameters(res, hjd_minr, samp_r)

# start making the figure
fig_ringsv = plt.figure('ringsv', figsize=(11, 7))

# rings
p1v = plt.axes([0.1, 0.3, 0.85, 0.65])
p1v.axis('scaled')

# photometry curve
p2v = plt.axes([0.1, 0.11, 0.85, 0.20], sharex=p1v)

# residuals
p3v = plt.axes([0.1, 0.06, 0.85, 0.05], sharex=p1v)

# draw the rings
exorings.draw_rings_vector(rad_rings, exorings.y_to_tau(taun_rings), res[1], res[2], res[3], p1v)

# draw the no photometry rings
exorings.draw_badrings(rstart, rend, res[1], res[2], res[3], p1v)

# draw the path of the star behind the rings
star_line = mpl.patches.Rectangle((hjd_minr-50, res[0]-dstar/2.), 100, dstar, color='g', zorder=-15)
p1v.add_patch(star_line)

strip, convo, g = exorings.ellipse_strip(rad_rings, exorings.y_to_tau(taun_rings), \
    res[0], res[1], res[2], res[3], kern, dstar)

# error bars on the photometry
eb = dict(fmt='.', color='white', ecolor='red', capsize=0.0, \
    marker='o', mfc='red', mec='red', ms=3, mew=0.001, \
    elinewidth=0.5)
p2v.errorbar(time, flux, flux_err, zorder=10, **eb)

fit_time = g[0]
fit_flux = g[1]

p2v.plot(fit_time, fit_flux, linewidth=1, color='green')

fluxcurve = interp1d(fit_time, fit_flux, kind='linear')
flux_fit = fluxcurve(time)

p3v.errorbar(time, flux-flux_fit, flux_err, zorder=10, **eb)

# adjust the ticks on the photometry plot
for ax in fig_ringsv.axes: # go over all the subplots in the figure fig
    for i in iter(ax.spines.values()): # ... and go over all the axes too...
        i.set_linewidth(2)
    ax.minorticks_on() # switch on the minor ticks
    # set the tick lengths and tick widths
    ax.tick_params('both', length=5, width=2, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')

p3v.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))

# switch off plot box for rings
p1v.set_axis_off()

(x1, x2, y1, y2) = p1v.axis()
p1v.axis((54190., 54258., y1, y2))

# vertically zoom the lower plot centred on the splined fit
(x1, x2, y1, y2) = p2v.axis()
p2v.axis((x1, x2, 0., 1.1))

(x1, x2, y1, y2) = p3v.axis()
p3v.axis((x1, x2, -0.19, 0.19))

majorLocator = MultipleLocator(0.1)
p3v.yaxis.set_major_locator(majorLocator)
p3v.axhline(y=0., color='k', ls='dashed')

p3v.set_xlabel('HJD - 2450000 [Days]')
p3v.set_ylabel('Error')
p2v.set_ylabel('Normalized Intensity')

fig_ringsv.savefig(plotout)
