""" makes animation of star moving behind ring system.
    python ring_animation_press_release.py -r 54220.65.try9.33kms.fits -s 33000.
    ffmpeg -r 25 -vsync 1 -i _tmp%04d.png -f mp4 -qscale 5 -vcodec libx264 -pix_fmt yuv420p animation.mp4
"""
import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
import exorings
import matplotlib as mpl
import j1407
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

from scipy.interpolate import InterpolatedUnivariateSpline

from astropy.time import Time

# set numbers to not go to sci notation
# and make sure that no offset (e.g. +2.71e5) appears on axes
mpl.rc('axes.formatter', useoffset=False, limits=(-7, 9))

# set sensible imshow defaults
mpl.rc('image', interpolation='nearest', origin='lower', cmap='gray')

# read in j1407 photometry
(time, flux, flux_err) = j1407.j1407_photom_binned('j1407_bincc.dat', 54160.0, 54300.0)
print ('number of photometric points: %d' % time.size)

# get j1407 gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

# parse command line options
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

# smoothed curve useful for animation tracking
splfunc = UnivariateSpline(time, flux, s=5)

# make the plots
ringcolor = 'orange'
bgndcolor = 'black'

fig_ringsv = plt.figure('ringsv', figsize=(10, 7))

fig_ringsv.patch.set_facecolor(bgndcolor)

p1v = plt.axes([0.0, 0.3, 1.0, 0.7], axisbg=bgndcolor)
#p1v.axis('scaled')
# why am I not using this 'scaled' to make the rings have the correct
# aspect ratio?
# it turns out that if you use 'scaled' the values reported from
# p1v.axis() are not updated at all - they stick with 'default' values
# of -50 to +50. So, to draw the clock below, I cannot use transform()
# functions and instead rely on getting the correct aspect ratio as
# reported for the p1v.transAxes below.....

# find out pixel values for the ring window
x0, y0 = p1v.transAxes.transform((0, 0)) # lower left in pixels
x1, y1 = p1v.transAxes.transform((1, 1)) # upper right in pixes
dx = x1 - x0
dy = y1 - y0
xtoy_scale = dx/dy

p2v = plt.axes([0.1, 0.05, 0.87, 0.25], sharex=p1v, axisbg=bgndcolor)
files = []

i = 0
nframes = 3

timeframes = np.linspace(54192.56, 54262, nframes)
#timeframes = np.linspace(54192.56, 54193.56, nframes)

fps = 25 # frames per second in final animation
# xps = 3.0 day / second

x_time = np.array([  54190., 54195, 54196., 54203.5, 54204.5, 54209.0, 54210.0, 54230, 54232., 54250.0])
xps    = np.array([     1.0,    1.0,    3.0,    3.0,     0.5,     0.5,     3.0,   3.0,    2.0,     2.0]) / fps
zoom   = np.array([      10,     10,     10,     10,       3,       3,      10,    10,     30,      30])
alpt   = np.array([     1.0,    1.0,     0.,     0.,      0.,      0.,      0.,    0.,     0.,      0.])

# make functions that return xps and zoom for a given x_time
fxps = InterpolatedUnivariateSpline(x_time, xps, k=1)
fzoom = InterpolatedUnivariateSpline(x_time, zoom, k=1)
falpt = InterpolatedUnivariateSpline(x_time, alpt, k=1)

# walk along from first value to last value
frame = x_time[0]
p_curr = x_time[0]

while p_curr < x_time[-1]:
    frame = np.append(frame, p_curr)
    p_curr += fxps(p_curr)

frame_zoom = fzoom(frame)
frame_alpt = falpt(frame)

# generating 'fade in and out' for text

# ring gap
x_ti   = np.array([ 54190., 54203.5, 54204.5,  54206.0, 54206.5 , 54250.0])
alph1  = np.array([    0.0,     0.0,     1.0,      1.0,       0.,      0.])
falph1 = InterpolatedUnivariateSpline(x_ti, alph1, k=1)
frame_alph1 = falph1(frame)

# missing data
x_ti   = np.array([ 54190., 54217. , 54218., 54225., 54226., 54250.0])
alph2  = np.array([    0.0,     0.0,    1.0,    1.0,     0.,      0.])
falph2 = InterpolatedUnivariateSpline(x_ti, alph2, k=1)
frame_alph2 = falph2(frame)

print ('%d frames to generate' % frame.size)

for now_hjd, now_zoom in zip(frame, frame_zoom):
    p1v.cla()
    p2v.cla()

    exorings.draw_rings_vector(rad_rings, exorings.y_to_tau(taun_rings), \
        res[1], res[2], res[3], p1v, ringcolor)
    #    if (badrings):
    #draw_badrings(rstart, rend, res[1], res[2], res[3], p1v)

    # plot J1407b
    circ2 = plt.Circle((res[1], 0), radius=0.1, color='brown', zorder = 12)
    p1v.add_patch(circ2)
    p1v.text(res[1], 0.5, 'J1407b', zorder=12, color='black', fontsize=14, \
        fontweight='bold', va='bottom', ha='center')

    # draw star
    circ = plt.Circle((now_hjd, res[0]), radius=dstar/2., color='yellow', zorder = -20)
    p1v.add_patch(circ)

    # setting the zoom of the plot
    # this makes sure that p1v is centred on the star impact parameter
    p1v.set_xlim(now_hjd - now_zoom, now_hjd + now_zoom)
    now_y_zoom = now_zoom / xtoy_scale
    p1v.set_ylim(res[0] - now_y_zoom, res[0] + now_y_zoom)

    # remove the axis box
    p1v.set_axis_off()

    strip, convo, g = exorings.ellipse_strip(rad_rings, exorings.y_to_tau(taun_rings), \
        res[0], res[1], res[2], res[3], kern, dstar)

    # error bars on the photometry
    p2v.errorbar(time, flux, flux_err, fmt='o', ecolor='r', capsize=0, \
        elinewidth=5, zorder=10)
    p2v.scatter(time, flux, marker='o', edgecolor='none', s=40, \
        facecolor='yellow', zorder=20)
    p2v.scatter(time, flux, marker='o', edgecolor='none', s=4, \
        facecolor='black', zorder=20)

    # make the frames white
    for child in p2v.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color('white')

    p2v.tick_params(axis='both', colors='white')

    daynumber = np.floor(now_hjd)
    dayfrac = now_hjd - daynumber

    t = Time(daynumber, format='mjd', scale='utc')

    outt = Time(t, format='iso', out_subfmt='date')

    tyb = dict(color='white', fontsize=16, fontweight='bold', va='center', \
        ha='left')

    # text relative to plot box
    p1v.text(0.78, 0.85, outt, \
        transform=p1v.transAxes, **tyb)

    fit_time = g[0]
    fit_flux = g[1]

    p2v.plot(fit_time[(fit_time < now_hjd)], \
        fit_flux[(fit_time < now_hjd)], \
        linewidth=5, color=ringcolor)

    # now vertically zoom the lower plot centred on the splined fit

    (x1, x2, y1, y2) = p2v.axis()

    dy = 0.5
    p2v.axis((x1, x2, -0.05, 1.1))
    p2v.ticklabel_format(axis='both', scilimits=(-2, 9))
    p2v.text(0.95, 0.05, 'Modified Julian Date [days]', \
        transform=p2v.transAxes, \
        color='white', fontsize=12, fontweight='bold', va='bottom', \
        ha='right')

    # make ticks thicker
    for isaa  in p2v.spines.itervalues(): # ... and go over all the axes too...
        isaa.set_linewidth(2)
    # set the tick lengths and tick widths
    p2v.tick_params('both', length=5, width=2, which='major')

    p2v.set_ylabel('Star Brightness',color='white', fontweight='bold')

    # draw text annotations
    # time clock as a pie chart
    # I want to keep the clock in one part of the screen but with the
    # correct aspect ratio. The ring system coord system has Aspect =
    # 1.0, so if we take a centre point and the 12 o'clock point for the
    # time clock in the display coords, we can then find their positions
    # in the axes coords and draw a circle there.

    clock_centre = p1v.transAxes.transform((0.95,0.85))
    clock_up     = p1v.transAxes.transform((0.95,0.88))
    # now transform to p1v coords
    inv = p1v.transData.inverted()
    clock_cenp = inv.transform(clock_centre)
    clock_upp  = inv.transform(clock_up)
    radius = clock_upp[1] - clock_cenp[1]

    p1v.add_patch(Circle(clock_cenp, radius, color='white') )

    # make the time wedge colour swap with the background colour
    # every other day, so that the colour change at midnight isn't so
    # jarring
    wedcol = 'black'
    if daynumber%2 :
        wedcol = 'white'
        p1v.add_patch(Circle(clock_cenp, radius*0.9, color='black') )
    if dayfrac < 0.25:
        p1v.add_patch(Wedge(clock_cenp, radius * 0.9, 360.*(0.25-dayfrac), 90., \
            color=wedcol) )
    else:
        p1v.add_patch(Wedge(clock_cenp, radius * 0.9, 0., 90., \
            color=wedcol) )
        p1v.add_patch(Wedge(clock_cenp, radius * 0.9, 450.-(360.*dayfrac), 360., \
            color=wedcol) )

    # text animations
    s1 = 'Modelling Giant Extrasolar Ring Systems in Eclipse\n and the Case of J1407b: Sculpting by Exomoons?\n\nM.A. Kenworthy and E.E.  Mamajek\n\nTo be published in\nthe Astrophysical Journal'

    # text relative to plot box
    tyc = dict(color='white', fontsize=16, fontweight='bold', va='center', \
        ha='center', alpha = frame_alpt[i])

    p1v.text(0.5, 0.81, s1, \
        transform=p1v.transAxes, **tyc)

    p1v.text(0.5, 0.45, 'The star J1407', \
        transform=p1v.transAxes, **tyc)

    p2v.text(54187, 0.75, 'Yellow with black dots are measurements of J1407', \
       color='white', fontsize=12, fontweight='bold', alpha=frame_alpt[i] )

    p2v.text(54204.9, 0.2, 'Ring gap?', \
       color='white', fontsize=12, ha='center', fontweight='bold', alpha=frame_alph1[i] )
    p2v.arrow(54204.9, 0.4, 0., 0.2, \
        head_width=0.1, head_length=0.1, width=0.02, \
        fc='white', ec='white', alpha=frame_alph1[i])

    p2v.text(54220.0, 0.6, 'No data taken', \
       color='white', fontsize=12, fontweight='bold', alpha=frame_alph2[i] )

    p1v.text(0.7, 0.1, 'Julian Days', \
        transform=p2v.transAxes, **tyc)
    # write out image
    fname = '_tmp%04d.png' % i
    print ('Saving frame', fname)
    fig_ringsv.savefig(fname, facecolor=fig_ringsv.get_facecolor(), edgecolor='none')
    files.append(fname)
    i += 1

#print 'Making movie animation.mpg - this make take a while'
#os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=25 -ovc lavc -lavcopts vcodec=ljpeg -oac copy -o animation.mpg")
# mencoder 'mf://_tmp*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=ljpeg -oac copy -o animation.mpg

# QuickTime movie
# ffmpeg -r 25 -vsync 1 -i _tmp%04d.png -f mp4 -qscale 5 -vcodec libx264 -pix_fmt yuv420p animation.mp4
