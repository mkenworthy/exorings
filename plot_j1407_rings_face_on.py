import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import exorings3 as exorings
import j1407

# read in j1407 photometry
(time, flux, flux_err) = j1407.j1407_photom_binned('j1407_bincc.dat', 54160.0, 54300.0)
print ('number of photometric points: %d' % time.size)

# get j1407 gradients
(grad_time, grad_mag, grad_mag_norm) = j1407.j1407_gradients('j1407_gradients.txt')

fitsin = '54220.65.try9.33kms.fits'
plotout = 'j1407_rings_face_on.pdf'

print ('Reading in ring and disk parameters from %s' % fitsin)
(res, taun_rings, rad_rings, dstar) = exorings.read_ring_fits(fitsin)

# set up stellar disk
kern = exorings.make_star_limbd(21, 0.8)

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
fig_ringsv = plt.figure('ringsv', figsize=(11, 11))

# rings
p1v = plt.axes([0.0, 0.0, 1.0, 1.0])
p1v.axis('scaled')

res[3]=0.0
res[2]=0.0

# draw the rings
exorings.draw_rings_vector(rad_rings, exorings.y_to_tx(taun_rings), res[1], res[2], res[3], p1v)

# draw the no photometry rings
exorings.draw_badrings(rstart, rend, res[1], res[2], res[3], p1v)

# switch off plot box for rings
p1v.set_axis_off()

(x1, x2, y1, y2) = p1v.axis()
p1v.axis((54190.-30., 54258.+30., y1-30, y2+30))

fig_ringsv.savefig(plotout)
