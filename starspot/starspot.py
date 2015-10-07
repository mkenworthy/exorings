import numpy as np
from isocosa import *

import pdb
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', interpolation='nearest', origin='lower', cmap='gray')
mpl.rc('axes.formatter', limits=(-7,7))


(fa, ve) = make_isocosahedron()
ve=normalise_vertices(ve)

print ' number of faces is %d with %d vertices' % (fa.shape[0], ve.shape[0])
(fa2, ve2) = split_iso(fa, ve)
print ' number of faces is %d with %d vertices' % (fa2.shape[0], ve2.shape[0])
(fa3, ve3) = split_iso(fa2, ve2)
print ' number of faces is %d with %d vertices' % (fa3.shape[0], ve3.shape[0])
(fa4, ve4) = split_iso(fa3, ve3)
print ' number of faces is %d with %d vertices' % (fa4.shape[0], ve4.shape[0])
(fa5, ve5) = split_iso(fa4, ve4)
print ' number of faces is %d with %d vertices' % (fa5.shape[0], ve5.shape[0])
# plot the vertices

ve3=normalise_vertices(ve3)
fa3_mean = np.mean((ve3[fa3]),axis=1)
facols_3 = np.abs(fa3_mean)

ve4=normalise_vertices(ve4)
fa4_mean = np.mean((ve4[fa4]),axis=1)
facols_4 = np.abs(fa4_mean)

ve5=normalise_vertices(ve5)
fa5_mean = np.mean((ve5[fa5]),axis=1)
facols_5 = np.abs(fa5_mean)


# make starspots

# calculate positions and sizes
bdir1 = np.array([[1,1,1]])
bdir2 = np.array([[0,1,0]])
n_bdir1 = normit(bdir1)
n_bdir2 = normit(bdir2)
n_fa5_mean = normit(fa5_mean)

# angle = arccos( dot(a, b) )
ang1 = np.dot(n_fa5_mean, n_bdir1.T)
ang2 = np.dot(n_fa5_mean, n_bdir2.T)

cspot = starspot(ang1, 30., 5.) * starspot(ang2, 15, 4.)

def fillstar2d(im, tup, face, vert, tricol):
    'do a 2d projection onto an image xy plane of 3d triangles'
    for (fa, col) in zip(face, tricol):
        # select pixels in a given triangle
        intri = triangle_image(tup, vert[fa[0]], vert[fa[1]], vert[fa[2]])
        # add the colour/intensity to those pixels
        im[intri] = im[intri] + col
    return(im)

ty, tx = np.mgrid[-1.2:1.2:0.01,-1.2:1.2:0.01]

# set up an image
im0 = np.zeros((ty.shape[0],ty.shape[1],3))
intens = facols_3

# coordinates for the image of the star
ty, tx = np.mgrid[-1.1:1.1:0.02,-1.1:1.1:0.02]

# set up an image array for the star
im0 = np.zeros((ty.shape[0],ty.shape[1],3))

# we need an array for the velocity of the star surface
# the intensity radiating in the +ve z direction due to limb darkening

def limb_darken(vect, u=0.85):
    # calculate cos theta
    # theta is angle between vector and z axis
    costheta = np.dot(vect,np.array([0,0,1])) / np.linalg.norm(vect,axis=1)
    I = (1 - u * (1 - costheta))
    return(I)

# we have starspots that are a function of position on the star only
# we have limb darkening which is a function of viewing angle
# we have velocity which is a function of position


def pos2vel(pos):
    # rotation axis for star is y axis
    # so position (x,y,z) has velocity(z,0,-x)
    vel = np.zeros_like(pos)
    vel[:,0] = -pos[:,2]
    vel[:,2] = pos[:,0]
    return(vel)

vel = pos2vel(fa5_mean)


# have r g b image planes


fig = plt.figure()
ax = plt.subplot(111)


# +ve x rotates top edge down towards bottom edge
# +ve y moves lhs side of star to the right
# +ve z rotates the star surface anticlockwise
for (i, ni) in enumerate(np.linspace(0,360,1)):

    im0 = np.zeros((ty.shape[0],ty.shape[1],3))
    # stack matrices in reverse order
    Matrix = np.dot(rotz(30),rotx(40))
    Matrix = np.dot(Matrix,roty(i))
    ve5rot = (np.dot(Matrix, ve5.T)).T

    velrot = (np.dot(Matrix, vel.T)).T

    # rotate the star
    # recalculate the center of all the triangles
    posrot = np.mean((ve5rot[fa5]),axis=1)

    # fa5_mean is the rotated star positions

    # now calculate the limb darkening for these points

    limbd = limb_darken(posrot)

    total_intens = limbd * cspot.ravel()
#    total_vel    = total_intens * velrot

    # select faces which are positive in the z axis
    zplus = (posrot[:,2]>0)
    velplus = velrot[zplus]

    # pull out the radial velocity component
    velz = velrot[:,2]

    # convert - 1 to +1 to a Doppler velocity table
    # v = -1 is red     [1,0,0]
    # v = 0 is white    [1,1,1]
    # v = +1 is blue    [0,0,1]
    vcol_r = (velz < 0) + (velz >=0) * (1 - velz)
    vcol_g = (1. + velz) * (velz < 0) + (velz >=0) * (1 - velz)
    vcol_b = (1. + velz) * (velz < 0) + (velz >=0)

    velcol = np.vstack((vcol_r,vcol_g,vcol_b)).T

    #f = fillstar2d(im0, (ty, tx), fa5[zplus], ve5rot, total_intens[zplus])
   #f = fillstar2d(im0, (ty, tx), fa5[zplus], ve5rot, velplus[:,2])
#    pdb.set_trace()
    f = fillstar2d(im0, (ty, tx), fa5[zplus], ve5rot, velcol[zplus])
    plt.imshow(f)
    plt.draw()
    plt.show()
    # write out image
#    fname = '/Users/kenworthy/_tmp%04d.png' % i
#    print 'Saving frame', fname
#    fig.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')


# mencoder 'mf://_tmp*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=ljpeg -oac copy -o animation.mpg

# QuickTime movie
# ffmpeg -r 25 -vsync 1 -i _tmp%04d.png -f mp4 -qscale 5 -vcodec libx264 -pix_fmt yuv420p animation.mp4


