import numpy as np
def make_isocosahedron():
    g = (1. + np.sqrt(5)) / 2.  # the golden ratio
    # looking outside the isocosahedron in to the origin
    # the vertices go in an anticlockwise direction around each face
    # and the faces touch each other sequentially
    v = np.array([
        [ 0, 1, g],
        [ 0,-1, g],
        [ 0,-1,-g],
        [ 0, 1,-g],
        [ 1, g, 0],
        [-1, g, 0],
        [-1,-g, 0],
        [ 1,-g, 0],
        [ g, 0, 1],
        [ g, 0,-1],
        [-g, 0,-1],
        [-g, 0, 1]])
    f = np.array([[0,11,1],
        [11,6,1],
        [1,6,7],
        [1,7,8],
        [0,1,8],
        [0,8,4],
        [0,4,5],
        [11,0,5],
        [11,5,10],
        [11,10,6],
        [6,10,2],
        [6,2,7],
        [7,2,9],
        [8,7,9],
        [4,8,9],
        [4,9,3],
        [5,4,3],
        [5,3,10],
        [3,2,10],
        [3,9,2]
        ])
    return(f,v)

def np_to_verts(p):
    # converts numpy array to list suitable for COllection3D
    x = np.ndarray.tolist(p[:,0])
    y = np.ndarray.tolist(p[:,1])
    z = np.ndarray.tolist(p[:,2])
    verts = [zip(x, y, z)]
    return(verts)

def get_vertex_ind(vert, point):
    # see if point is already in vert
    delta = np.sum( np.power(point - vert,2.), axis=1)
    # if delta is nearly zero, get index of that element
    # otherwise push new coordinate onto vert and return that number
    anydup = np.isclose(delta, 0)
    if np.any(anydup):
        # there a duplicate
        return(vert, np.ravel(np.where(anydup)))
    else:
        vert = np.append(vert, [point], axis=0)
        return(vert, vert.shape[0] - 1)

def normalise_vertices(v):
    # normalise to unit length
    vs = v*v
    vssum = np.power(np.sum(vs, axis=1), 0.5)
    vn = v / vssum[:,np.newaxis]
    return(vn)

def normit(x):
    nx = x / np.linalg.norm(x, axis=1)[:,np.newaxis]
    return(nx)


def split_iso(fa, ve):
    # split each triangle face into four sub triangles
    newface = np.empty((0,3), dtype=int)

    for face in fa:
        # generate new points from vertex list
        ip0, ip1, ip2 = face
        p0 = ve[ip0]
        p1 = ve[ip1]
        p2 = ve[ip2]

        # see if this point already exists
        # if it does, return the index of already point
        # if it doesn't, push it on the list and return that

        (ve, ip3) = get_vertex_ind(ve, (p0+p1)/2.)
        (ve, ip4) = get_vertex_ind(ve, (p1+p2)/2.)
        (ve, ip5) = get_vertex_ind(ve, (p2+p0)/2.)

        newface = np.append(newface,np.array([[ip0,ip3,ip5],[ip5,ip3,ip4],[ip3,ip1,ip4],[ip5,ip4,ip2]]), axis=0)

    ven = normalise_vertices(ve)
    return(newface,ven)

def sigmoid(x):
    ' sigmoid - goes from 0 to 1 as you go from -inf to inf'
    result = 1. / (1. + np.exp(-x))
    return(result)

def starspot(cosang, spotsize, angwid):
    # convert to degrees
    angle = np.arccos(cosang) * 180. / np.pi
    tang = (angle - spotsize) / angwid
    return(sigmoid(tang))

def vec2sph(vec):
    theta = np.arcsin(vec[:,0])
    lambd = np.arctan2(vec[:,1], vec[:,0])
    return(theta,lambd)

def sph2vec(theta,lambd):
    xx = np.cos(theta) * np.cos(lambd)
    yy = np.cos(theta) * np.sin(lambd)
    zz = np.sin(theta)
    return(np.hstack((xx,yy,zz)))


def intriangle(tup, p1, p2, p3):
    'tup is a tuple containing x, y p1, p2, p3 are the three points defining the triangle return a tuple of points from inside the triangle'

    x, y = tup

    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]

    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b

    T = (a>=0) & (a<=1) & (b>=0) & (b<=1) & (c>=0) & (c<=1)

    tupout = (x[T], y[T])

    return(tupout)


def triangle_image(tup, p1, p2, p3):
    'mgri is a tuple containing x, y from mgrid p1, p2, p3 are the three points defining the triangle return a boolean array of points from inside the triangle'

    y, x = tup

    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]

    aden = (y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)
    adiv = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    bden = (y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)
    bdiv = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)

    if np.isclose(adiv,0.):
        a = 0.
    else:
        a = aden / adiv

    if np.isclose(bdiv,0.):
        b = 0.
    else:
        b = bden / bdiv


#    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
#    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    c = 1 - a - b

    T = (a>=0) & (a<=1) & (b>=0) & (b<=1) & (c>=0) & (c<=1)

    return(T)

def rotx(theta):
    t = theta * np.pi / 180.
    ct = np.cos(t)
    st = np.sin(t)

    m = np.array([[  1,  0,  0 ],
                  [  0, ct, -st],
                  [  0, st,  ct]])
    return(m)

def roty(theta):
    t = theta * np.pi / 180.
    ct = np.cos(t)
    st = np.sin(t)

    m = np.array([[ ct,  0,  st],
                  [  0,  1,   0],
                  [-st,  0,  ct]])
    return(m)

def rotz(theta):
    t = theta * np.pi / 180.
    ct = np.cos(t)
    st = np.sin(t)

    m = np.array([[ ct,-st,   0],
                  [ st, ct,   0],
                  [  0,  0,   1]])
    return(m)



if __name__ == '__main__':

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('image', interpolation='nearest', origin='lower', cmap='gray')
    mpl.rc('axes.formatter', limits=(-7,7))


    fig = plt.figure()
    ax = Axes3D(fig)

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
    ax.scatter(ve[:,0], ve[:,1], ve[:,2])

    #facols = np.ones_like(fa3) * np.random.random(fa3.shape)
    ve3=normalise_vertices(ve3)
    fa3_mean = np.mean((ve3[fa3]),axis=1)
    facols_3 = np.abs(fa3_mean)

    ve4=normalise_vertices(ve4)
    fa4_mean = np.mean((ve4[fa4]),axis=1)
    facols_4 = np.abs(fa4_mean)
    #ax.scatter(fa3_mean[:,0], fa3_mean[:,1], fa3_mean[:,2])

    ve5=normalise_vertices(ve5)
    fa5_mean = np.mean((ve5[fa5]),axis=1)
    facols_5 = np.abs(fa5_mean)
    #ax.scatter(fa3_mean[:,0], fa3_mean[:,1], fa3_mean[:,2])


    # plot the faces
    for (face, facol) in zip(fa4, facols_4):
        ax.add_collection3d(Poly3DCollection(np_to_verts(ve4[face]),color=facol))

    # label the vertices
    #for n,j in enumerate(ve3):
    #    ax.text(j[0], j[1], j[2], n, fontsize=30, \
    #        color='red', zorder=50, ha='center', va='center')

    # testing intriangle()
#    fig = plt.figure()
#
#    x = np.random.random(10000)
#    y = np.random.random(10000)
#
#    p1 = np.random.random(2)
#    p2 = np.random.random(2)
#    p3 = np.random.random(2)
#
#    plt.scatter(x,y, color='blue')
#    (xt,yt) = intriangle( (x,y), p1, p2, p3)
#    plt.scatter(xt,yt,color='red')
#    plt.show()

    # END testing intriangle()

    # make starspots

    # calculate
    bdir1 = np.array([[1,1,1]])
    bdir2 = np.array([[0,-1,0]])
    n_bdir1 = normit(bdir1)
    n_bdir2 = normit(bdir2)
    n_fa5_mean = normit(fa5_mean)

    # angle = arccos( dot(a, b) )
    ang1 = np.dot(n_fa5_mean, n_bdir1.T)
    ang2 = np.dot(n_fa5_mean, n_bdir2.T)

    cspot = starspot(ang1, 30., 5.) * starspot(ang2, 20, 1.)

    # method using plot+trisurf - it's fast, but no color control
#    ax.plot_trisurf(ve5[:,0], ve5[:,1], ve5[:,2], triangles=fa5, color ='orange', cmap='rainbow')

#    fig = plt.figure()
#    ax = Axes3D(fig)

#    for (face, facol) in zip(fa5, np.hstack((cspot, cspot, cspot)) ):
#        ax.add_collection3d(Poly3DCollection(np_to_verts(ve5[face]),color=facol))


    # make a 2D array and fill it to the corners


    def fillstar2d(im, tup, face, vert, tricol):
        'do a 2d projection onto an image xy plane of 3d triangles'
        for (fa, col) in zip(face, tricol):
            intri = triangle_image(tup, vert[fa[0]], vert[fa[1]], vert[fa[2]])
            im[intri] = im[intri] + col
        return(im)

    ty, tx = np.mgrid[-1.2:1.2:0.01,-1.2:1.2:0.01]

    # set up an image
    im0 = np.zeros((ty.shape[0],ty.shape[1],3))
    intens = facols_3

    # select faces which are positive in the z axis
    # select faces which are positive in the z axis
    # select faces which are positive in the z axis
#    zplus = (fa3_mean[:,2]>0)
#
#    fa3_zplus = fa3[zplus]
#
#    f = fillstar2d(im0, (ty, tx), fa3_zplus, ve3, intens[zplus])
#
#    fig = plt.figure()
#    ax = plt.subplot(111)
#    plt.imshow(f)
    # END select faces which are positive in the z axis
    # END select faces which are positive in the z axis
    # END select faces which are positive in the z axis

    # 3d plot hemisphere of positive z values
    # 3d plot hemisphere of positive z values
    # 3d plot hemisphere of positive z values
#    fig = plt.figure()
#    ax = Axes3D(fig)
#
#    for (face, facol) in zip(fa3_zplus, facols_3[zplus]):
#        ax.add_collection3d(Poly3DCollection(np_to_verts(ve3[face]),color=facol))
#
#    plt.show()
    # END 3d plot hemisphere of positive z values
    # END 3d plot hemisphere of positive z values
    # END 3d plot hemisphere of positive z values

#    fig = plt.figure()
#    ax = plt.subplot(111)
#    # +ve x rotates top edge down towards bottom edge
#    # +ve y moves lhs side of star to the right
#    # +ve z rotates the star surface anticlockwise
#
#    for i in np.linspace(0,180,10):
##
#        im0 = np.zeros((ty.shape[0],ty.shape[1],3))
#        # stack matrices in reverse order
#        Matrix = np.dot(rotx(30),roty(i))
#        ve3rot = (np.dot(Matrix, ve3.T)).T
#
#        fa3_mean = np.mean((ve3rot[fa3]),axis=1)
#
#        # select faces which are positive in the z axis
#        zplus = (fa3_mean[:,2]>0)
#        fa3_zplus = fa3[zplus]
#
#        f = fillstar2d(im0, (ty, tx), fa3_zplus, ve3rot, intens[zplus])
#        plt.imshow(f)
#        plt.draw()
#        raw_input('press return to continue')
#

    fig = plt.figure()
    ax = plt.subplot(111)
    # +ve x rotates top edge down towards bottom edge
    # +ve y moves lhs side of star to the right
    # +ve z rotates the star surface anticlockwise
    ty, tx = np.mgrid[-1.2:1.2:0.02,-1.2:1.2:0.02]

    # set up an image
    im0 = np.zeros((ty.shape[0],ty.shape[1],3))
    intens = facols_3


    for (i, ni) in enumerate(np.linspace(0,360,360)):

        im0 = np.zeros((ty.shape[0],ty.shape[1],3))
        # stack matrices in reverse order
        Matrix = np.dot(rotx(30),roty(i))
        ve5rot = (np.dot(Matrix, ve5.T)).T

        fa5_mean = np.mean((ve5rot[fa5]),axis=1)

        # select faces which are positive in the z axis
        zplus = (fa5_mean[:,2]>0)
        fa5_zplus = fa5[zplus]

        f = fillstar2d(im0, (ty, tx), fa5_zplus, ve5rot, cspot[zplus])
        plt.imshow(f)
 #       plt.draw()
        # write out image
        fname = '/Users/kenworthy/_tmp%04d.png' % i
        print 'Saving frame', fname
        fig.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')


# mencoder 'mf://_tmp*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=ljpeg -oac copy -o animation.mpg

# QuickTime movie
# ffmpeg -r 25 -vsync 1 -i _tmp%04d.png -f mp4 -qscale 5 -vcodec libx264 -pix_fmt yuv420p animation.mp4


