import urllib2
import matplotlib.pyplot as plt
from PIL import Image
from StringIO import StringIO
import numpy as np

import scikits.image.io as sio
import scikits.image.transform as tf

import time

def get_image():
    f = urllib2.urlopen("http://localhost:8080/?action=snapshot")
    data = f.read()
    pil_image = Image.open(StringIO(data))
    return np.array(pil_image)

print "Display your image"

time.sleep(5)
img = get_image()
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')

plt.suptitle('Dewarping\nSelect 4 points on the left grid, then 4 in the right '
             'image.')

def estimate_homography(sc, tc):
    """
    sc == source coordinates, (x,y)
    tc == target coordinates, (x,y)

    """
    sc = np.asarray(sc)
    tc = np.asarray(tc)

    sx = sc[:, 0]
    sy = sc[:, 1]
    tx = tc[:, 0]
    ty = tc[:, 1]

    A = np.zeros((8, 9))
    A[::2, 0] = -sx
    A[::2, 1] = -sy
    A[::2, 2] = -1
    A[::2, 6] = tx * sx
    A[::2, 7] = tx * sy
    A[::2, 8] = tx

    A[1::2, 3] = -sx
    A[1::2, 4] = -sy
    A[1::2, 5] = -1
    A[1::2, 6] = ty * sx
    A[1::2, 7] = ty * sy
    A[1::2, 8] = ty

    u, s, v = np.linalg.svd(A)
    v = v.T
    h = v[:, -1].reshape((3, 3))

    return h


# Source coordinates
sc = np.array(plt.ginput(n=4))

screen_coords = [(0, 0),
                 (1023, 0),
                 (1023, 767),
                 (0, 767)]

# Homography: source to camera
H_SC = estimate_homography(screen_coords, sc)

tc = np.array(plt.ginput(n=4))
print tc

tc_in_screen = \
      np.dot(np.linalg.inv(H_SC),
             np.hstack((tc, np.ones((4,1)))).T).T
tc_in_screen /= tc_in_screen[:, 2, np.newaxis]

# Screen to screen homography
H_SS = estimate_homography(screen_coords,
                           tc_in_screen)

grid = sio.imread('fatgrid.jpg')
grid_warp = tf.homography(grid, H_SS,
                          output_shape=grid.shape,
                          order=2)

np.save('/tmp/H_SS.npy', H_SS)
sio.imsave('/tmp/grid_warp.png', grid_warp)
