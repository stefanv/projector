import urllib2
import matplotlib.pyplot as plt
from PIL import Image
from StringIO import StringIO
import numpy as np
import time

import scikits.image.io as sio

def get_image():
    f = urllib2.urlopen("http://localhost:8080/?action=snapshot")
    data = f.read()
    pil_image = Image.open(StringIO(data))
    return np.array(pil_image)

time.sleep(5)

img = get_image()
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
sio.imsave('/tmp/output.png', img)

plt.show()
