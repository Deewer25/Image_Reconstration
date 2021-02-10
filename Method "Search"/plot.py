from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im1 = Image.open('ImColl/bellucci.jpg')
im1.save('bel.jpg')
im = Image.open('bel.jpg')
im = im.convert('L')
im.save('belorig.jpg')
pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))


plt.plot(pix[140])
plt.show()