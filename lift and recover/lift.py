from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

im1 = Image.open('/Users/albert/PycharmProjects/test/ImColl/bellucci.jpg')
im1.save('bel.jpg')
im = Image.open('bel.jpg')
im = im.convert('L')
im.save('belorig.jpg')
pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))
print(pix[1][1])
print(pix.ndim)


b = 0
while b < 256:
    for a in pix[b]:
        if a > 0:
            a -= 1
    b += 1


# test
i = 1

while i < 256:
    # pix[142][i] = 255
    # pix[143][i] = 255
    # pix[144][i] = 255
    # pix[145][i] = 255
    # pix[146][i] = 255
    # pix[147][i] = 255
    # pix[148][i] = 255
    # pix[149][i] = 255
    pix[132][i] = 255
    pix[133][i] = 255
    pix[134][i] = 255
    pix[135][i] = 255
    pix[136][i] = 255
    pix[137][i] = 255
    pix[138][i] = 255
    pix[139][i] = 255
    # pix[122][i] = 255
    # pix[123][i] = 255
    # pix[124][i] = 255
    # pix[125][i] = 255
    # pix[126][i] = 255
    # pix[127][i] = 255
    # pix[128][i] = 255
    # pix[129][i] = 255
    # pix[112][i] = 255
    # pix[113][i] = 255
    # pix[114][i] = 255
    # pix[115][i] = 255
    # pix[116][i] = 255
    # pix[117][i] = 255
    # pix[118][i] = 255
    # pix[119][i] = 255
    i += 1


f_y = np.zeros((256, 256))
f_x = np.zeros((256, 256))


i = 1
j = 1

# Differentiation

while i < 255:
    while j < 255:
        if pix[i][j] == 255:
            f_x[i][j] = 255
            f_y[i][j] = 255
            j += 1
            continue

        if (pix[i][j + 1] != 255) and (pix[i][j - 1] != 255):
            f_x[i][j] = (pix[i][j + 1] - pix[i][j - 1]) / 2
        elif (pix[i][j + 1] == 255) and (pix[i][j - 1] != 255):
            f_x[i][j] = pix[i][j] - pix[i][j - 1]
        elif (pix[i][j + 1] != 255) and (pix[i][j - 1] == 255):
            f_x[i][j] = pix[i][j + 1] - pix[i][j]
        else:
            f_x[i][j] = 255

        if (pix[i + 1][j] != 255) and (pix[i - 1][j] != 255):
            f_y[i][j] = (pix[i + 1][j] - pix[i - 1][j]) / 2
        elif (pix[i + 1][j] == 255) and (pix[i - 1][j] != 255):
            f_y[i][j] = pix[i][j] - pix[i - 1][j]
        elif (pix[i + 1][j] != 255) and (pix[i - 1][j] == 255):
            f_y[i][j] = pix[i + 1][j] - pix[i][j]
        else:
            f_y[i][j] = 255

        j += 1

    i += 1
    j = 1


i = 1
j = 1
p = np.zeros((256, 256))

#

while i < 255:
    while j < 255:
        if f_y[i][j] != 0:
            p[i][j] = float('{:.3f}'.format(- math.atan(f_x[i][j] / f_y[i][j])))
        else:
            p[i][
                j] = - math.pi / 2  # Тут мы доопределяем неопределенность типа 0/0 значением, равным pi/2 (нуждает в уточнении)

        if f_y[i][j] == 255:
            p[i][j] = 255

        j += 1
    i += 1
    j = 1

nx, ny = 256, 256
x = range(nx)
y = range(ny)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, p)

NewPix = np.zeros((256, 256, 10001))

i = 1
j = 1
k = -5000
while i < 255:
    while j < 255:
        while k < 5000:
            if k == (1000 * p[i][j]):
                NewPix[i][j][k + 5000] = pix[i][j]
            else:
                NewPix[i][j][k + 5000] = 256

            k += 1

        j += 1
    i += 1

print(NewPix)
