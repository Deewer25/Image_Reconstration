from PIL import Image
import numpy as np
im1 = Image.open('ImColl/bellucci.jpg')
im1.save('bel.jpg')
im = Image.open('bel.jpg')
im = im.convert('L')
im.save('belorig.jpg')
pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))
print(pix[1][1])
print(pix.ndim)



i = 0
b = 0
while b < 255:
    for a in pix[b]:
        if a > 0:
            a -= 1
    b += 1

while i < 255:
    pix[72][i] = 255
    pix[73][i] = 255
    pix[74][i] = 255
    pix[75][i] = 255
    pix[76][i] = 255
    pix[77][i] = 255
    pix[78][i] = 255
    pix[79][i] = 255
    i += 1

i = 1
j = 1
x = 1
y = 1
count = 0
sum = 0

up = 0
down = 0
left = 0
right = 0

ustep = 0
dstep = 0
lstep = 0
rstep = 0

while i < 255:
    while j < 255:
        if pix[i][j] == 255.0:
            while pix[i + x][j] == 255.0:
                x += 1

            down = pix[i + x][j]
            dstep = x
            x = 1

            while pix[i - x][j] == 255.0:
                x += 1

            up = pix[i - x][j]
            ustep = x
            x = 1

#            while pix[i][j + x] == 255.0:
#                x += 1

#            right = pix[i][j + x]
#            rstep = x
#            x = 1

#            while pix[i][j - x] == 255.0:
#                x += 1

#            left = pix[i][j - x]
#            lstep = x
#            x = 1

#            sum = (left * rstep + right * lstep)/(lstep + rstep) + (up * dstep + down * ustep)/(dstep + ustep)
            sum = (up * dstep + down * ustep)/(dstep + ustep)
            pix[i][j] = sum

        sum = 0
        count = 0
        j += 1
    i += 1
    j = 1


pix = np.asarray(pix, dtype=np.uint8)

ImRes = Image.fromarray(pix, mode='L')
#ImRes.show()
ImRes.save('resbel.jpg')