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

#Усовершенственный t3, который будет работать и для лево-право



i = 1
j = 1
b = 0
while b < 256:
    for a in pix[b]:
        if a > 0:
            a -= 1
    b += 1

#test
mask = Image.open('ImColl/eye-corr-white.jpg')
print(mask.mode)
mask = mask.convert('L')
mpix = np.asarray(mask.getdata(), dtype=np.float64).reshape((mask.size[1], mask.size[0]))


while i < 255:
    while j < 255:
        if mpix[i][j] > 247:
            pix[i][j] = 255
        j += 1

    i += 1
    j = 0




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

k = 0

ustep = 0
dstep = 0
lstep = 0
rstep = 0

damaged_pix = 0

while i < 255:
    while j < 255:
        if pix[i][j] == 255.0:
            damaged_pix += 1
        j += 1
    i += 1
    j = 1

i = 1
j = 1

pixcopy = pix.copy()

while damaged_pix > 0:

    while i < 255:
        while j < 255:
            if pix[i][j] == 255.0:  # если нашел поврежденный пиксель

                while i + x < 256:  # поиск вниз
                    if pix[i + x][j] == 255.0:
                        x += 1
                    else:
                        break

                if i + x >= 256:
                    x -= 1
                count = 1

                for a in [pix[i + x][j]]:
                    if a == 255:
                        count -= 1
                    else:
                        down += a

                if count == 0:
                    down = 0
                    dstep = 0
                else:
                    down = down / count
                    dstep = x

                x = 1

                while i - x >= 0:  # поиск вверх
                    if pix[i - x][j] == 255.0:
                        x += 1
                    else:
                        break

                if i - x < 0:
                    x -= 1

                count = 1

                for a in [pix[i - x][j]]:
                    if a == 255:
                        count -= 1
                    else:
                        up += a

                if count == 0:
                    up = 0
                    ustep = 0
                else:
                    up = up / count
                    ustep = x

                x = 1

                while j + x < 256:  # поиск вправо
                    if pix[i][j + x] == 255.0:
                        x += 1
                    else:
                        break

                if j + x >= 256:
                    x -= 1

                count = 1

                for a in [pix[i][j + x]]:
                    if a == 255:
                        count -= 1
                    else:
                        right += a

                if count == 0:
                    right = 0
                    rstep = 0
                else:
                    right = right / count
                    rstep = x

                x = 1

                while j - x >= 0:  # поиск влево
                    if pix[i][j - x] == 255.0:
                        x += 1
                    else:
                        break

                if j - x < 0:
                    x -= 1

                count = 1

                for a in [pix[i][j - x]]:
                    if a == 255:
                        count -= 1
                    else:
                        left += a

                if count == 0:
                    left = 0
                    lstep = 0
                else:
                    left = left / count
                    lstep = x

                x = 1

                #            sum = (left * rstep + right * lstep)/(lstep + rstep) + (up * dstep + down * ustep)/(dstep + ustep)


                Radius = 8


                if ustep * dstep != 0:
                    if ustep + dstep < 2*Radius:
                        sum1 = ((up * dstep) + (down * ustep)) / (dstep + ustep)
                        sum = sum1
                    else:
                        sum1 = 0
                else:
                    sum1 = 0


                if lstep * rstep != 0:
                    if lstep + rstep < 2*Radius:
                        sum2 = ((left * rstep) + (right * lstep)) / (lstep + rstep)
                        if (lstep + rstep) < (ustep + dstep):
                            sum = sum2
                    else: sum2 = 0
                else:
                    sum2 = 0




                    # if (sum1 == 0) | (sum2 == 0):
                    #     k = 1
                    # else:
                    #     k = 2
                    #
                    # pixcopy[i][j] = (sum1+sum2)/k   #тут основная проблема. Суммы нужно выбирать тоже с разными коэффициентами
                if sum != 0:
                    pixcopy[i][j] = sum
                    damaged_pix -= 1
            sum1 = 0
            sum2 = 0
            sum = 0
            down = 0
            up = 0
            left = 0
            right = 0
            j += 1
        i += 1
        j = 1
    pix = pixcopy.copy()
    i = 1

pix = np.asarray(pix, dtype=np.uint8)

ImRes = Image.fromarray(pix, mode='L')
#ImRes.show()
ImRes.save('resbelSQ2.jpg')