from PIL import Image
import numpy as np
im1 = Image.open('bricks1.png')
im1.save('br.png')
im = Image.open('br.png')
im = im.convert('L')
im.save('brickOrig.png')
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

# while i < 255:
#     pix[142][i] = 255
#     pix[143][i] = 255
#     pix[144][i] = 255
#     pix[145][i] = 255
#     pix[146][i] = 255
#     pix[147][i] = 255
#     pix[148][i] = 255
#     pix[149][i] = 255
#     pix[132][i] = 255
#     pix[133][i] = 255
#     pix[134][i] = 255
#     pix[135][i] = 255
#     pix[136][i] = 255
#     pix[137][i] = 255
#     pix[138][i] = 255
#     pix[139][i] = 255
#     pix[122][i] = 255
#     pix[123][i] = 255
#     pix[124][i] = 255
#     pix[125][i] = 255
#     pix[126][i] = 255
#     pix[127][i] = 255
#     pix[128][i] = 255
#     pix[129][i] = 255
#     pix[112][i] = 255
#     pix[113][i] = 255
#     pix[114][i] = 255
#     pix[115][i] = 255
#     pix[116][i] = 255
#     pix[117][i] = 255
#     pix[118][i] = 255
#     pix[119][i] = 255
#
#     pix[132][i] = 255
#     pix[133][i] = 255
#     pix[134][i] = 255
#     pix[135][i] = 255
#     pix[136][i] = 255
#     pix[137][i] = 255
#     pix[138][i] = 255
#     pix[139][i] = 255
#     i += 1

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


                Radius = 50


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
ImRes.save('resbrick.png')