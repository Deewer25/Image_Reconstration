from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im1 = Image.open('ImColl/bellucci.jpg')
im1.save('bel.jpg')
im = Image.open('bel.jpg')
im = im.convert('L')
im.save('belorig.jpg')
pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))
print(pix[1][1])
print(pix.ndim)

# plt.plot(pix[177])
# plt.show()

i = 0
j = 0
b = 0
while b < 255:
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
count = 0
damaged_pix = 0
sum = 0




while i < 255:
    while j < 255:
        if pix[i][j] == 255.0:
            damaged_pix += 1
        j += 1
    i += 1
    j = 1

i = 1
j = 1



while damaged_pix > 0:
    pixcopy = pix.copy()
    while i < 255:
        while j < 255:
            if pixcopy[i][j] == 255.0:
                for e in [pixcopy[i - 1][j - 1], pixcopy[i - 1][j], pixcopy[i - 1][j + 1], pixcopy[i][j - 1], pixcopy[i][j + 1],
                          pixcopy[i + 1][j - 1], pixcopy[i + 1][j], pixcopy[i + 1][j + 1]]:
                    if e != 255:
                        count += 1
                        sum += e
                if count != 0:
                    pix[i][j] = sum / count
                    damaged_pix -= 1
            sum = 0
            count = 0
            j += 1
        i += 1
        j = 1
    i = 1







# while i < 255:
#     while j < 255:
#         if pix[i][j] == 255.0:
#             for e in [pix[i-1][j-1], pix[i-1][j], pix[i-1][j+1], pix[i][j-1], pix[i][j+1], pix[i+1][j-1], pix[i+1][j], pix[i+1][j+1]]:
#                 if e != 255:
#                     count += 1
#                     sum += e
#             if count != 0:
#                pix[i][j] = sum/count
#         sum = 0
#         count = 0
#         j += 1
#     i += 1
#     j = 1


pix = np.asarray(pix, dtype=np.uint8)

ImRes = Image.fromarray(pix, mode='L')
#ImRes.show()
ImRes.save('resbel.jpg')

# plt.plot(pix[177])
# plt.show()

#im1.save('bel.jpg')
#im1.convert('L').save()
#ar = np.array([1, 2, 3])
#ar = np.array(im)

#print(ar[1][1][5])

#Мылсли:

#Результат зависит от последовательности выполнения усреднения
#Gодружить разные края
#
#
#
#
#
#
#
