# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
im1 = Image.open('/Users/albert/PycharmProjects/test/ImColl/bellucci.jpg')
im1.save('bel.jpg')
im = Image.open('bel.jpg')
im = im.convert('L')
im.save('belorig.jpg')
pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))
print(pix[1][1])
print(pix.ndim)


# %%

# pix = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))


# %%
b = 0
while b < 256:
    for a in pix[b]:
        if a > 0:
            a -= 1
    b += 1


# %%
def distance(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(1/2)


# %%
AQQURACY = 10000


# %%
#test
i = 100

while i < 200:
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


# %%
f_y = np.zeros((256, 256))
f_x = np.zeros((256, 256))


# %%
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

        if (pix[i][j+1] != 255) and (pix[i][j-1] != 255):
            f_x[i][j] = (pix[i][j+1] - pix[i][j-1])/2
        elif (pix[i][j+1] == 255) and (pix[i][j-1] != 255):
            f_x[i][j] = pix[i][j] - pix[i][j-1]
        elif (pix[i][j+1] != 255) and (pix[i][j-1] == 255):
            f_x[i][j] = pix[i][j+1] - pix[i][j]
        else:
            f_x[i][j] = 255



        if (pix[i+1][j] != 255) and (pix[i-1][j] != 255):
            f_y[i][j] = (pix[i+1][j] - pix[i-1][j])/2
        elif (pix[i+1][j] == 255) and (pix[i-1][j] != 255):
            f_y[i][j] = pix[i][j] - pix[i-1][j]
        elif (pix[i+1][j] != 255) and (pix[i-1][j] == 255):
            f_y[i][j] = pix[i+1][j] - pix[i][j]
        else:
            f_y[i][j] = 255

        j += 1

    i += 1
    j = 1


# %%
i = 1
j = 1
p = np.zeros((256, 256))

# 

while i < 255:
    while j < 255:
        if f_y[i][j] != 0:
            p[i][j] = float('{:.2f}'.format(- math.atan(f_x[i][j]/f_y[i][j])))
        else:
            p[i][j] = - math.pi/2   #Тут мы доопределяем неопределенность типа 0/0 значением, равным pi/2 (нуждает в уточнении)

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
X, Y = np.meshgrid(x, y) # `plot_surface` expects `x` and `y` data to be 2D 
ha.plot_surface(X, Y, p) 


# %%
NewPix = np.zeros((256, 256, 100001))
ccc = 0
i = 2
j = 2

while i < 255:
    while j < 255:
        if(p[i][j] < 200):
            NewPix[i][j][int((AQQURACY * p[i][j]) + 5*AQQURACY)]
        # while k < 5*AQQURACY:
        #     if k == (AQQURACY * p[i][j]):
                
        #         NewPix[i][j][k+5*AQQURACY] = pix[i][j] 
               

        #     else:
        #         NewPix[i][j][k+5*AQQURACY] = 256
                
            
        #     k += 1

        j += 1
        # k = -5*AQQURACY
        
    i += 1
    j = 1






# %%
MainSq = np.zeros((4,2), dtype=int)

MainSq[0] = [1, 4]
print(NewPix[252][105])


# %%

# Поиск столбцов

i = 1
j = 1
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
k = -50
x = 1

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

                MainSq[0] = [i + x, j] 
                x = 1




                while i - x >= 0:  # поиск вверх
                    if pix[i - x][j] == 255.0:
                        x += 1
                    else:
                        break

                if i - x < 0:
                    x -= 1

                MainSq[1] = [i - x, j]
                x = 1

                while j + x < 256:  # поиск вправо
                    if pix[i][j + x] == 255.0:
                        x += 1
                    else:
                        break

                if j + x >= 256:
                    x -= 1

                MainSq[2] = [i, j + x]
                x = 1

                while j - x >= 0:  # поиск влево
                    if pix[i][j - x] == 255.0:
                        x += 1
                    else:
                        break

                if j - x < 0:
                    x -= 1

                MainSq[3] = [i, j - x]

                x = 1

                

                a = min(p[MainSq[0][0]][MainSq[0][1]], p[MainSq[1][0]][MainSq[1][1]], p[MainSq[2][0]][MainSq[2][1]], p[MainSq[3][0]][MainSq[3][1]])
                b = max(p[MainSq[0][0]][MainSq[0][1]], p[MainSq[1][0]][MainSq[1][1]], p[MainSq[2][0]][MainSq[2][1]], p[MainSq[3][0]][MainSq[3][1]])
                k = a * AQQURACY
                k = int(k)
                while k <= b:
                    dist0 = distance(i, j, k + 5*AQQURACY, MainSq[0][0], MainSq[0][1], (int(p[MainSq[0][0]][MainSq[0][1]] * AQQURACY)))
                    dist1 = distance(i, j, k + 5*AQQURACY, MainSq[1][0], MainSq[1][1], (int(p[MainSq[1][0]][MainSq[1][1]] * AQQURACY)))
                    dist2 = distance(i, j, k + 5*AQQURACY, MainSq[2][0], MainSq[2][1], (int(p[MainSq[2][0]][MainSq[2][1]] * AQQURACY)))
                    dist3 = distance(i, j, k + 5*AQQURACY, MainSq[3][0], MainSq[3][1], (int(p[MainSq[3][0]][MainSq[3][1]] * AQQURACY)))


                    NewPix[i][j][k + 5*AQQURACY] = pix[MainSq[0][0]][MainSq[0][1]] * (dist1 + dist2 + dist3) + pix[MainSq[1][0]][MainSq[1][1]] * (dist0 + dist2 + dist3) + pix[MainSq[2][0]][MainSq[2][1]] * (dist1 + dist0 + dist3) + pix[MainSq[3][0]][MainSq[3][1]] * (dist1 + dist2 + dist0)

                    NewPix[i][j][k + 5*AQQURACY] = NewPix[i][j][k + 5*AQQURACY] / (3 * (dist0 + dist1 + dist2 + dist3))
                    

                    k += 1
### ТУТ НУЖНО НЕ ЗАБЫТЬ ОБНУЛИТЬ K ДА И ВООБЩЕ КРЕПКО ПОДУМАТЬ О ЖИЗНИ
                damaged_pix -= 1
                pixcopy[i][j] = 255.0
                for it in NewPix[i][j]:
                    if ((it < pixcopy[i][j]) and (it > 0)):
                        pixcopy[i][j] = it


                
            j += 1
        i += 1
        j = 1
    pix = pixcopy.copy()
    i = 1







                #            sum = (left * rstep + right * lstep)/(lstep + rstep) + (up * dstep + down * ustep)/(dstep + ustep)


                # Radius = 8


                # if ustep * dstep != 0:
                #     if ustep + dstep < 2*Radius:
                #         sum1 = ((up * dstep) + (down * ustep)) / (dstep + ustep)
                #         sum = sum1
                #     else:
                #         sum1 = 0
                # else:
                #     sum1 = 0


                # if lstep * rstep != 0:
                #     if lstep + rstep < 2*Radius:
                #         sum2 = ((left * rstep) + (right * lstep)) / (lstep + rstep)
                #         if (lstep + rstep) < (ustep + dstep):
                #             sum = sum2
                #     else: sum2 = 0
                # else:
                #     sum2 = 0




                    # if (sum1 == 0) | (sum2 == 0):
                    #     k = 1
                    # else:
                    #     k = 2
                    #
                    # pixcopy[i][j] = (sum1+sum2)/k   #тут основная проблема. Суммы нужно выбирать тоже с разными коэффициентами



            #     if sum != 0:
            #         pixcopy[i][j] = sum
            #         damaged_pix -= 1
            # sum1 = 0
            # sum2 = 0
            # sum = 0
            # down = 0
            # up = 0
            # left = 0
            # right = 0
            


# %%
print(NewPix[132][130])


# %%
print(pix[132][130])


# %%
pix = np.asarray(pix, dtype=np.uint8)

ImRes = Image.fromarray(pix, mode='L')

ImRes.save('resbel.jpg')


# %%



