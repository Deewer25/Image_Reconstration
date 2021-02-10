#test №1

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







#test №2

while i < 256:
    while j < 256:
        pix[i][j] = 255
        j += 2

    i += 1
    j = 0

i = 0
j = 0

while i < 256:
    while j < 256:
        pix[i][j] = 255
        j += 1

    i += 2
    j = 0











#test №3

while i < 255:
    while j < 255:
        pix[i][j] = 255
        pix[i][j+1] = 255
        pix[i][j-1] = 255
        j += 4

    i += 1
    j = 1

i = 1
j = 1

while i < 255:
    while j < 255:
        pix[i][j] = 255
        pix[i+1][j] = 255
        pix[i-1][j] = 255
        j += 1

    i += 4
    j = 1


#test №4

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
