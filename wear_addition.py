from matplotlib.path import Path
import cv2
import random
from scipy import io
import os
filepath = 'sg5-31.jpg' #Import image here
dir_name, full_file_name = os.path.split(filepath)
file_name, file_ext = os.path.splitext(full_file_name)
img1=cv2.imread(filepath)
listline=[]
mat=io.loadmat('matlab-sg-5-31.mat') #Import mat file of image here
N=mat['N']
N=int(N)
for i in range(1, N+1): #Steps for enumerating lane markings marked in MATLAB
    exec(f"conc{i} = mat['conc{i}']")
    exec("p{} = Path(conc{})".format(i, i))
    p = eval(f"p{i}")
    for iq in range(0, 900):
        for jq in range(0, 1600):
            if p.contains_points([(iq, jq)]) == True:
                listline.append([iq,jq])
sta = img1[899,800]
n1 = len(listline)
print(n1)
ratio=0.1 #Ratio of wear range from 0 to 1
n1 = round(n1*ratio)
ck = 2  # ck == 1 for regional distribution, ck == 2 for random distribution
n0 = 0
listline_ebb = listline
nk0 = n1
print(nk0)

if ck == 1: #regional distribution
    nleft = n1
    rand0 = random.sample(listline_ebb, 1)  #choose the initial pixel for adding wear randomly
    randa = rand0
    n = 1
    list3 = []
    print(listline_ebb)
    while nleft > 0:
        for [i3, j3] in listline_ebb:  #choosing the pixel of horizontal direction
            if i3 == rand0[0][0] and i3 < 900:
                list3.append([i3, j3])
                nleft -= 1
                print([i3,j3])
        rand0[0][0] = randa[0][0] + n*(-1)**n # choosing the pixel of adjacent rows of initial pixel
        rand0[0][1] = randa[0][1]
        n += 1
        print(n, nleft,len(list3),rand0[0][0])
    for k in range(0, len(list3)):  # Start adding wear
        b0k = 0
        g0k = 0
        r0k = 0
        nk = 0
        for jk in range(list3[k][1] - 30, list3[k][1] + 30):
            if [list3[k][0], jk] not in listline:
                if jk < 1600:  # Preventing randlist[k][1]+30 exceed the size of image
                    bkk = img1[list3[k][0], jk][0]
                    gkk = img1[list3[k][0], jk][1]
                    rkk = img1[list3[k][0], jk][2]
                    if max(bkk, gkk, rkk) - min(bkk, gkk, rkk) < 40 and min(bkk, gkk, rkk) < 210:  # Color sifting
                        b0k += bkk
                        g0k += gkk
                        r0k += rkk
                        nk += 1
        if nk != 0:
            b0k = round(b0k / nk)
            g0k = round(g0k / nk)
            r0k = round(r0k / nk)
            img1[list3[k][0], list3[k][1]] = [b0k, g0k, r0k]
        print(len(list3) - k)

if ck == 2:  #random distribution
    while nk0 > 1600:   #Adding large-size wear
        randlist0 = random.sample(listline_ebb, 1)
        listadd = []
        for ik in range(-7, 7):
            for jk in range(-7, 7):  #Produce pixel area (maximum area is a square consisted by 196 pixels)
                if [randlist0[0][0] + ik, randlist0[0][1] + jk] in listline_ebb:
                    listadd.append([randlist0[0][0] + ik, randlist0[0][1] + jk])
        if len(listadd) > 0:
            rand = random.randint(1, len(listadd))  #Choose size of wear area randomly
            nk0 -= rand
            randlist = random.sample(listadd, rand)
            for [irand, jrand] in randlist:
                listline_ebb.remove([irand, jrand])
                n0 += 1
            print(len(randlist), nk0, len(listline_ebb))
            for k in range(0, len(randlist)):   #Start adding wear
                b0k = 0
                g0k = 0
                r0k = 0
                nk = 0
                'listx=[]'
                for jk in range(randlist[k][1]-30,randlist[k][1]+30):
                    if [randlist[k][0], jk] not in listline:
                        if jk<1600: # Preventing randlist[k][1]+30 exceed the size of image
                            bkk=img1[randlist[k][0], jk][0]
                            gkk=img1[randlist[k][0], jk][1]
                            rkk = img1[randlist[k][0], jk][2]
                            if max(bkk,gkk,rkk)-min(bkk,gkk,rkk)<40 and min(bkk,gkk,rkk)<210: #Color sifting
                                b0k+=bkk
                                g0k+=gkk
                                r0k+=rkk
                                nk+=1
                if nk != 0:
                    b0k = round(b0k / nk)
                    g0k = round(g0k / nk)
                    r0k = round(r0k / nk)
                    img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
    while 0 < nk0 <= 1600:  #Adding medium-size wear
        if nk0 > 3:
            randlist0 = random.sample(listline_ebb, 1)
            listadd = []
            '''for ik in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                for jk in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):'''
            for ik in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):  #Limiting max size of wear to the number of pixel left for coloring
                for jk in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                    if [randlist0[0][0] + ik, randlist0[0][1] + jk] in listline_ebb:
                        listadd.append([randlist0[0][0] + ik, randlist0[0][1] + jk])
            print(len(randlist0), nk0, len(listline_ebb))
            if nk0 >= len(listadd) and len(listadd) != 0:
                nk0 -= len(listadd)
                randlist = random.sample(listadd, len(listadd))
                for [irand, jrand] in randlist:
                    listline_ebb.remove([irand, jrand])
                    n0 += 1
                for k in range(0, len(randlist)):
                    b0k = 0
                    g0k = 0
                    r0k = 0
                    nk = 0
                    for jk in range(randlist[k][1] - 30, randlist[k][1] + 30):
                        if [randlist[k][0], jk] not in listline:
                            if jk < 1600: # Preventing randlist[k][1]+30 exceed the size of image
                                bkk = img1[randlist[k][0], jk][0]
                                gkk = img1[randlist[k][0], jk][1]
                                rkk = img1[randlist[k][0], jk][2]
                                if max(bkk, gkk, rkk) - min(bkk, gkk, rkk) < 40 and min(bkk, gkk, rkk) < 210: #Color sifting
                                    b0k += bkk
                                    g0k += gkk
                                    r0k += rkk
                                    nk += 1
                    if nk!=0:
                        b0k = round(b0k / nk)
                        g0k = round(g0k / nk)
                        r0k = round(r0k / nk)
                    else:  #If reference color is not found, use standard color as replacement
                        b0k=sta[0]
                        g0k=sta[1]
                        r0k=sta[2]
                    img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
        if nk0 <= 3: #Adding last 3 wear
            randlist = random.sample(listline_ebb, nk0)
            for [irand, jrand] in randlist:
                listline_ebb.remove([irand, jrand])
                n0 += 1
            print(len(randlist), nk0, len(listline_ebb))
            for k in range(0, len(randlist)):
                b0k = 0
                g0k = 0
                r0k = 0
                nk = 0
                for jk in range(randlist[k][1] - 30, randlist[k][1] + 30):
                    if [randlist[k][0], jk] not in listline:
                        if jk < 1600: # Preventing randlist[k][1]+30 exceed the size of image
                            bkk = img1[randlist[k][0], jk][0]
                            gkk = img1[randlist[k][0], jk][1]
                            rkk = img1[randlist[k][0], jk][2]
                            if max(bkk, gkk, rkk) - min(bkk, gkk, rkk) < 40 and min(bkk, gkk, rkk) < 210: #Color sifting
                                b0k += bkk
                                g0k += gkk
                                r0k += rkk
                                nk += 1
                if nk != 0:
                    b0k = round(b0k / nk)
                    g0k = round(g0k / nk)
                    r0k = round(r0k / nk)
                else: #If reference color is not found, use standard color as replacement
                    b0k = sta[0]
                    g0k = sta[1]
                    r0k = sta[2]
                img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
            nk0 = 0
print(n0)
cv2.imshow('new.jpg',img1)
cv2.waitKey(0)
if cv2.waitKey(0) == 115:  #Press 's' to save the image
    cv2.imwrite(file_name+'_new.jpg', img1)