import cv2
import numpy as np
import random
import os
filepath = '62.jpg'
dir_name, full_file_name = os.path.split(filepath)
file_name, file_ext = os.path.splitext(full_file_name)
img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
'''img1 = cv2.imread('11.jpg')
for i in range(150,200):
    for j in range(2*i-100,2*i):
        img1[i,j]=[255,255,255]
cv2.imshow('img1',img1)
cv2.waitKey()
edge_img=cv2.Canny(img,300,400)
cv2.imshow('edges',edge_img)
mask=np.zeros_like(edge_img)s
mask=cv2.fillPoly(mask,np.array([[[100,720],[520,260],[1000,260],[1280,720]]]),color=255)
masked=cv2.bitwise_and(edge_img,mask)
lines=cv2.HoughLinesP(masked,1,np.pi/180,15,minLineLength=40,maxLineGap=20)
def calculate_slope(line):
    k = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
    return k
left_lines=[line for line in lines if calculate_slope(line)<0]
right_lines=[line for line in lines if calculate_slope(line)>0]
print(len(lines))
print(len(left_lines))
print(len(right_lines))
def ral(lines,thres):
    slopes=[calculate_slope(line) for line in lines]
    while len(lines):
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx]>thres:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines
ral(left_lines,thres=0.2)
ral(right_lines,thres=0.2)
print((left_lines))
print((right_lines))
def least_squares_fit(lines):
    xc=np.ravel([[line[0][0],line[0][2]] for line in lines])
    yc=np.ravel([[line[0][1],line[0][3]] for line in lines])
    poly=np.polyfit(xc,yc,deg=1)
    point_min=(np.min(xc),np.polyval(poly,np.min(xc)))
    point_max=(np.max(xc),np.polyval(poly,np.max(xc)))
    return np.array([point_min,point_max], dtype=np.int32)
left_line=(least_squares_fit(left_lines))
right_line=(least_squares_fit(right_lines))
print((left_line))
print((right_line))'''
img1 = cv2.imread(filepath)
'''l1=round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])
l2=710'''
'''k0=(left_line[0][1]-left_line[1][1])/(left_line[0][0]-left_line[1][0])
b=left_line[0][1]-k0*left_line[0][0]'''
b0=0
g0=0
r0=0
n=0
lpx=img1[719,640]
print(lpx)
'''for i0 in range(left_line[1][1],left_line[0][1]):
    'for j in range(round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])-20,left_line[1][0]+10):'
    j0=round((i0-b)/k0)
    n += 1
    b0 = b0 + img1[j0, i0][0]
    g0 = g0 + img1[j0, i0][1]
    r0 = r0 + img1[j0, i0][2]
lpxs=[round(b0/n),round(g0/n),round(r0/n)]'''
listall=[]  #'ROI中所有像素坐标'
listroad=[]  #'ROI中道路像素坐标'
listline0=[]  #ROI中车道线像素坐标
b1=720
k1=(260-b1)/580
k2=460/(1280-750)
b2=260-750*k2

for i0 in range(260,720):
    for j0 in range(round((i0-b1)/k1),round((i0-b2)/k2)):
        listall.append([i0, j0])
        b00 = img1[i0, j0][0]
        g00 = img1[i0, j0][1]
        r00 = img1[i0, j0][2]
        if (max(b00, g00, r00) - min(b00, g00, r00) < 50 and min(b00, g00, r00) < 210):
            listroad.append([i0, j0])
        if (max(b00, g00, r00) - min(b00, g00, r00) >= 50 or min(b00, g00, r00) >= 210):
            listline0.append([i0,j0])
print(len(listall),len(listroad),len(listline0))
for x0 in range(0,len(listline0)):
    [isite,jsite] = listline0[x0]
    b0 = b0 + img1[isite,jsite][0]
    g0 = g0 + img1[isite,jsite][1]
    r0 = r0 + img1[isite,jsite][2]
b0 = round(b0 / len(listline0))
g0 = round(g0 / len(listline0))
r0 = round(r0 / len(listline0))
lpxs=[b0,g0,r0]
print(lpxs)
'''l1=round((left_line[0][1]+left_line[1][1])/2)
l2=round((left_line[0][0]+left_line[1][0])/2)'''
listline=[]
for i in range(260,720):
    for j in range(round((i-b1)/k1),round((i-b2)/k2)):
         'if ((lpxs[0]-70<=img1[i,j][0]<=lpxs[0]+70) and (lpxs[1]-70<=img1[i,j][1]<=lpxs[1]+70) and (lpxs[2]-70<=img1[i,j][2]<=lpxs[2]+70) and (max(img1[i,j][0],img1[i,j][1],img1[i,j][2])-min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>30 or min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>210)):'
         if (max(img1[i,j][0],img1[i,j][1],img1[i,j][2])-min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>30 or min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>210):
             listline.append([i,j])
n1=len(listline)
print(n1)
nkey=1
n1=round(n1*nkey)
ck=4
if ck == 0:
    rlist=random.sample(listline,n1)
    for k in range(0,n1-1):
        b0k=0
        g0k=0
        r0k=0
        nk=0
        for [iik,jjk] in listroad:
            if iik == rlist[k][0]:
                nk += 1
                b0k += img1[iik,jjk][0]
                g0k += img1[iik, jjk][1]
                r0k += img1[iik, jjk][2]
        b0k = round(b0k/nk)
        g0k = round(g0k / nk)
        r0k = round(r0k / nk)
        img1[rlist[k][0],rlist[k][1]]=[b0k,g0k,r0k]
if ck == 1:
    rlist = random.sample(listline, round(n1/9))
    for k in range(0,round(n1/9)-1):
        b0k=0
        g0k=0
        r0k=0
        nk=0
        for [iik,jjk] in listroad:
            if iik == rlist[k][0]:
                nk += 1
                b0k += img1[iik,jjk][0]
                g0k += img1[iik, jjk][1]
                r0k += img1[iik, jjk][2]
        b0k = round(b0k/nk)
        g0k = round(g0k / nk)
        r0k = round(r0k / nk)
        img1[rlist[k][0],rlist[k][1]]=[b0k,g0k,r0k]
        img1[rlist[k][0]-1, rlist[k][1]] = [b0k, g0k, r0k]
        img1[rlist[k][0]-1, rlist[k][1]-1] = [b0k, g0k, r0k]
        img1[rlist[k][0]-1, rlist[k][1]+1] = [b0k, g0k, r0k]
        img1[rlist[k][0], rlist[k][1]-1] = [b0k, g0k, r0k]
        img1[rlist[k][0], rlist[k][1]+1] = [b0k, g0k, r0k]
        img1[rlist[k][0]+1, rlist[k][1]-1] = [b0k, g0k, r0k]
        img1[rlist[k][0]+1, rlist[k][1]] = [b0k, g0k, r0k]
        img1[rlist[k][0]+1, rlist[k][1]+1] = [b0k, g0k, r0k]
if ck == 4:
    '''for i4 in range(260,720):
        print(len(listroad))
        locali4=[]
        left=[]
        right=[]
        for j4 in range(round((i4-b1)/k1),round((i4-b2)/k2)):
            if ([i4, j4] in listline0) and j4<=640:
                left.append(j4)
            if ([i4, j4] in listline0) and j4>640:
                right.append(j4)
        if len(left)!=0 and len(right)!=0:
            lmin=min(left)
            lmax=max(left)
            rmin=min(right)
            rmax=max(right)'''


    '''if rmin-lmax>40:
                for i4v in range(i4,720):
                    for j4v in range(lmax,rmin):
                        if [i4v,j4v] in listroad:
                            listroad.remove([i4v,j4v])
        print(len(listroad))'''


    '''for j4v in range(round((i4 - b1) / k1), round((i4 - b2) / k2)):
            if [i4,j4v] in listroad and (j4v<lmin-20 or lmax+20<j4v<rmin-20 or j4v>rmax+20):
                listroad.remove([i4,j4v])'''





    n0=0
    listline_ebb=listline
    listarea=[]
    nk0=n1
    print(nk0)
    while nk0 > 1600:
        randlist0=random.sample(listline_ebb,1)
        listadd=[]
        for ik in range(-20,20):
            for jk in range(-20,20):
                if [randlist0[0][0]+ik,randlist0[0][1]+jk] in listline_ebb:
                    listadd.append([randlist0[0][0] + ik, randlist0[0][1] + jk])
        if len(listadd) > 0:
            rand = random.randint(1, len(listadd))
            nk0 -= rand
            randlist=random.sample(listadd,rand)
            for [irand,jrand] in randlist:
                listline_ebb.remove([irand,jrand])
                n0 +=1
            print(len(randlist),nk0, len(listline_ebb))
            for k in range(0,len(randlist)):
                b0k = 0
                g0k = 0
                r0k = 0
                nk = 0
                'listx=[]'
                for [iik, jjk] in listroad:
                    if iik == randlist[k][0]:
                        'if 1:'
                        if randlist[k][1]-30<jjk<randlist[k][1]+30:
                            nk += 1
                            b0k += img1[iik, jjk][0]
                            g0k += img1[iik, jjk][1]
                            r0k += img1[iik, jjk][2]
                if nk!=0:
                    b0k = round(b0k / nk)
                    g0k = round(g0k / nk)
                    r0k = round(r0k / nk)
                    img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
    while 0 < nk0 <= 1600:
        if nk0>3:
            randlist0 = random.sample(listline_ebb, 1)
            listadd = []
            for ik in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                for jk in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                    if [randlist0[0][0] + ik, randlist0[0][1] + jk] in listline_ebb:
                            listadd.append([randlist0[0][0] + ik, randlist0[0][1] + jk])
            print(len(randlist0),nk0, len(listline_ebb))
            if nk0 >= len(listadd) and len(listadd)!=0:
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
                    for [iik, jjk] in listroad:
                        if iik == randlist[k][0]:
                            nk += 1
                            b0k += img1[iik, jjk][0]
                            g0k += img1[iik, jjk][1]
                            r0k += img1[iik, jjk][2]
                    b0k = round(b0k / nk)
                    g0k = round(g0k / nk)
                    r0k = round(r0k / nk)
                    img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
        if nk0<=3:
            randlist = random.sample(listline_ebb, nk0)
            for [irand, jrand] in randlist:
                listline_ebb.remove([irand, jrand])
                n0 += 1
            print(len(randlist),nk0,len(listline_ebb))
            for k in range(0, len(randlist)):
                b0k = 0
                g0k = 0
                r0k = 0
                nk = 0
                for [iik, jjk] in listroad:
                    if iik == randlist[k][0]:
                        nk += 1
                        b0k += img1[iik, jjk][0]
                        g0k += img1[iik, jjk][1]
                        r0k += img1[iik, jjk][2]
                b0k = round(b0k / nk)
                g0k = round(g0k / nk)
                r0k = round(r0k / nk)
                img1[randlist[k][0], randlist[k][1]] = [b0k, g0k, r0k]
            nk0=0
    print(n0)






'''k0r = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
br = right_line[0][1] - k0r * right_line[0][0]'''
b0r = 0
g0r = 0
r0r = 0
nr = 0














cv2.imshow('img1',img1)
imgc=cv2.imread(filepath,cv2.IMREAD_COLOR)
'''cv2.line(imgc,tuple(left_line[0]),tuple(left_line[1]),color=(0,255,255),thickness=3)
cv2.line(imgc,tuple(right_line[0]),tuple(right_line[1]),color=(0,255,255),thickness=3)
cv2.imshow('masked',masked)'''
cv2.imshow('img',imgc)
cv2.waitKey(0)
if cv2.waitKey(0) == 115:
    cv2.imwrite(file_name+'_new.jpg', img1)
