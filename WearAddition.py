import cv2
import numpy as np
import random
import os
filepath = '14.jpg'
dir_name, full_file_name = os.path.split(filepath)
file_name, file_ext = os.path.splitext(full_file_name)
img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
'''img1 = cv2.imread('11.jpg')
for i in range(150,200):
    for j in range(2*i-100,2*i):
        img1[i,j]=[255,255,255]
cv2.imshow('img1',img1)
cv2.waitKey()'''
edge_img=cv2.Canny(img,300,400)
cv2.imshow('edges',edge_img)
mask=np.zeros_like(edge_img)
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
print((right_line))
img1 = cv2.imread(filepath)
'''l1=round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])
l2=710'''
k0=(left_line[0][1]-left_line[1][1])/(left_line[0][0]-left_line[1][0])
b=left_line[0][1]-k0*left_line[0][0]
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
for i0 in range(left_line[1][1]-10,719):
    'for j in range(round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])-20,left_line[1][0]+10):'
    if round((i0 - b) / k0 - 30) > 1:
        for j0 in range(round((i0-b)/k0-30),round((i0-b)/k0+30)):
            listall.append([i0,j0])
            b00=img1[i0,j0][0]
            g00=img1[i0,j0][1]
            r00=img1[i0,j0][2]
            if (max(b00,g00,r00)-min(b00,g00,r00)<50 and min(b00,g00,r00)<210):
                listroad.append([i0,j0])
    else:
        for j0 in range(1,round((i0-b)/k0+30)):
            listall.append([i0, j0])
            b00 = img1[i0, j0][0]
            g00 = img1[i0, j0][1]
            r00 = img1[i0, j0][2]
            if (max(b00, g00, r00) - min(b00, g00, r00) < 50 and min(b00, g00, r00) < 210):
                listroad.append([i0, j0])

listline0=[]  #ROI中车道线像素坐标
for [ii,jj] in listall:
    if [ii,jj] not in listroad:
        listline0.append([ii,jj])
print((len(listline0)))
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
for i in range(left_line[1][1]-10,719):
    if round((i0 - b) / k0 - 30) > 1:
        'for j in range(round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])-20,left_line[1][0]+10):'
        for j in range(round((i-b)/k0-30),round((i-b)/k0+30)):
            if ((lpxs[0]-70<=img1[i,j][0]<=lpxs[0]+70) and (lpxs[1]-70<=img1[i,j][1]<=lpxs[1]+70) and (lpxs[2]-70<=img1[i,j][2]<=lpxs[2]+70) and (max(img1[i,j][0],img1[i,j][1],img1[i,j][2])-min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>30 or min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>210)):
                listline.append([i,j])
    else:
        for j in range(1,round((i-b)/k0+30)):
            if ((lpxs[0]-70<=img1[i,j][0]<=lpxs[0]+70) and (lpxs[1]-70<=img1[i,j][1]<=lpxs[1]+70) and (lpxs[2]-70<=img1[i,j][2]<=lpxs[2]+70) and (max(img1[i,j][0],img1[i,j][1],img1[i,j][2])-min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>30 or min(img1[i,j][0],img1[i,j][1],img1[i,j][2])>210)):
                listline.append([i,j])
n1=len(listline)
print(n1)
nkey=0.85
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
    while 0 < nk0 <= 1600:
        if nk0>3:
            randlist0 = random.sample(listline_ebb, 1)
            listadd = []
            for ik in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                for jk in range(-int((nk0 ** 0.5) / 2), int((nk0 ** 0.5) / 2)):
                    if [randlist0[0][0] + ik, randlist0[0][1] + jk] in listline_ebb:
                            listadd.append([randlist0[0][0] + ik, randlist0[0][1] + jk])
            print(len(randlist),nk0, len(listline_ebb))
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






k0r = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
br = right_line[0][1] - k0r * right_line[0][0]
b0r = 0
g0r = 0
r0r = 0
nr = 0
lpxr = img1[719, 640]
listallr=[]  #'ROI中所有像素坐标'
listroadr=[]  #'ROI中道路像素坐标'
for i0r in range(right_line[0][1]-10,719):
    if round((i0r - br) / k0r + 30) < 1280:
        'for j in range(round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])-20,left_line[1][0]+10):'
        for j0r in range(round((i0r-br)/k0r-30),round((i0r-br)/k0r+30)):
                listallr.append([i0r,j0r])
                b00r=img1[i0r,j0r][0]
                g00r=img1[i0r,j0r][1]
                r00r=img1[i0r,j0r][2]
                if (max(b00r,g00r,r00r)-min(b00r,g00r,r00r)<50 and min(b00r,g00r,r00r)<210):
                    listroadr.append([i0r,j0r])
    else:
        for j0r in range(round((i0r-br)/k0r-30),1279):
                listallr.append([i0r,j0r])
                b00r=img1[i0r,j0r][0]
                g00r=img1[i0r,j0r][1]
                r00r=img1[i0r,j0r][2]
                if (max(b00r,g00r,r00r)-min(b00r,g00r,r00r)<50 and min(b00r,g00r,r00r)<210):
                    listroadr.append([i0r,j0r])

listline0r=[]  #ROI中车道线像素坐标
for [iir,jjr] in listallr:
    if [iir,jjr] not in listroadr:
        listline0r.append([iir,jjr])
print((len(listline0r)))
for x0r in range(0,len(listline0r)-1):
    [isiter,jsiter] = listline0r[x0r]
    b0r = b0r + img1[isiter,jsiter][0]
    g0r = g0r + img1[isiter,jsiter][1]
    r0r = r0r + img1[isiter,jsiter][2]
b0r = round(b0r / len(listline0r))
g0r = round(g0r / len(listline0r))
r0r = round(r0r / len(listline0r))
lpxsr=[b0r,g0r,r0r]
print(lpxsr)
'''l1=round((left_line[0][1]+left_line[1][1])/2)
l2=round((left_line[0][0]+left_line[1][0])/2)'''
listliner=[]
for ir in range(right_line[0][1]-10,719):
    if round((i0r - br) / k0r + 30) < 1280:
        'for j in range(round((710-left_line[0][1])/(left_line[1][1]-left_line[0][1])*(left_line[0][1]-left_line[0][0])+left_line[0][0])-20,left_line[1][0]+10):'
        for jr in range(round((ir-br)/k0r-30),round((ir-br)/k0r+30)):
            if img1[ir,jr][0]>200 and img1[ir,jr][1]>200 and img1[ir,jr][2]>200:
                listliner.append([ir,jr])
            else:
                if ((lpxsr[0]-50<=img1[ir,jr][0]<=lpxsr[0]+50) and (lpxsr[1]-50<=img1[ir,jr][1]<=lpxsr[1]+50) and (lpxsr[2]-50<=img1[ir,jr][2]<=lpxsr[2]+50) and (max(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])-min(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])>30 or min(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])>210)):
                    listliner.append([ir,jr])
    else:
        for jr in range(round((ir-br)/k0r-30),1279):
            if img1[ir,jr][0]>200 and img1[ir,jr][1]>200 and img1[ir,jr][2]>200:
                listliner.append([ir,jr])
            else:
                if ((lpxsr[0]-50<=img1[ir,jr][0]<=lpxsr[0]+50) and (lpxsr[1]-50<=img1[ir,jr][1]<=lpxsr[1]+50) and (lpxsr[2]-50<=img1[ir,jr][2]<=lpxsr[2]+50) and (max(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])-min(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])>30 or min(img1[ir,jr][0],img1[ir,jr][1],img1[ir,jr][2])>210)):
                    listliner.append([ir,jr])

n1r=len(listliner)
print(n1r)
n1r=round(n1r*nkey)
ckr=4
if ckr == 0:
    rlistr = random.sample(listliner, n1r)
    for kr in range(0,n1r-1):
        b0kr = 0
        g0kr = 0
        r0kr = 0
        nkr = 0
        for [iikr, jjkr] in listroadr:
            if iikr == rlistr[kr][0]:
                nkr += 1
                b0kr += img1[iikr, jjkr][0]
                g0kr += img1[iikr, jjkr][1]
                r0kr += img1[iikr, jjkr][2]
        if nkr !=0 :
            b0kr = round(b0kr / nkr)
            g0kr = round(g0kr / nkr)
            r0kr = round(r0kr / nkr)
            img1[rlistr[kr][0],rlistr[kr][1]]=[b0kr,g0kr,r0kr]
if ckr == 1:
    rlistr = random.sample(listliner, round(n1r/5))
    for kr in range(0,round(n1r/5-1)):
        b0kr = 0
        g0kr = 0
        r0kr = 0
        nkr = 0
        for [iikr, jjkr] in listroadr:
            if iikr == rlistr[kr][0]:
                nkr += 1
                b0kr += img1[iikr, jjkr][0]
                g0kr += img1[iikr, jjkr][1]
                r0kr += img1[iikr, jjkr][2]
        if nkr != 0:
            b0kr = round(b0kr / nkr)
            g0kr = round(g0kr / nkr)
            r0kr = round(r0kr / nkr)
            img1[rlistr[kr][0],rlistr[kr][1]]=[b0kr,g0kr,r0kr]
            img1[rlistr[kr][0]-1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1]-1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1]+1] = [b0kr, g0kr, r0kr]
if ckr == 2:
    rlistr = random.sample(listliner, round(n1r / 9))
    for kr in range(0, round(n1r / 9 - 1)):
        b0kr = 0
        g0kr = 0
        r0kr = 0
        nkr = 0
        for [iikr, jjkr] in listroadr:
            if iikr == rlistr[kr][0]:
                nkr += 1
                b0kr += img1[iikr, jjkr][0]
                g0kr += img1[iikr, jjkr][1]
                r0kr += img1[iikr, jjkr][2]
        if nkr != 0:
            b0kr = round(b0kr / nkr)
            g0kr = round(g0kr / nkr)
            r0kr = round(r0kr / nkr)
            img1[rlistr[kr][0], rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1]-1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1]+1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1]-1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1]+1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
'''if ckr == 3:
    rlistr = random.sample(listliner, round(n1r / 25))
    for kr in range(0, round(n1r / 25 - 1)):
        b0kr = 0
        g0kr = 0
        r0kr = 0
        nkr = 0
        for [iikr, jjkr] in listroadr:
            if iikr == rlistr[kr][0]:
                nkr += 1
                b0kr += img1[iikr, jjkr][0]
                g0kr += img1[iikr, jjkr][1]
                r0kr += img1[iikr, jjkr][2]
        if nkr != 0:
            b0kr = round(b0kr / nkr)
            g0kr = round(g0kr / nkr)
            r0kr = round(r0kr / nkr)
            img1[rlistr[kr][0], rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] - 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0], rlistr[kr][1] + 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1] - 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 1, rlistr[kr][1] + 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1] - 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 1, rlistr[kr][1] + 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 2, rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 2, rlistr[kr][1] - 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 2, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 2, rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] + 2, rlistr[kr][1] + 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 2, rlistr[kr][1] - 2] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 2, rlistr[kr][1] - 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 2, rlistr[kr][1]] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 2, rlistr[kr][1] + 1] = [b0kr, g0kr, r0kr]
            img1[rlistr[kr][0] - 2, rlistr[kr][1] + 2] = [b0kr, g0kr, r0kr]'''
if ckr == 4:
    n0=0
    listliner_ebb=listliner
    listarear=[]
    nkr0=n1r
    print(nkr0)
    while nkr0 > 1600:
        randlistr0=random.sample(listliner_ebb,1)
        listaddr=[]
        for ikr in range(-20,20):
            for jkr in range(-20,20):
                if [randlistr0[0][0]+ikr,randlistr0[0][1]+jkr] in listliner_ebb:
                        listaddr.append([randlistr0[0][0] + ikr, randlistr0[0][1] + jkr])
        if len(listaddr) > 0:
            randr = random.randint(1, len(listaddr))
            nkr0 -= randr
            randlistr=random.sample(listaddr,randr)
            for [irandr,jrandr] in randlistr:
                listliner_ebb.remove([irandr,jrandr])
                n0 +=1
            print(len(randlistr),nkr0, len(listliner_ebb))
            for kr in range(0,len(randlistr)):
                b0kr = 0
                g0kr = 0
                r0kr = 0
                nkr = 0
                for [iikr, jjkr] in listroadr:
                    if iikr == randlistr[kr][0]:
                        nkr += 1
                        b0kr += img1[iikr, jjkr][0]
                        g0kr += img1[iikr, jjkr][1]
                        r0kr += img1[iikr, jjkr][2]
                b0kr = round(b0kr / nkr)
                g0kr = round(g0kr / nkr)
                r0kr = round(r0kr / nkr)
                img1[randlistr[kr][0], randlistr[kr][1]] = [b0kr, g0kr, r0kr]
    while 0 < nkr0 <= 1600:
        if nkr0>3:
            randlistr0 = random.sample(listliner_ebb, 1)
            listaddr = []
            for ikr in range(-int((nkr0 ** 0.5) / 2), int((nkr0 ** 0.5) / 2)):
                for jkr in range(-int((nkr0 ** 0.5) / 2), int((nkr0 ** 0.5) / 2)):
                    if [randlistr0[0][0] + ikr, randlistr0[0][1] + jkr] in listliner_ebb:
                            listaddr.append([randlistr0[0][0] + ikr, randlistr0[0][1] + jkr])
            print(len(randlistr),nkr0, len(listliner_ebb))
            if nkr0 >= len(listaddr) and len(listaddr)!=0:
                nkr0 -= len(listaddr)
                randlistr = random.sample(listaddr, len(listaddr))
                for [irandr, jrandr] in randlistr:
                    listliner_ebb.remove([irandr, jrandr])
                    n0 += 1
                for kr in range(0, len(randlistr)):
                    b0kr = 0
                    g0kr = 0
                    r0kr = 0
                    nkr = 0
                    for [iikr, jjkr] in listroadr:
                        if iikr == randlistr[kr][0]:
                            nkr += 1
                            b0kr += img1[iikr, jjkr][0]
                            g0kr += img1[iikr, jjkr][1]
                            r0kr += img1[iikr, jjkr][2]
                    b0kr = round(b0kr / nkr)
                    g0kr = round(g0kr / nkr)
                    r0kr = round(r0kr / nkr)
                    img1[randlistr[kr][0], randlistr[kr][1]] = [b0kr, g0kr, r0kr]
            '''if nkr0 < len(listaddr):
                randlistr = random.sample(listaddr, nkr0)
                for [irandr, jrandr] in randlistr:
                    listliner_ebb.remove([irandr, jrandr])
                    n += 1
                for kr in range(0, len(randlistr)):
                    b0kr = 0
                    g0kr = 0
                    r0kr = 0
                    nkr = 0
                    for [iikr, jjkr] in listroadr:
                        if iikr == randlistr[kr][0]:
                            nkr += 1
                            b0kr += img1[iikr, jjkr][0]
                            g0kr += img1[iikr, jjkr][1]
                            r0kr += img1[iikr, jjkr][2]
                    b0kr = round(b0kr / nkr)
                    g0kr = round(g0kr / nkr)
                    r0kr = round(r0kr / nkr)
                    img1[randlistr[kr][0], randlistr[kr][1]] = [b0kr, g0kr, r0kr]'''
        if nkr0<=3:
            randlistr = random.sample(listliner_ebb, nkr0)
            for [irandr, jrandr] in randlistr:
                listliner_ebb.remove([irandr, jrandr])
                n0 += 1
            print(len(randlistr),nkr0,len(listliner_ebb))
            for kr in range(0, len(randlistr)):
                b0kr = 0
                g0kr = 0
                r0kr = 0
                nkr = 0
                for [iikr, jjkr] in listroadr:
                    if iikr == randlistr[kr][0]:
                        nkr += 1
                        b0kr += img1[iikr, jjkr][0]
                        g0kr += img1[iikr, jjkr][1]
                        r0kr += img1[iikr, jjkr][2]
                b0kr = round(b0kr / nkr)
                g0kr = round(g0kr / nkr)
                r0kr = round(r0kr / nkr)
                img1[randlistr[kr][0], randlistr[kr][1]] = [b0kr, g0kr, r0kr]
            nkr0=0
    print(n0)















cv2.imshow('img1',img1)
imgc=cv2.imread(filepath,cv2.IMREAD_COLOR)
cv2.line(imgc,tuple(left_line[0]),tuple(left_line[1]),color=(0,255,255),thickness=3)
cv2.line(imgc,tuple(right_line[0]),tuple(right_line[1]),color=(0,255,255),thickness=3)
cv2.imshow('masked',masked)
cv2.imshow('img',imgc)
cv2.waitKey(0)
if cv2.waitKey(0) == 115:
    cv2.imwrite(file_name+'_new.jpg', img1)