import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
from scipy import interpolate

def smooth_plot(x_arr, y_arr):
    fig = plt.figure()   # 创建一个figure
    ax = Subplot(fig, 111)   # 利用Subplot将figure加入ax
    fig.add_axes(ax)
    ax.axis['bottom'].set_axisline_style("->", size=1.5)  # x轴加上箭头
    ax.axis['left'].set_axisline_style("->", size=1.5)  # y轴加上上箭头
    ax.axis['top'].set_visible(False)  # 去除上方坐标轴
    ax.axis['right'].set_visible(False)  # 去除右边坐标轴
    xmin = min(x_arr)
    xmax = max(x_arr)
    xnew = np.arange(xmin, xmax, 0.0005)  # 在最大最小值间以间隔为0.0005插入点
    func = interpolate.interp1d(x_arr, y_arr)
    ynew = func(xnew)  # 得到插入x对应的y值
    plt.plot(xnew, ynew, '-')  # 绘制图像
    plt.show()  # show图像



filepath = '17.jpg'
img1=cv2.imread(filepath)
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


import cv2
import numpy as np
import random
import os
filepath1 = '17_new.jpg'
img3=cv2.imread(filepath1)
dir_name1, full_file_name1 = os.path.split(filepath1)
file_name1, file_ext1 = os.path.splitext(full_file_name1)
img4 = cv2.imread(filepath1,cv2.IMREAD_GRAYSCALE)
edge_img1=cv2.Canny(img4,300,400)
cv2.imshow('edges',edge_img1)
mask1=np.zeros_like(edge_img1)
mask1=cv2.fillPoly(mask1,np.array([[[100,720],[520,260],[1000,260],[1280,720]]]),color=255)
masked1=cv2.bitwise_and(edge_img1,mask1)
lines1=cv2.HoughLinesP(masked1,1,np.pi/180,15,minLineLength=40,maxLineGap=20)
def calculate_slope(line1):
    k1 = (line1[0][3] - line1[0][1]) / (line1[0][2] - line1[0][0])
    return k1
left_lines1=[line1 for line1 in lines1 if calculate_slope(line1)<0]
right_lines1=[line1 for line1 in lines1 if calculate_slope(line1)>0]
print(len(lines1))
print(len(left_lines1))
print(len(right_lines1))
def ral(lines1,thres1):
    slopes1=[calculate_slope(line1) for line1 in lines1]
    while len(lines1):
        mean1 = np.mean(slopes1)
        diff1 = [abs(s1 - mean1) for s1 in slopes1]
        idx1 = np.argmax(diff1)
        if diff1[idx1]>thres1:
            slopes1.pop(idx1)
            lines1.pop(idx1)
        else:
            break
    return lines1
ral(left_lines1,thres1=0.2)
ral(right_lines1,thres1=0.2)
print((left_lines1))
print((right_lines1))
def least_squares_fit(lines1):
    xc1=np.ravel([[line1[0][0],line1[0][2]] for line1 in lines1])
    yc1=np.ravel([[line1[0][1],line1[0][3]] for line1 in lines1])
    poly1=np.polyfit(xc1,yc1,deg=1)
    point_min1=(np.min(xc1),np.polyval(poly1,np.min(xc1)))
    point_max1=(np.max(xc1),np.polyval(poly1,np.max(xc1)))
    return np.array([point_min1,point_max1], dtype=np.int32)
left_line1=(least_squares_fit(left_lines1))
right_line1=(least_squares_fit(right_lines1))
print((left_line1))
print((right_line1))


#Calculating length
lenl0=((left_line[0][1]-left_line[1][1])**2+(left_line[0][0]-left_line[1][0])**2)**0.5
lenl1=((left_line1[0][1]-left_line1[1][1])**2+(left_line1[0][0]-left_line1[1][0])**2)**0.5
lenr0=((right_line[0][1] - right_line[1][1])**2+(right_line[0][0] - right_line[1][0])**2)**0.5
lenr1=((right_line1[0][1] - right_line1[1][1])**2+(right_line1[0][0] - right_line1[1][0])**2)**0.5
diff_hrz_l=lenl1/lenl0
diff_hrz_r=lenr1/lenr0
print(diff_hrz_l,diff_hrz_r)

# Plotting X-Y PLOT
l0=[]
r0=[]
l1=[]
r1=[]
diff_l=[]
diff_r=[]
k0l=(left_line[0][1]-left_line[1][1])/(left_line[0][0]-left_line[1][0])
bl=left_line[0][1]-k0l*left_line[0][0]
for i0l in range(left_line[1][1],left_line[0][1]):
    j0l=((i0l - bl) / k0l)
    l0.append([i0l,j0l])

k1l=(left_line1[0][1]-left_line1[1][1])/(left_line1[0][0]-left_line1[1][0])
bl1=left_line1[0][1]-k1l*left_line1[0][0]
for i1l in range(left_line1[1][1],left_line1[0][1]):
    j1l=((i1l - bl1) / k1l)
    l1.append([i1l,j1l])

k0r = (right_line[0][1] - right_line[1][1]) / (right_line[0][0] - right_line[1][0])
br = right_line[0][1] - k0r * right_line[0][0]
for i0r in range(right_line[0][1], right_line[1][1]):
    j0r = ((i0r - br) / k0r)
    r0.append([i0r, j0r])

k1r = (right_line1[0][1] - right_line1[1][1]) / (right_line1[0][0] - right_line1[1][0])
b1r = right_line1[0][1] - k1r * right_line1[0][0]
for i1r in range(right_line1[0][1], right_line1[1][1]):
    j1r = ((i1r - b1r) / k1r)
    r1.append([i1r, j1r])

for [il,jl] in l0:
    for [il1,jl1] in l1:
        if il == il1:
            diff_l.append([il,jl1-jl])
print(diff_l)

for [ir,jr] in r0:
    for [ir1,jr1] in r1:
        if ir == ir1:
            diff_r.append([ir,jr1-jr])
print(diff_r)

if __name__ == '__main__':
    xl=[]
    yl=[]
    for [il, jl] in diff_l:
        xl.append(il)
        yl.append(jl)
    smooth_plot(xl,yl)

if __name__ == '__main__':
    xr=[]
    yr=[]
    for [ir, jr] in diff_r:
        xr.append(ir)
        yr.append(jr)
    smooth_plot(xr,yr)



cv2.imshow('img3',img3)
imgc=cv2.imread(filepath,cv2.IMREAD_COLOR)
imgc1=cv2.imread(filepath1,cv2.IMREAD_COLOR)
cv2.line(imgc,tuple(left_line[0]),tuple(left_line[1]),color=(0,255,255),thickness=3)
cv2.line(imgc,tuple(right_line[0]),tuple(right_line[1]),color=(0,255,255),thickness=3)
cv2.line(imgc1,tuple(left_line1[0]),tuple(left_line1[1]),color=(0,255,255),thickness=3)
cv2.line(imgc1,tuple(right_line1[0]),tuple(right_line1[1]),color=(0,255,255),thickness=3)
cv2.imshow('masked',masked1)
cv2.imshow('imgc',imgc)
cv2.imshow('imgc1',imgc1)
cv2.waitKey(0)
if cv2.waitKey(0) == 115:
    cv2.imwrite(file_name1+'_new.jpg', img3)

