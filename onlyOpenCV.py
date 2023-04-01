import cv2
import numpy as np
import random
import os
filepath = '61.jpg'
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
edge_img=cv2.Canny(img,150,200)
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
cv2.imshow('img1',img1)
imgc=cv2.imread(filepath,cv2.IMREAD_COLOR)
cv2.line(imgc,tuple(left_line[0]),tuple(left_line[1]),color=(0,255,255),thickness=3)
cv2.line(imgc,tuple(right_line[0]),tuple(right_line[1]),color=(0,255,255),thickness=3)
cv2.imshow('masked',masked)
cv2.imshow('img',imgc)
cv2.waitKey(0)
if cv2.waitKey(0) == 115:
    cv2.imwrite(file_name+'_new.jpg', img1)