import cv2
import imutils
import numpy as np
import random
def saturation(path):
    def l_s_b(arg):
        fImg = img.astype(np.float32)
        fImg = fImg / 255.0
        hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
        lnum = cv2.getTrackbarPos('lightness', wname)
        snum = cv2.getTrackbarPos('saturation', wname)
        cnum = cv2.getTrackbarPos('contrast', wname)
        #Adjust brightness, saturation and contrast linearly
        hlsImg[:, :, 1] = (0.5 + lnum / float(MAX_VALUE)) * hlsImg[:, :, 1]
        hlsImg[:, :, 2] = (0.5 + snum / float(MAX_VALUE)) * hlsImg[:, :, 2]
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
        """BGR--对比度"""
        h, w, ch = lsImg.shape  # Obtain the values of the shape, height, width, and channel
        # Create a new all-zero array img2
        img2 = np.zeros([h, w, ch], lsImg.dtype)
        dst = cv2.addWeighted(lsImg, cnum, img2, 1 - cnum, 0)
        result=imutils.resize(dst, 400)
        # Show adjusted image
        cv2.imshow(wname, result)
        if cv2.waitKey(0) == 115: #Press 's' to save the image
            cv2.imwrite('result_lsb.jpg', result*255)
    wname = 'simple_edit'
    img = cv2.imread(path)
    if __name__ == '__main__':
        MIN_VALUE = -100
        MAX_VALUE = 100
        cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)  #Setting range of values
        cv2.createTrackbar("lightness", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar("saturation", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar('contrast', wname, 1, 10, l_s_b)
        l_s_b(0)
def noise(path,n,sp,noisetype,mean,var):
    def sp_noise(img):
        prob=n/100
        resultImg = np.zeros(img.shape, np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if sp == 0:
                    if rdn < prob:  # Once the random number is less than the noise ratio, turn the pixel to black (pepper noise)
                        resultImg[i][j] = 0
                    else:
                        resultImg[i][j] = img[i][j]  #Left other pixel unchanged
                if sp == 1:
                    if rdn > thres:  # Once the random number is bigger than the noise ratio, turn the pixel to white (salt noise)
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]   #Left other pixel unchanged
                if sp == 2:
                    if rdn < prob:  # Once the random number is less than the noise ratio, turn the pixel to black (pepper noise)
                        resultImg[i][j] = 0
                    elif rdn > thres:  # Once the random number is bigger than the noise ratio, turn the pixel to white (salt noise)
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]   #Left other pixel unchanged
        return resultImg

    def gauss_noise(img):
        image = np.array(img / 255, dtype=float)  # Normalize the pixel value of the original image and divide it by 255
        noise = np.random.normal(mean, var ** 0.6, image.shape)  # Create an image matrix with a mean and a Gaussian distribution with a variance (var)
        out = image + noise  # Add the noise to the original image to get the noised image
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        resultImg = np.clip(out, low_clip, 1.0)
        resultImg = np.uint8(resultImg * 255)  #Restore the image
        return resultImg
    if __name__ == '__main__':
        img = cv2.imread(path)
        if noisetype == 1:
            resultImg = sp_noise(img)
        elif noisetype == 2:
            resultImg = gauss_noise(img)
        cv2.imshow('origin', img)
        cv2.imshow('result', resultImg)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
        if cv2.waitKey(0) == 115:
            cv2.imwrite('result_noise.jpg', resultImg)

import PySimpleGUI as sg
sg.theme('Dark')
#Create GUI interface
layout = [  [sg.Text('Import the image'), sg.InputText(),sg.FileBrowse()],
            [sg.Text('Saturation'), sg.OK('Edit', key='OK1')],
            [sg.Text('Type of noise (1-saltpaper,2-gauss) (Fill 0 in the following blanks once it should be placed vacant)')],
            [sg.InputText(),sg.Text('Ratio of noise'), sg.InputText(), sg.Text('%'), sg.Text('noise type:0-pepper, 1-salt, 2-pepper+salt'), sg.InputText()],
            [sg.Text('Gaussian mean'),sg.InputText(),sg.Text('Gaussian variance'),sg.InputText(),sg.OK('Edit', key='OK2')],
            [sg.Cancel('Cancel', key='Cancel')]]
window = sg.Window('Window Title', layout)
while True:
    event, values = window.read()
    value=values.values()
    path=list(value)[0]

    if event == 'OK1':
        saturation(path)
    if event == 'OK2':
        noisetype = int(list(value)[2])
        n=float(list(value)[3])
        sp=int(list(value)[4])
        mean=float(list(value)[5])
        var=float(list(value)[6])
        noise(path, n, sp, noisetype, mean, var)
    if event == 'Cancel':
        break
window.close()

