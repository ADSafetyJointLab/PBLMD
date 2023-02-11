import cv2
import imutils
import numpy as np
import random
def saturation(path):
    def l_s_b(arg):
        # 图像归一化，且转换为浮点型, 颜色空间转换 BGR转为HLS
        # astype用于转换数组的数据类型
        fImg = img.astype(np.float32)
        fImg = fImg / 255.0
        # HLS空间，三个通道分别是: Hue色相、lightness明度、saturation饱和度
        # 通道0是色相、通道1是明度、通道2是饱和度
        hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)

        """HSL--明亮度和饱和度"""
        """
            函数 cv2.getTrackbarPos():
            功能：得到滑动条的数值
            参数：1.滑动条名字 2.窗口名
            """
        lnum = cv2.getTrackbarPos('lightness', wname)
        snum = cv2.getTrackbarPos('saturation', wname)
        cnum = cv2.getTrackbarPos('contrast', wname)
        # 1.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
        hlsImg[:, :, 1] = (0.5 + lnum / float(MAX_VALUE)) * hlsImg[:, :, 1]
        # hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
        # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
        hlsImg[:, :, 2] = (0.5 + snum / float(MAX_VALUE)) * hlsImg[:, :, 2]
        # hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)

        """BGR--对比度"""
        h, w, ch = lsImg.shape  # 获取shape的数值，height和width、通道
        # 新建全零图片数组img2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        img2 = np.zeros([h, w, ch], lsImg.dtype)
        dst = cv2.addWeighted(lsImg, cnum, img2, 1 - cnum, 0)  # addWeighted函数说明如下
        result=imutils.resize(dst, 400)
        # 显示调整后的效果
        cv2.imshow(wname, result)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
        if cv2.waitKey(0) == 115:
            cv2.imwrite('result_lsb.jpg', result*255)

    wname = 'simple_edit'
    img = cv2.imread(path)

    if __name__ == '__main__':
        MIN_VALUE = -100
        MAX_VALUE = 100
        cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)
        # 第一个数为默认值，第二个数为最大范围
        cv2.createTrackbar("lightness", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar("saturation", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar('contrast', wname, 1, 10, l_s_b)
        l_s_b(0)

def noise(path,n,sp,noisetype,mean,var):
    def sp_noise(img):
        '''
        添加椒盐噪声
        :param img: 原始图片
        :param prob: 噪声比例
        :param sp: 0: 椒噪声, 1: 盐噪声, 2: 椒盐噪声
        :return: resultImg
        '''
        prob=n/100
        resultImg = np.zeros(img.shape, np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()  # 随机生成0-1之间的数字
                if sp == 0:
                    if rdn < prob:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                        resultImg[i][j] = 0
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
                if sp == 1:
                    if rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
                if sp == 2:
                    if rdn < prob:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                        resultImg[i][j] = 0
                    elif rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
        return resultImg

    def gauss_noise(img):
        '''
        添加高斯噪声
        :param img: 原始图像
        :param mean: 均值
        :param var: 方差,越大，噪声越大
        :return: resultImg
        '''
        image = np.array(img / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.6, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        resultImg = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        resultImg = np.uint8(resultImg * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
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







import PySimpleGUI as sg   # 导入PySimpleGUI模块，并命名为sg(sg为官网推荐命名，可用其他字符替代)
sg.theme('Dark')
layout = [  [sg.Text('导入所需要编辑的车道图片'), sg.InputText(),sg.FileBrowse()],
            [sg.Text('色彩饱和度'), sg.OK('Edit', key='OK1')], #本行1个控件，静态显示文本
            [sg.Text('噪声类型(1-saltpaper,2-gauss) 以下若空填0'), sg.InputText(),sg.Text('噪声比例'), sg.InputText(), sg.Text('%'), sg.Text('sp类型-0椒1盐2椒盐'), sg.InputText()],
            [sg.Text('gauss均值'),sg.InputText(),sg.Text('gauss方差'),sg.InputText(),sg.OK('Edit', key='OK2')], # 本行2个控件，分别是静态显示文本和文本输入框
            [sg.Text('车道线磨损程度'), sg.InputText()],
            [sg.Cancel('Cancel', key='Cancel')]] # 本行2个控件，分别是"确定"和"取消"按键,参数1为按键的显示文字，参数key为按键的键值
window = sg.Window('Window Title', layout)
while True:
    event, values = window.read() # 用于读取页面上的事件和输入的数据。
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
    # 其返回值为('事件', {0: '输入控件1接收的值', 1: '输入控件2接受的值'})
    if event == 'Cancel':
        break  # 退出循环
window.close()


'''def saturation(path):
    def l_s_b(arg):
        # 图像归一化，且转换为浮点型, 颜色空间转换 BGR转为HLS
        # astype用于转换数组的数据类型
        fImg = img.astype(np.float32)
        fImg = fImg / 255.0
        # HLS空间，三个通道分别是: Hue色相、lightness明度、saturation饱和度
        # 通道0是色相、通道1是明度、通道2是饱和度
        hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)

        """HSL--明亮度和饱和度"""
        """
            函数 cv2.getTrackbarPos():
            功能：得到滑动条的数值
            参数：1.滑动条名字 2.窗口名
            """
        lnum = cv2.getTrackbarPos('lightness', wname)
        snum = cv2.getTrackbarPos('saturation', wname)
        cnum = cv2.getTrackbarPos('contrast', wname)
        # 1.调整亮度饱和度(线性变换)、 2.将hlsCopy[:,:,1]和hlsCopy[:,:,2]中大于1的全部截取
        hlsImg[:, :, 1] = (0.5 + lnum / float(MAX_VALUE)) * hlsImg[:, :, 1]
        # hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
        # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
        hlsImg[:, :, 2] = (0.5 + snum / float(MAX_VALUE)) * hlsImg[:, :, 2]
        # hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)

        """BGR--对比度"""
        h, w, ch = lsImg.shape  # 获取shape的数值，height和width、通道
        # 新建全零图片数组img2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        img2 = np.zeros([h, w, ch], lsImg.dtype)
        dst = cv2.addWeighted(lsImg, cnum, img2, 1 - cnum, 0)  # addWeighted函数说明如下

        # 显示调整后的效果
        cv2.imshow(wname, imutils.resize(dst, 400))

    def noise(nnum,sp):
        nnum = cv2.getTrackbarPos('noise', wname)
        sp = cv2.getTrackbarPos('sp', wname)
        '''
'''添加椒盐噪声
            :param img: 原始图片
            :param prob: 噪声比例
            :param sp: 0: 椒噪声, 1: 盐噪声, 2: 椒盐噪声
            :return: resultImg
            '''
'''resultImg = np.zeros(img.shape, np.uint8)
        thres = 1 - nnum
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()  # 随机生成0-1之间的数字
                if sp == 0:
                    if rdn < nnum:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                        resultImg[i][j] = 0
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
                if sp == 1:
                    if rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
                if sp == 2:
                    if rdn < nnum:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                        resultImg[i][j] = 0
                    elif rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                        resultImg[i][j] = 255
                    else:
                        resultImg[i][j] = img[i][j]  # 其他情况像素点不变
        return resultImg

    wname = 'simple_edit'
    img = cv2.imread(path)

    if __name__ == '__main__':
        MAX_VALUE = 100
        cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)
        # 第一个数为默认值，第二个数为最大范围
        cv2.createTrackbar("lightness", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar("saturation", wname, 50, MAX_VALUE, l_s_b)
        cv2.createTrackbar('contrast', wname, 1, 10, l_s_b)
        cv2.createTrackbar('noise', wname, 0, 1, noise)
        cv2.createTrackbar('sp', wname, 0, 2, noise)
        l_s_b(0)
        if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()'''