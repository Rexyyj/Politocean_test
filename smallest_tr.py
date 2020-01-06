
import cv2
import numpy as np
 
# 函数cv2.pyrDown()是降低图像分辨率，变为原来一半
# img = cv2.pyrDown(cv2.imread("G:/Python_code/OpenCVStudy/images/timg.jpg", cv2.IMREAD_UNCHANGED))
cap = cv2.VideoCapture("./VID_20200104_211218.mp4")
 

while True:
    ret,img=cap.read()
    
    # 将图片转化为灰度，再进行二值化
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # 边界框:
        # find bounding box coordinates
        # boundingRect()将轮廓转化成(x,y,w,h)的简单边框,cv2.rectangle()画出矩形[绿色(0, 255, 0)]
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # 最小矩形区域:
        # 1 计算出最小矩形区域 2 计算这个的矩形顶点 3 由于计算出来的是浮点数，而像素是整型，所以进行转化 4 绘制轮廓[红色(0, 0, 255)]
        # find minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours
        for i in range(1,4):
            length=abs(box[0][0]-box[i][0])
            if abs(abs(box[0][0]-box[i][0])-abs(box[0][1]-box[i][1]))<5 and length>200:
                ratio=33/length
                cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    
        # # 最小闭圆的轮廓:
        # # calculate center and radius of minimum enclosing circle[蓝色(255, 0, 0)]
        # (x, y), radius = cv2.minEnclosingCircle(c)
        # # cast to integers
        # center = (int(x), int(y))
        # radius = int(radius)
        # # draw the circle
        # img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    
    # 轮廓检测:绘制轮廓
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow("contours", img)
    
    cv2.waitKey(10)
cv2.destroyAllWindows()
