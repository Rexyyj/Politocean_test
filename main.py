# import sys
# sys.path.append(r'/home/rex/PoliTocean/idea_test/ratio')  
import cv2
import numpy as np 
from scipy import stats

cap = cv2.VideoCapture("./VID_20191211_153344.mp4")

ratio=0
counter=0
speedx=0
speedy=0
pic_arry=[]

#for line
point_color = (0, 0, 255) # BGR
thickness = 1 
lineType = 4



#for char display
org1 = (40, 80)
org2 = (40, 180)
org3 = (40, 280)
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
fontcolor_blu = (0, 255, 0) # BGR
fontcolor_red = (0, 0, 255) # BGR
thickness = 1
lineType = 4
bottomLeftOrigin = 1

# params for ShiTomasi corner detection 设置 ShiTomasi 角点检测的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow 设置 lucas kanade 光流场的参数
# maxLevel 为使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def rotate_bound(image,angle):
    #获取图像的尺寸
    #旋转中心
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    
    return cv2.warpAffine(image,M,(nW,nH))

def calculate_ratio(img):
      
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
            if abs(abs(box[0][0]-box[i][0])-abs(box[0][1]-box[i][1]))<5 and length>230 and length<300:
                ratio=33/length
                return ratio
    
    return 0



def renew_original(old_frame):
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes 创建一个掩膜为了后面绘制角点的光流轨迹
    # mask = np.zeros_like(old_frame)
    return [old_gray,p0]

def calculate_speed(old_gray,p0,img,ratio):
    pics=len(img)
    for i in range(pics):
        frame=img[i]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow 能够获取点的新位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points 取好的角点，并筛选出旧的角点对应的新的角点
        good_new = p1[st == 1]
        if i ==0:
            p0_ori=p0[st == 1]
        good_old = p0[st == 1]
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    speed_x=[]
    speed_y=[]
    for i, (new, old) in enumerate(zip(good_new, good_old)):#maybe have bug of missing points
        a, b = new.ravel()
        c, d = old.ravel()
        speed_x.append(a-c)
        speed_y.append(b-d)

    speed_x_pic=stats.mode(speed_x)[0][0]
    speed_x_pic=(speed_x_pic*ratio)/(pics/59)#unit:mm/s

    speed_y_pic=stats.mode(speed_y)[0][0]
    speed_y_pic=(speed_y_pic*ratio)/(pics/59)#unit:mm/s
    
    return [speed_x_pic,speed_y_pic]

def safe_zone_cal(width,speed):
    speed=abs(speed)
    max_acce=15
    t=speed/max_acce
    s_stop=speed*t+0.5*max_acce*t*t
    safe_zone_result=width/2-s_stop
    if safe_zone_result>540:
        safe_zone_result=540
    elif safe_zone_result<0:
        safe_zone_result=0
    return int(safe_zone_result)


def average_filter(window,new_val):
    temp=[]
    sum_val=0
    for i in range(len(window)-1):
        temp.append(window[i+1])
    temp.append(new_val)

    for i in range(len(temp)):
        sum_val=sum_val+temp[i]
    
    average=sum_val/len(temp)
    
    return [temp,average]




if __name__=="__main__":

    while ratio==0:
        ret,pic =cap.read()
        pic=rotate_bound(pic,90)
        ratio=calculate_ratio(pic)

    width=1080*ratio
    width_remain=(200-width)/ratio
    window=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while True:
        (ret, pic) = cap.read()
        pic=rotate_bound(pic,90)
        counter=counter+1
        if(counter%3==1):
            old_grey,p0=renew_original(pic)
        else:
            pic_arry.append(pic)

        if counter%3==0:
            speedx,speedy=calculate_speed(old_grey,p0,pic_arry,ratio)
            pic_arry=[]

        if counter == 1000:
            counter=0

        cv2.putText(pic, "speedx="+str(speedx)+"mm/s", org1, fontFace, fontScale, fontcolor_blu, thickness, lineType)
        cv2.putText(pic, "speedy="+str(speedy)+"mm/s", org2, fontFace, fontScale, fontcolor_blu, thickness, lineType)

#predit safe zone

        window,speedx_filted=average_filter(window,speedx/ratio)
        safe_zone_val=safe_zone_cal(width_remain,speedx_filted)
        
        ptStart1 = (540-safe_zone_val, 0)
        ptEnd1 = (540-safe_zone_val, 1920)
        cv2.line(pic, ptStart1, ptEnd1, point_color, thickness, lineType)

        ptStart2 = (540+safe_zone_val, 0)
        ptEnd2= (540+safe_zone_val, 1920)
        cv2.line(pic, ptStart2, ptEnd2, point_color, thickness, lineType)
        cv2.putText(pic, "safe_zone="+str(safe_zone_val),org3, fontFace, fontScale, fontcolor_blu, thickness, lineType)

        cv2.imshow('frame', pic)
        cv2.waitKey(20)

        




