import numpy as np
import cv2 as cv

# params for ShiTomasi corner detection 设置 ShiTomasi 角点检测的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow 设置 lucas kanade 光流场的参数
# maxLevel 为使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


def renew_original(old_frame):
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes 创建一个掩膜为了后面绘制角点的光流轨迹
    mask = np.zeros_like(old_frame)
    return [old_gray,p0,mask]

def calculate_speed(old_gray,p0,img,ratio):
    pics=np.size(img)
    for i in range(pics):
        frame=img[i]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow 能够获取点的新位置
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
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

    counts=np.bincount(speed_x)
    speed_x_pic=np.argmax(counts)
    speed_x_pic=(speed_x_pic*ratio)/(pics/60)#unit:mm/s

    counts=np.bincount(speed_y)
    speed_y_pic=np.argmax(counts)
    speed_y_pic=(speed_y_pic*ratio)/(pics/60)#unit:mm/s
    
    return [speed_x_pic,speed_y_pic]
    
