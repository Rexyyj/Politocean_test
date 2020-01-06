import cv2
import numpy as np  
import time

#for char display
org1 = (40, 80)
org2 = (10, 180)
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.8
fontcolor_blu = (0, 255, 0) # BGR
fontcolor_red = (0, 0, 255) # BGR
thickness = 0.5
lineType = 4
bottomLeftOrigin = 1

#for point display
point_size = 10
point_color = (0, 0, 255) # BGR
thickness = 4 # 可以为 0 、4、8

# camera structure
cap = cv2.VideoCapture(0)

while(1):
    ret, original_img = cap.read()
    #img = cv2.resize(original_img,None,fx=0.8, fy=0.8, interpolation = cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(original_img,(3,3),0)
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值
    result = img.copy()

    if lines is not None:
        #collete all the points of the found lines
        point1=[]
        point2=[]
        collection_l1=[]
        collection_l2=[]

        for line in lines:
            rho = line[0][0]  #第一个元素是距离rho
            theta= line[0][1] #第二个元素是角度theta
            
            if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
                pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
                point1.append(pt1)
                #该直线与最后一行的焦点
                pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
                point2.append(pt2)
                # cv2.line( result, pt1, pt2, (255))             # 绘制一条白线
            else:                                                  #水平直线
                pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
                #该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
                # cv2.line(result, pt1, pt2, (255), 1)           # 绘制一条直线
        
        #find the two lines belonging to the two side of the track
        if len(point1)>=2:
            for i in range(len(point1)-1):
                for j in range(i+1,len(point1),1):
                    temp=point1[i][0]-point1[j][0]
                    if abs(temp)>50 and abs(temp)<200:
                        collection_l1.append(point1[i])
                        collection_l2.append(point1[j])
                        break
                if len(collection_l1)!=0:
                    break
            if len(collection_l1)!=0:
                collection_l1.append(point2[point1.index(collection_l1[0])])
                collection_l2.append(point2[point1.index(collection_l2[0])])

        if len(collection_l1)>=2:
            cv2.line(result,collection_l1[0],collection_l1[1],(255),3)
            cv2.line(result,collection_l2[0],collection_l2[1],(255),3)

            pixel_diff=abs(collection_l1[0][0]-collection_l2[0][0])-abs(collection_l1[1][0]-collection_l2[1][0])
            cv2.putText(result, "pixel_differ="+str(pixel_diff), org1, fontFace, fontScale, fontcolor_blu, thickness, lineType)


    cv2.imshow('Canny', edges )
    cv2.imshow('Result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.8)
cap.release()
cv2.destroyAllWindows() 