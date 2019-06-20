import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import numpy as np
import math

def get_point(vector_pic, mag_mult=1):
    mean_angle = np.mean(vector_pic[...,0])
    mean_angle = np.mean(vector_pic[...,0])
    mean_mag = np.mean(vector_pic[...,2])*mag_mult
    ret = (int(mean_mag*math.cos(mean_angle)),int(mean_mag*math.sin(mean_angle)))

    return ret



cap = cv.VideoCapture('./20190319_10y_249400_mild tracheomalacia uneditted.mpg')
out = cv.VideoWriter('./trachea_dense_flow.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30.0, (int(cap.get(3)),int(cap.get(4))))
print (cap.isOpened())
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
print(frame1.shape)
ctr = 0
while(cap.isOpened()):
    print(ctr)
    ctr = ctr+1
    ret, frame2 = cap.read()
    if np.shape(frame2) == (): #i.e. empty frame
        break
    # cv.imshow("frame", frame2)
    # cv.waitKey(0)
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)#mag*20#

    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

    roi_hsv_l = hsv[...,0][95:145,76:176]
    roi_l = bgr[95:145,76:176]
    roi_l = cv.normalize(roi_l,None,0,255,cv.NORM_MINMAX)
    roi_l = cv.arrowedLine(roi_l, (int(roi_l.shape[1]/2),int(roi_l.shape[0]/2)), get_point(roi_hsv_l, mag_mult=0.3), (255,255,255))
    cv.imshow("L",roi_l)

    roi_hsv_r = hsv[...,0][95:145,176:276]
    roi_r = bgr[95:145,176:276]
    roi_r = cv.normalize(roi_r,None,0,255,cv.NORM_MINMAX)
    roi_r = cv.arrowedLine(roi_r, (int(roi_l.shape[1]/2),int(roi_l.shape[0]/2)), get_point(roi_hsv_r, mag_mult=0.3), (255,255,255))
    cv.imshow("R",roi_r)


    bgr = cv.arrowedLine(bgr, (126,120), get_point(roi_hsv_l, mag_mult=0.3), (255,255,255))
    bgr = cv.arrowedLine(bgr, (226,120), get_point(roi_hsv_r, mag_mult=0.3), (255,255,255))
    cv.imshow("frame",bgr)

    # arrow = cv.arrowedLine(bgr, (126,120), (100,100), (255,255,255))
    # cv.imshow("arrow",arrow)

    cv.waitKey(100)
    # plt.imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
    # plt.show()
    # out.write(bgr.astype('uint8'))

    prvs = next

out.release()

print("Done!")