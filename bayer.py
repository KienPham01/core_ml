# import  cv2
# import numpy as image_np
# # cap = cv2.VideoCapture(0)
#
# image = cv2.imread('bilode.jpg',0)
# cv2.imshow('image',image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2

videocap = cv2.VideoCapture('videotest.mov')
sucess,image = videocap.read()
count = 0

sucess = True

while sucess:
    cv2.imwrite('frame%d.jpg'%count,image)
    sucess,image = videocap.read()
    print('read a new frame',sucess)
    count += 1
print(count)