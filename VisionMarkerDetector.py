import cv2
import numpy as np
import VisionMarker as vm

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
vmd = vm.VisionMarker()

while True:
    ret, frame = cap.read()
    lst, mask, contour = vmd.VisionDetect(frame)
    marker_lst = vmd.CompareImg(lst)
    
    if not ret:
        print("Can't open the Camera")
        
    cv2.imshow('mask', mask)
    cv2.imshow('Contour', contour)
    cv2.imshow('cam',frame)
    print(marker_lst)

    k = cv2.waitKey(1)
    if k == 27:
       break

cap.release()
cv2.destroyAllWindows()