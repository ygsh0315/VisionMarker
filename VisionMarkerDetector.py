import cv2
import numpy as np
import VisionMarker as vm
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vmd = vm.VisionMarker()
vmd.setTrackbar()

prev_time = time.time()
fps_display_interval = 0.5
fps_counter = 0
fps = 0

while True:
    ret, frame = cap.read()
    lst, mask, contour, valid_contours = vmd.VisionDetect(frame)
    marker_lst = vmd.CompareImg(lst, valid_contours, frame)

    current_time = time.time()
    fps_counter += 1
    if current_time - prev_time >= fps_display_interval:
        fps = fps_counter / (current_time - prev_time)
        prev_time = current_time
        fps_counter = 0

    fps_text = f'FPS : {int(fps)}'

    w, h, ch = frame.shape

    cv2.putText(frame, fps_text, (w, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if not ret:
        print("Can't open the Camera")
    
    cv2.imshow('Detected Markers', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('Contour', contour)

    k = cv2.waitKey(1)
    if k == 27:
       break

cap.release()
cv2.destroyAllWindows()