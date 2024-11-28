import cv2
import numpy as np
import VisionMarker as vm
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vmd = vm.VisionMarker()
vmd.setTrackbar()
prev_time = time.time()
fps_counter = 0
fps = 0
def checkFrame(interval):
    global fps_counter, prev_time, fps
    fps_display_interval = interval
    current_time = time.time()
    fps_counter += 1
    if current_time - prev_time >= fps_display_interval:
        fps = fps_counter / (current_time - prev_time)
        prev_time = current_time
        fps_counter = 0
    fps_text = f'FPS : {int(fps)}'
    cv2.putText(frame, fps_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

while True:
    ret, frame = cap.read()
    mask, contour, marker_lst=vmd.VisionMarkerDetect(frame)
    checkFrame(0.5)
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

