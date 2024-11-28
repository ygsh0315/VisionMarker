import numpy as np
import cv2
import VisionMarker as vm
import VisionMarkerLibrary as vml
vmd = vm.VisionMarker()

def SaveMarker(key, bgr, path):
    marker = vml.vm_dict[key]
    bgr_marker = np.zeros((*marker.shape, 3), dtype=np.uint8)
    bgr_marker[marker == 0] = bgr
    bgr_marker[marker == 1] = 255
    bgr_marker = cv2.resize(bgr_marker, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, bgr_marker)

for key in vml.vm_dict:
    #파일명 유의
    SaveMarker(key, (80, 80, 80), f'Markers/{key}#505050.jpg')
