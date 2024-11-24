import numpy as np
import cv2
import VisionMarker as vm
import VisionMarkerLibrary as vml
vmd = vm.VisionMarker()

for key in vml.vm_dict:
    vmd.SaveMarker(key, (177, 103, 247), f'Markers/{key}#F767B1.jpg')