import numpy as np
import cv2
import visionMarkerLibrary as vml

def SetMarker(key, bgr):
    array = vml.vm_dict[key]

    resized_arr = cv2.resize(array, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_NEAREST)

    # RGB 배열 생성
    height, width = resized_arr.shape
    bgr_array = np.ones((height, width, 3), dtype=np.uint8)

    # 값에 따라 색상 지정
    bgr_array[resized_arr == 0] = bgr  # 색상 지정
    bgr_array[resized_arr == 1] = [255, 255, 255]

    return bgr_array

def ShowMarker(key,bgr):
    marker_i = SetMarker(key, bgr)
    cv2.imshow("img",marker_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CompareImg(ImgPath):
    arr_dst = np.zeros((7,7), dtype=np.uint8)

    gray = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    h, w = bin.shape[:2]

    block_height = h // 7
    block_width = w // 7

    for y in range(7):  # 정확히 7번 반복
        for x in range(7):  # 정확히 7번 반복
            # 7x7 블록 슬라이싱
            start_y = y * block_height
            start_x = x * block_width
            block = bin[start_y:start_y + block_height, start_x:start_x + block_width]
            black_num = np.sum(block.flatten() == 0)
            white_num = np.sum(block.flatten() == 255)

            if black_num < white_num:
                arr_dst[y, x] = 1

    for key, val in vml.vm_dict.items():
        if np.array_equal(val, arr_dst):
            return key


    

ShowMarker("K",(213,152,30))
id = CompareImg("MarkerSamples/K.png")
print(id)


# class VisionMarker:
#     a = "value"
#     def __init__(self,attribute1):
#         self.a=attribute1
