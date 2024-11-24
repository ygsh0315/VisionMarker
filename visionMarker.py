import numpy as np
import cv2
import visionMarkerLibrary as vml

#마커 배경 색상 지정 함수
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

#마커 출력 함수
def ShowMarker(key,bgr):
    marker_i = SetMarker(key, bgr)
    cv2.imshow("img",marker_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 인식한 마커와 dicionary에 저장된 마커 비교
# def CompareImg(ImgPath):
#     arr_dst = np.zeros((7,7), dtype=np.uint8)

#     gray = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
#     _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#     h, w = bin.shape[:2]

#     block_height = h // 7
#     block_width = w // 7

#     for y in range(7):  # 정확히 7번 반복
#         for x in range(7):  # 정확히 7번 반복
#             # 7x7 블록 슬라이싱
#             start_y = y * block_height
#             start_x = x * block_width
#             block = bin[start_y:start_y + block_height, start_x:start_x + block_width]
#             black_num = np.sum(block.flatten() == 0)
#             white_num = np.sum(block.flatten() == 255)

#             if black_num < white_num:
#                 arr_dst[y, x] = 1

#     for key, val in vml.vm_dict.items():
#         if np.array_equal(val, arr_dst):
#             return key
#         else:
#             print("No matching marker")
        
def CompareImg(lst):
    marker_list = []
    for img in lst:
        cv2.imshow('w',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        arr_dst = np.zeros((7,7), dtype=np.uint8)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
                marker_list.append(key)
        
    return marker_list
        # else:
        #     print("No matching marker")

def VisionDetect(img):  
    #1. 영상을 불러온다.(이미지를 불러온다.)
    received_img = img
    #1.1 이미지 크기가 크니까 사이즈를 축소한다.
    # received_img = cv2.resize(received_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    #1.2 노이즈를 제거한다.
    received_img_blur = cv2.GaussianBlur(received_img, (5, 5), 10)
    # inRange 메서드를 사용하기 위해 hsv로 변환
    img_hsv = cv2.cvtColor(received_img_blur, cv2.COLOR_BGR2HSV)
    # marker 색상 범위에 해당하면 
    mask = cv2.inRange(img_hsv, np.array([95, 215, 210], dtype=np.uint8), np.array([105, 220, 215], dtype=np.uint8))
    res = cv2.bitwise_and(received_img, received_img, mask=mask)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, res_bin = cv2.threshold(res_gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(res_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(received_img, contours, -1, (0,0,255), 3)
    cv2.imshow('test', received_img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows
    lst = []
    for ct in contours:
        x, y, w, h = cv2.boundingRect(ct)
        lst.append(received_img[y:y+h, x:x+w])


    # largest_contour = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(largest_contour)
    # marker_img = received_img[y:y+h, x:x+w]


    # cv2.imshow('Original Image', received_img)
    # cv2.imshow('Detedcted Contour', mask)
    # cv2.imshow('Largest Contour', marker_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # cv2.drawContours(received_img, contours, -1, (0, 0, 255), 3)
    # cv2.imshow('test', received_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()



    #3. ROI를 정사각형 모양으로 보정한다.
    #4. 보정한 정사각형 모양의 ROI에 대해 CompareImg를 수행한다.        

    # return marker_img
    return lst
                                                                             

# HSV : [100 219 213]
# ShowMarker("W",(213,152,30)) 
# id = CompareImg("MarkerSamples/K.png")
# print(id)
img = cv2.imread("MarkerSamples/NoiseZ.png")
marker_img = VisionDetect(img)
# cv2.imshow('detected marker', marker_img)
id = CompareImg(marker_img)
print(id)
cv2.waitKey(0)
cv2.destroyAllWindows()



class VisionMarker:
    a = "value"
    def __init__(self,attribute1):
        self.a=attribute1
        #동영상촬영, VisionDetect()

#vm = VisionMarker()
