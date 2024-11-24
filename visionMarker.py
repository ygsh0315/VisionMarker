import numpy as np
import cv2
import VisionMarkerLibrary as vml

class VisionMarker:
    def callback(x):
        minHSV[0] = cv2.getTrackbarPos('minH', 'Settings')
        maxHSV[0] = cv2.getTrackbarPos('maxH', 'Settings')
        minHSV[1] = cv2.getTrackbarPos('minS', 'Settings')
        maxHSV[1] = cv2.getTrackbarPos('maxS', 'Settings')
        minHSV[2] = cv2.getTrackbarPos('minV', 'Settings')
        maxHSV[2] = cv2.getTrackbarPos('maxV', 'Settings')

    #마커 배경 색상 지정 함수
    def SetMarker(self, key, bgr):
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
    def ShowMarker(self, key, bgr):
        marker_i = self.SetMarker(key, bgr)
        cv2.imshow("img",marker_i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def SaveMarker(self,key, bgr, path):
        marker= self.SetMarker(key, bgr)
        self.ShowMarker(key, bgr)
        cv2.imwrite(path, marker)
                
    def CompareImg(self, lst):
        marker_list = []
        for img in lst:
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

    def VisionDetect(self, img):  
        #1. 영상을 불러온다.(이미지를 불러온다.)
        received_img = img
        #1.1 이미지 크기가 크니까 사이즈를 축소한다.
        # received_img = cv2.resize(received_img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        #1.2 노이즈를 제거한다.
        received_img_blur = cv2.GaussianBlur(received_img, (5, 5), 10)
        # inRange 메서드를 사용하기 위해 hsv로 변환
        img_hsv = cv2.cvtColor(received_img_blur, cv2.COLOR_BGR2HSV)
        # marker 색상 범위에 해당하면
        # mask = cv2.inRange(img_hsv, np.array([50, 100, 100], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8))
        mask = cv2.inRange(img_hsv, np.array([70, 70, 70], dtype=np.uint8), np.array([200, 150, 255], dtype=np.uint8))
        res = cv2.bitwise_and(received_img, received_img, mask=mask)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, res_bin = cv2.threshold(res_gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(res_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(received_img, contours, -1, (0,0,255), 3)
        # cv2.imshow('test', received_img)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows
        lst = []
        for ct in contours:
            x, y, w, h = cv2.boundingRect(ct)
            cv2.drawContours(res, ct, -1, (0, 255, 0), 2)
            cv2.rectangle(res, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # 컨투어 근사화
            epsilon = 0.02 * cv2.arcLength(ct, True)
            approx = cv2.approxPolyDP(ct, epsilon, True)

            # 꼭짓점이 4개인 경우만 처리
            if len(approx) == 4:
                # 좌표를 정렬하여 정확히 매핑
                src_pts = approx.reshape(4, 2).astype(np.float32)

                # 좌표를 각각 정렬: 좌표의 위치 기준으로 구분
                sorted_pts = np.zeros((4, 2), dtype=np.float32)
                s = src_pts.sum(axis=1)  # x + y
                diff = np.diff(src_pts, axis=1)  # y - x

                sorted_pts[0] = src_pts[np.argmin(s)]  # 왼쪽 위
                sorted_pts[2] = src_pts[np.argmax(s)]  # 오른쪽 아래
                sorted_pts[1] = src_pts[np.argmin(diff)]  # 오른쪽 위
                sorted_pts[3] = src_pts[np.argmax(diff)]  # 왼쪽 아래

                # 정사각형의 목적지 좌표
                dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                # 원근 변환 매트릭스 계산
                M = cv2.getPerspectiveTransform(sorted_pts, dst_pts)

                # ROI를 원근 변환
                dst = cv2.warpPerspective(received_img, M, (w, h))
                lst.append(dst)

        # 처리된 ROI 목록과 마스크, 결과 이미지 반환
        return lst, mask, res