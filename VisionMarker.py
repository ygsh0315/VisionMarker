import numpy as np
import cv2
import VisionMarkerLibrary as vml
import time
class VisionMarker:
    # 생성자 함수, track bar 초기값 설정
    def __init__(self):
        # 초기 HSV 범위 설정
        self.minH, self.maxH = 127, 205
        self.minS, self.maxS = 73, 252
        self.minV, self.maxV = 73, 255
        self.minHSV = np.array([self.minH, self.minS, self.minV])
        self.maxHSV = np.array([self.maxH, self.maxS, self.maxV])
    # 트랙바 생성
    def setTrackbar(self):
        cv2.namedWindow('Settings')
        cv2.createTrackbar('minH', 'Settings', self.minH, 255, self.callback)
        cv2.createTrackbar('maxH', 'Settings', self.maxH, 255, self.callback)
        cv2.createTrackbar('minS', 'Settings', self.minS, 255, self.callback)
        cv2.createTrackbar('maxS', 'Settings', self.maxS, 255, self.callback)
        cv2.createTrackbar('minV', 'Settings', self.minV, 255, self.callback)
        cv2.createTrackbar('maxV', 'Settings', self.maxV, 255, self.callback)
    # 트랙바 callback
    def callback(self, x):
        # 트랙바 값으로 HSV 범위 업데이트
        self.minHSV[0] = cv2.getTrackbarPos('minH', 'Settings')
        self.maxHSV[0] = cv2.getTrackbarPos('maxH', 'Settings')
        self.minHSV[1] = cv2.getTrackbarPos('minS', 'Settings')
        self.maxHSV[1] = cv2.getTrackbarPos('maxS', 'Settings')
        self.minHSV[2] = cv2.getTrackbarPos('minV', 'Settings')
        self.maxHSV[2] = cv2.getTrackbarPos('maxV', 'Settings')
    # 근사화 좌표 정렬
    def sortPoints(self, points):
            # 좌표 정렬: 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
            sorted_pts = np.zeros((4, 2), dtype=np.float32)
            s = points.sum(axis=1)
            diff = np.diff(points, axis=1)

            sorted_pts[0] = points[np.argmin(s)]
            sorted_pts[2] = points[np.argmax(s)]
            sorted_pts[1] = points[np.argmin(diff)]
            sorted_pts[3] = points[np.argmax(diff)]

            return sorted_pts
    # 컨투어 검출 함수
    def ContourDetect(self, img):
        # 1. 노이즈 제거 및 색상 추출
        img_blur = cv2.GaussianBlur(img, (3, 3), 10)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, self.minHSV, self.maxHSV)

        # 2. 마스크 노이즈 제거
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. 컨투어 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lst = []
        valid_contours = []

        for ct in contours:
            # 컨투어 확장
            x, y, w, h = cv2.boundingRect(ct)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)

            # 컨투어 근사화
            epsilon = 0.02 * cv2.arcLength(ct, True)
            approx = cv2.approxPolyDP(ct, epsilon, True)

            # 꼭짓점이 4개인 경우 처리
            if len(approx) == 4:
                src_pts = approx.reshape(4, 2).astype(np.float32)
                sorted_pts = self.sortPoints(src_pts)
                dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                # 원근 변환 및 ROI 추출
                M = cv2.getPerspectiveTransform(sorted_pts, dst_pts)
                dst = cv2.warpPerspective(img, M, (w, h))

                # ROI 크기 정규화
                dst = cv2.resize(dst, (100, 100), interpolation=cv2.INTER_CUBIC)
                lst.append(dst)
                valid_contours.append(ct)

        # 처리된 ROI 목록과 마스크, 결과 이미지 반환
        res = cv2.bitwise_and(img, img, mask=mask)
        return lst, mask, res, valid_contours  # 컨투어 추가 반환  
    # 이미지와 딕셔너리 비교, 마커 검출 함수
    def CompareImg(self, lst, contours, img):
        marker_list = []
        for img_roi in lst:
            arr_dst = np.zeros((7, 7), dtype=np.uint8)
            gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            h, w = bin.shape[:2]

            block_height = h // 7
            block_width = w // 7

            for y in range(7):
                for x in range(7):
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

        print(marker_list)
        cv2.putText(img, f'{marker_list}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  
        
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

        return marker_list
    #마커 검출 함수
    def VisionMarkerDetect(self,frame):
        lst, mask, contour, valid_contours = self.ContourDetect(frame)
        marker_lst = self.CompareImg(lst, valid_contours, frame)
        return mask, contour, marker_lst


    