import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from solver import solver as slvr
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Processor:
    HEIGHT, WIDTH = 512, 512
    WARPW, WARPH = 450, 450

    def largest_contour(self, contours):
        biggestCnt = None
        maxArea = 0
        four_pts = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50000:
                peri = cv2.arcLength(cnt, True)
                corners = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if area > maxArea and len(corners) == 4:
                    biggestCnt = cnt
                    four_pts = corners
                    maxArea = area
        
        return (biggestCnt, four_pts)
    
    def ordered_corners(self, pts):
        rect = np.zeros((4, 2))

        s = np.sum(pts, axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        d = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        return rect
    
    def get_transform(self, rect):
        (tl, tr, br, bl) = rect
        w1 = np.sqrt((tr[1] - tl[1])**2 + (tr[0] - tl[0])**2)
        w2 = np.sqrt((br[1] - bl[1])**2 + (br[0] - bl[0])**2)
        maxW = max(int(w1), int(w2))

        h1 = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[0])**2)
        h2 = np.sqrt((bl[1] - tl[1])**2 + (bl[0] - tl[0])**2)
        maxH = max(int(h1), int(h2))

        T = np.float32([
            [0, 0],
            [maxW-1, 0],
            [maxW-1, maxH-1],
            [0, maxH-1]
        ])
        return T, maxH, maxW
    
    def get_model(self):
        model = None
        model = keras.models.load_model("model_new.h5")
        return model
    
    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            work_img = img.copy()
            work_img = cv2.resize(work_img, (self.WIDTH, self.HEIGHT))
            gray = cv2.cvtColor(work_img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 1)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
            imgCnt = work_img.copy()
            contours, heir = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(imgCnt, contours, -1, (0, 255, 0), 2)

            bigCnt, four_pts = self.largest_contour(contours)

            cv2.drawContours(imgCnt, bigCnt, -1, (255, 0, 0), 3)
            cv2.drawContours(imgCnt, four_pts, -1, (0, 0, 255), 25)

            four_pts = four_pts.reshape(4, 2,)

            rect = self.ordered_corners(four_pts)
            rect = np.float32(rect)
            # (tl, tr, br, bl) = rect

            T, maxH, maxW = self.get_transform(rect)
            M = cv2.getPerspectiveTransform(rect, T)

            warped_img_gray = cv2.warpPerspective(gray, M, (maxW, maxH))
            sudoku = warped_img_gray.copy()

            sudoku = cv2.resize(sudoku, (self.WARPW, self.WARPH))

            rows = np.vsplit(sudoku, 9)
            grids = []
            for row in rows:
                cols = np.hsplit(row, 9)
                for box in cols:
                    grids.append(box)

            grids = np.asarray(grids)
            grids = grids.reshape(grids.shape[0], grids.shape[1], grids.shape[2], 1)

            small_img = []

            for i in range(grids.shape[0]):
                img = grids[i]
                img = cv2.resize(grids[i], (32, 32))
                img = img / 255
                small_img.append(img)

            small_img = np.asarray(small_img)
            small_img = small_img.reshape(small_img.shape[0], small_img.shape[1], small_img.shape[2], 1)

            model = self.get_model()

            s = ""
            if model is not None:
                cnt = 0

                for img in small_img:
                    cnt += 1
                    sel_img = img[4:-4, 4:-4]
                    ret, thresh = cv2.threshold(sel_img, 0.5, 255, cv2.THRESH_BINARY)
                    sel_img = cv2.resize(thresh, (32, 32)).reshape(1, 32, 32, 1)
                    probability = model.predict(sel_img.reshape(1, 32, 32, 1), verbose=0)
                    class_pred = np.argmax(probability)
                    s += str(int(class_pred))
                    if cnt % 9 == 0 and cnt != 81:
                        s += "\n"

                # matrix = slvr.make_mat(s)
                # res = slvr.solve(matrix, 0, 0)
                org = slvr.make_mat(s)
                org = np.asarray(org)
                copy = org.copy()

                res = slvr.solve(copy, 0, 0)
                mask = copy - org

                if res:
                    rec_img = np.zeros((9, 9, 50, 50, 3))

                    for i in range(grids.shape[0]):
                        row, col = i // 9, i % 9
                        if mask[row][col] > 0:
                            masked = rec_img[row][col]
                            rec_img[row][col] = cv2.putText(masked.astype(int), str(int(mask[row][col])), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
                    
                    result = cv2.hconcat(rec_img[0])

                    for i in range(1, 9):
                        temp = cv2.hconcat(rec_img[i])
                        result = np.concatenate((result, temp), axis=0)

                    rec_img2 = np.zeros((9, 9, 50, 50, 3))

                    for i in range(81):
                        row, col = i // 9, i % 9
                        if mask[row][col] == 0:
                            masked = rec_img2[row][col]
                            rec_img2[row][col] = cv2.putText(masked.astype(int), str(int(org[row][col])), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
                    
                    result2 = cv2.hconcat(rec_img2[0])

                    for i in range(1, 9):
                        temp = cv2.hconcat(rec_img2[i])
                        result2 = np.concatenate((result2, temp), axis=0)

                    net_result = result + result2

                    # cv2.imwrite("output/out_img.jpg", net_result)

                    return net_result

                return None
            return None 
