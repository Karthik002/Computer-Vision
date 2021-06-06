import cv2 as cv
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, static_image_mode = False, upper_body_only = False, smooth_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
            
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.upper_body_only, self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
           if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    self.lmList.append([id, cx, cy])
                    
                    if draw:
                        cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
        
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))
        print(angle)

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.line(img, (x2, y2), (x3, y3), (255, 0, 255), 3)
            
            cv.circle(img, (x1, y1), 10, (255, 255, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (255, 255, 255), 2)
            cv.circle(img, (x2, y2), 10, (255, 255, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 255, 255), 2)
            cv.circle(img, (x3, y3), 10, (255, 255, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (255, 255, 255), 2)

            cv.putText(img, str(int(abs(angle))), (x2-20, y2+50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return angle
        
def main():
    prevTime = 0
    currTime = 0
    cap = cv.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img) 
        lmList = detector.findPosition(img)
        
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()