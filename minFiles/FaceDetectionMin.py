import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection) # makes bounding box and dots
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                int(bboxC.width * w), int(bboxC.height * h)
            cv.rectangle(img, bbox, (0, 255, 0), 2)
            cv.putText(img, str(int(detection.score[0]*100)), (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Image", img)
    cv.waitKey(1)