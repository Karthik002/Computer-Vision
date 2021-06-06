import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence = 0.5, ):
        self.min_detection_confidence = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection) # makes bounding box and dots
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                
                bboxes.append([id, bbox, detection.score])
                
                cv.rectangle(img, bbox, (0, 255, 0), 2)
                cv.putText(img, str(int(detection.score[0]*100)), (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        return img, bboxes

def main():
    cap = cv.VideoCapture(0)
    prevTime = 0
    currTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()