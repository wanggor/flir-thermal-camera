# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os

class Darknet_OpenCV_YOLO():
    def __init__(self, detector_path, config = "yolov3-tiny.cfg", weight = "yolov3-tiny.weights"):
        self.labelsPath = os.path.sep.join([ detector_path, "data/config/coco.names" ])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")

        self.weightsPath = os.path.sep.join([ detector_path, "data/weights", weight])
        self.configPath = os.path.sep.join([ detector_path, "data/config", config])

        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [ self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers() ]

    def detect(self, frame, confidence_val = 0.2, class_list = ["person", "motor"],  size_min = 0.2, size_max = 0.8):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                if confidence > confidence_val:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.5)

        output = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                if self.LABELS[classIDs[i]] in class_list :
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    cx = int(x + (w / 2))
                    cy = int(y + (h / 2))

                    if (h/H > size_min) and (h/H < size_max) and (w/W > size_min) and (w/W < size_max):
                        
                        # output.append([
                        #     cx,cy, w, h, confidences[i], self.LABELS[classIDs[i]]
                        # ])
                        output.append([
                            cx,cy, w, h, confidences[i], classIDs[i]
                        ])
        
        return frame, output