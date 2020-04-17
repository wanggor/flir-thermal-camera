       
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

from pylepton import Lepton

header = cv2.imread("header.png")
header = cv2.resize(header, (1280, 150)) 
model_folder = "face_detection_model"
confidence_val = 0.2
detection = False

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([model_folder, "deploy.prototxt"])
modelPath = os.path.sep.join([model_folder,
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
last_nr = 0
device = "/dev/spidev0.0"
lepton_buf = np.zeros((60, 80, 1), dtype=np.uint16)
with Lepton(device) as l:
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        frame = cv2.flip( frame, 1)
        frame = imutils.rotate(frame, -90)

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        
        data_face = []
        if detection :
            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > confidence_val:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 5 or fH < 5:
                        continue
                    
                    data_face.append([(startX, startY), (endX, endY)])
                    # draw the bounding box of the face along with the
                    # associated probability
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
        
        _,nr = l.capture(lepton_buf)
        if nr == last_nr:
            # no need to redo this frame
            continue
        last_nr = nr
        lepton_buf = np.flip(lepton_buf, 0)
        #print(np.min(lepton_buf), np.max(lepton_buf))
        array = ((lepton_buf.copy() * 0.0439 - 321) * 12.5 - 287)
        array = array.astype(np.uint8)
        array = cv2.resize(array, (640, 480)) 
        array = cv2.applyColorMap(array, cv2.COLORMAP_JET)

        
        for i in data_face:
            x1 = i[0][0] - 10
            y1 = i[0][1] - 100
            x2 = i[1][0] - 10
            y2 = i[1][1] - 100
            #cv2.rectangle(array, (x1,y1), (x2,y2),
             #           (0, 0, 255), 2)
            
            x1_real = int(x1 * lepton_buf.shape[1] / array.shape[1])
            y1_real = int(y1 * lepton_buf.shape[0] / array.shape[0])
            x2_real = int(x2 * lepton_buf.shape[1] / array.shape[1])
            y2_real = int(y2 * lepton_buf.shape[0] / array.shape[0])
            val_array = lepton_buf[y1_real: y2_real, x1_real: x2_real, :]
            if val_array.size == 0:
                pass
            else :
                real_val = '{:.2f}'.format(np.max(val_array)* 0.0439 - 321)
                
                cv2.rectangle(frame, (i[0][0], i[0][1]), (i[0][0] + 50, i[0][1]-20),
                            (0, 0, 255), -1)
                
                cv2.putText(frame, real_val, (i[0][0] +3, i[0][1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                
        
        frame = cv2.resize(frame, (640, 480))
        
        vis = np.concatenate((array, frame), axis=1)
        vis = np.concatenate((header, vis), axis=0)
        vis = cv2.resize(vis,None,fx =0.9,fy =0.9)
        # update the FPS counter
        fps.update()

        # show the output frame
        #vis = cv2.flip( vis, 0)
        #vis = cv2.flip( vis, 1)
        cv2.imshow("Frame", vis)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

