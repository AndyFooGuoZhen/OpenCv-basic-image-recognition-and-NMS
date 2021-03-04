import cv2
import numpy as np

thres = 0.65 #threshold to detect object
nms_threshold = 0.5  #just added NMS threshold **************************************CHANGES

# now we use webcam instead of lena img
cap=cv2.VideoCapture(0)
cap.set(3,640) # (width, resoultion)
cap.set(4,480)

classNames=[]

#so we import from coco.names

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') #differentiates names via newline and store them into a list

#print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#creating models
#opencv only need to have weights and configPath, and will handle the processing
#you will also get the ids

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:
    success,img = cap.read()
    # confs : conficdence, bbox : boundary box, confThresHold : confidence threshold (50 percent then label object)
    classIds, confs, bbox = net.detect(img,confThreshold = thres)

    # convert bbox and confs to list type
    bbox = list(bbox) # **************************************CHANGES converted all np.array to list
    confs = list(np.array(confs).reshape(1,-1)[0]) # **************************************CHANGES converted [[][][][]] to []
    confs = list(map(float,confs)) # ************************************** CHANGES converted all numpy.float32 values in confs to normal float

    # print(type(confs))
    # print(confs)


    #but since here we are accessing 3 lists, we need to use zip

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold) # **************************************CHANGES eliminate all boox with lower confidence

    for i in indices:# **************************************CHANGES
        i = i[0] # remove double brackets convert [[0]] to [0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2],box[3]
        cv2.rectangle(img, (x,y), (x+w, h+y), color=(0, 255, 255), thickness=2)
        cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                    2)  # argument ( img, classNames, position of box, font type, font size, color of font, thickness )

    # if len(classIds) != 0:
    #
        for classId , confidence, box in zip(classIds.flatten(),confs, bbox): # **************************************CHANGES changes confs to not include .flatten as confs is list

            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 250, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        2)  # argument ( img, confidence level, position of box, font type, font size, color of font, thickness )

    cv2.imshow("Output",img)
    cv2.waitKey(1) #changed to 1