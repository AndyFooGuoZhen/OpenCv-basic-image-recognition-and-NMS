import cv2

img = cv2. imread('lena.PNG')

#you can manually write the names of the objects, but its troublesome

# classNames =['person', 'car'.......]

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


# confs : conficdence, bbox : boundary box, confThresHold : confidence threshold (50 percent then label object)
classIds, confs, bbox = net.detect(img,confThreshold = 0.5)

print(classIds,bbox) #our values used to make box

# normally to access single element in list we do
# for classId in ClassIDs:

#but since here we are accessing 3 lists, we need to use zip

for classId , confidence, box in zip(classIds.flatten(),confs.flatten(), bbox):
    cv2.rectangle(img,box,color=(0,255,255), thickness = 2) #green bounding box ( img, box, color of boundary box, thickness of border)
    cv2.putText (img, classNames [classId-1].upper(), (box[0] +10, box[1]+30),
                 cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) #argument ( img, classNames, position of box, font type, font size, color of font, thickness )

cv2.imshow("Output",img)
cv2.waitKey(0)