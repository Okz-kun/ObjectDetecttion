import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 20 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
CAR_WIDTH = 350.0 #INCHES
MBIKE_WIDTH = 50.0 #INCHES
# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
RED =(0,0,255)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
 
    data_list = []
    
        
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= GREEN
        
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-16), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==2: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==0:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==3:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # return list 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_car = cv.imread('ReferenceImages/image16.png')
ref_mbike = cv.imread('ReferenceImages/image17.png')
# checking object detection on reference image 
person_data = object_detector(ref_person)
cv.imshow('person', ref_person)
print(f'person Data', person_data)

cv.waitKey(0)
# the index will remain same for each entry, cause there is only one object in the image, if theres is multiples in single then index will be adjust accordingly 
person_width_in_rf = person_data[0][1]

car_data = object_detector(ref_car)
cv.imshow('car', ref_car)
print(f'car Data', car_data)
cv.waitKey(0)


car_width_in_rf = car_data[0][1]

mbike_data = object_detector(ref_mbike)
cv.imshow('bike', ref_mbike)
print(f'bike Data', mbike_data)
cv.waitKey(0)


mbike_width_in_rf = mbike_data[0][1]



print(f"Person width in pixels : {person_width_in_rf} car width in pixels: {car_width_in_rf} motorbike width in pixels: {mbike_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mbike = focal_length_finder(KNOWN_DISTANCE, MBIKE_WIDTH, mbike_width_in_rf)

focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)


cap = cv.VideoCapture(1)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='car':
            distance = distance_finder (focal_car, CAR_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='motorbike':
            distance = distance_finder (focal_mbike, MBIKE_WIDTH, d[1])
            x, y = d[2]
            
        distance = distance * 0.0254
        
        if distance >= 3.81: 
            cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(frame, f'Dis: {round(distance,2)} meters', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        else:
            cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(frame, f'Dis: {round(distance,2)} meters', (x+5,y+13), FONTS, 0.48, RED, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

