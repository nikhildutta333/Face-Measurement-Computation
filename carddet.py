import os
import sys
import glob
import cv2
import dlib
import numpy as np
import imutils
import operator
import math
from skimage import io
from imutils import face_utils
import imutils
import struct
import pandas
import seaborn

faces_folder = "/home/nikhil/1000lookz/dlib-19.6/examples/face"
f2="/home/nikhil/1000lookz/dlib-19.6/tools/imglab/build"
predictor_path = "/home/nikhil/1000lookz/dlib-19.6/python_examples/shape_predictor_68_face_landmarks.dat"




#card width in inches
KNOWN_WIDTH = 3.370
WIDTH_CM = 8.5598

scale = 0.6
delta = 0
ddepth = cv2.CV_16S

#svm training parameters
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 500
options.num_threads = 8
options.be_verbose = True

#function used to calculate measurements in pixels.
def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 35, 125)    
    (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def sort_contours(cnts, method="top-to-bottom"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)        

training_xml_path = os.path.join(faces_folder, "training.xml")
training_xml_path1 = os.path.join(f2, "card.xml")

#Responsible for training detectors with svm
#dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
#dlib.train_simple_object_detector(training_xml_path1, "detector1.svm", options)


# Now let's use the detector we trained for face and card
detector = dlib.simple_object_detector("detector.svm")
detector1 = dlib.simple_object_detector("detector1.svm")
predictor = dlib.shape_predictor(predictor_path)
detector2 = dlib.get_frontal_face_detector()

# Now let's run the detector over the images in the faces folder and display the
# results.



     
def shape_to_np(shape, dtype="double"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    return coords


for f in glob.glob(os.path.join(faces_folder, "f9.jpg")):
    print("Processing file: {}".format(f))
    
    img2 = cv2.imread(f)
    size1=img2.shape
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector2(img2, 1)
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.


        shape = predictor(img2, d)
        shape = face_utils.shape_to_np(shape)
        image_points =np.array([(shape[0]),(shape[16]),(shape[8])],dtype="double")
        print(image_points)
        (x1,y1)=image_points[0]
        (x2,y2)=image_points[1]
        (x4,y4)=image_points[2]
        dist=math.sqrt((x2-x1)**2+(y2-y1)**2)
        x3=(x1+x2)/2
        y3=y1-(size1[0]*.146341)
        height=math.sqrt((x4-x3)**2+(y4-y3)**2)
        print("The face width in pixels : {}".format(dist))
        # Draw the face landmarks on the screen.
        for (x,y) in image_points:
            cv2.circle(img2,(int(x),int(y)), 5, (255,255,255), -1)
            cv2.circle(img2,(int(x3),int(y3)), 5, (255,255,255), -1)
            
        cv2.imshow("Output for landmark", img2)
        cv2.waitKey(0)



        

print("Showing detections on the images in the faces folder...")
for f in glob.glob(os.path.join(faces_folder, "f9.jpg")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    accumEdged = np.zeros(img.shape[:2], dtype="uint8")

    img1 = img.copy()
    size=img.shape
    print(size)
    cv2.rectangle(img1,(0,0),(size[1],size[0]),(0,0,0),-1)

    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        x=d.left()
        y=d.top()
        w=d.right()
        h=d.bottom()
        cv2.rectangle(img,(x,y),(w,h),(0,0,0),1)
        cv2.rectangle(img,(0,0),(x,size[1]),(0,0,0),-1)
        cv2.rectangle(img,(0,0),(size[1],y),(0,0,0),-1)
        cv2.rectangle(img,(size[1],0),(w,size[0]),(0,0,0),-1)
        img=cv2.rectangle(img,(size[1],size[0]),(0,h),(0,0,0),-1)
    
    dets1 = detector1(img)    
    for k1, d1 in enumerate(dets1):
        x1=d1.left()-15
        y1=d1.top()-10
        w1=d1.right()+10
        h1=d1.bottom()-50
        cv2.rectangle(img,(x1,y1),(w1,h1),(255,255,255),3)
        #cv2.rectangle(img1,(x1,y1),(w1,h1),(255,255,255),1)
        cv2.rectangle(img,(0,0),(x1,size[1]),(0,0,0),-1)
        cv2.rectangle(img,(0,0),(size[1],y1),(0,0,0),-1)
        cv2.rectangle(img,(size[1],0),(w1,size[0]),(0,0,0),-1)
        img=cv2.rectangle(img,(size[1],size[0]),(0,h1),(0,0,0),-1)
        pts_dst=np.array([[x1,y1],[w1,y1],[w1,h1],[x1,h1]])
    cv2.imwrite(os.path.join(faces_folder , 'waka.jpg'), img)    
        
    """ Important region"""

    im_bw=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    im_bw = cv2.GaussianBlur(im_bw,(1,1),1000)
    cv2.imshow("imgb4contour",im_bw)
    cv2.waitKey(0)
    # Gradient-X
    grad_x = cv2.Sobel(im_bw,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(im_bw,ddepth,0,1,ksize = 3, scale = scale,delta = delta, borderType = cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    im_bw = cv2.addWeighted(abs_grad_y,.4,abs_grad_y,.4,0)
    cv2.imshow("sobel",im_bw)
    cv2.waitKey(0)
    
    im_bw = cv2.Canny(im_bw,160,540)
    cv2.imshow("canny",im_bw)
    cv2.waitKey(0)

    #kernel = np.ones((3,3), np.uint8)

    #img_dilation = cv2.dilate(im_bw, kernel, iterations=1)
    #im_bw = cv2.erode(img_dilation, kernel, iterations=1)

    #cv2.imshow('Erosion', im_bw)
    #cv2.waitKey(0)
   
    (_,contours,_) = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (cnts, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    b=[[],[],[]] 
    for j in range(0,len(cnts)):
        
        (x,y),radius = cv2.minEnclosingCircle(cnts[j])
        center = (int(x),int(y))
        radius = int(radius)
        length = (radius+radius)
        b[0].append(length)
        cv2.circle(img,center, radius, (0,0,255), 1)
        if length <(.0555559*size[1]):
            b[1].append(j)
        else:
            b[2].append(j)    
    print(b[0])        
    print(b[1])
    print(b[2])
    data=b[2]

    cnt=cnts[data[2]]    
    
    cv2.drawContours(img, cnts, -1, (0,255,0), 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,0,255),2)
    img1 = cv2.drawContours(img1,[box],0,(0,0,255),2)
    pts_src=np.array([box[2],box[3],box[0],box[1]])

    
    
    #cv2.imwrite(os.path.join(faces_folder , 'waka.jpg'), img)
    #cv2.imshow("img", im_bw)
    angle = cv2.minAreaRect(pts_src)[-1]
    
    if angle < -45:
	    angle = (90 + angle)
    else:
	    angle = angle 

    M = cv2.getRotationMatrix2D((size[1],size[0]), angle, 1.0)
    img1 = cv2.warpAffine(img1, M, (size[1], size[0]))
    
    #Distance calculation part:
    marker = find_marker(img1)

    #approximate focal length which works for any image dimentions
    focalLength = size[1]-(size[1]*0.303)

    inches = distance_to_camera(KNOWN_WIDTH, focalLength,marker[1][0])

    #draw a bounding box boxround the image and display it
    box1 = np.int0(cv2.boxPoints(marker))
    cv2.drawContours(img1, [box1], 0, (255, 0,0), 2)
    print("Distance in Metres {}:".format(inches*0.0254))
    pixcard=marker[1][0]
    print("The distance in pixels :{}".format(pixcard))
    #print("knwon value")
    #print(marker[1][0])
    


    cv2.imshow("before rotation",img)
    cv2.imshow("img1",img1)
    cv2.waitKey(0)
     
face_width =(dist*WIDTH_CM)/pixcard
print("The Face Width in centimeter :{}".format(face_width))
face_height=(height*WIDTH_CM)/pixcard
print("The Face Height in centimeter :{}".format(face_height))
