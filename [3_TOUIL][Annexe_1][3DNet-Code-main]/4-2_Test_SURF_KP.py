import numpy as np
import cv2 

#Path vers la vidéo
name_file = "data15"
cap = cv2.VideoCapture(f'../pyrealsense2/box_video_{name_file}.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 14, 
                       qualityLevel = 0.2, 
                       minDistance = 35, 
                       blockSize = 4 ) 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (9,9), 
                  maxLevel =4, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
#color = np.random.randint(0,255,(100,3))
color = np.array([255,255,255])
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
surf_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
#Declarer le SURF
Hessian_thresh_VAL = 500
surf = cv2.xfeatures2d.SURF_create(Hessian_thresh_VAL) 
kp,des = surf.detectAndCompute(surf_frame,None)
#Dessiner les points SURF dans le frame
surf_frame = cv2.drawKeypoints(surf_frame,kp,None,(255,0,0))
print('Number of Keypoints detected :',len(kp))
cv2.imshow('Tracking Cube',surf_frame)


mask = np.zeros_like(old_gray)
mask[:,:] = 255 

p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

print('P0 :',p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
j=1
while(1):
    ret,frame = cap.read()
    surf = cv2.xfeatures2d.SURF_create(Hessian_thresh_VAL)
    kp,des = surf.detectAndCompute(frame,None)
    #Dessiner les points SURF dans le frame
    surf_frame = cv2.drawKeypoints(frame,kp,None,(255,0,0))   
    print('Number of Keypoints detected :',len(kp))
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print('P1 : ',p1)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        print('good_new : ',good_new)
        good_old = p0[st==1]
        print('good_old : ',good_old)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        surf_frame = cv2.circle(surf_frame,(int(a),int(b)),7,color.tolist(),-1)
        
    
        
    img = cv2.add(surf_frame,mask)
    cv2.imshow('Tracking Cube',img)
    k = cv2.waitKey(30) & 0xff # waitkey 30 --> a chaque fois petit à chaqeuf ios on acceler la lecture de vidéo
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    j=j+1
    