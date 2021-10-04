import numpy as np
import cv2 
import argparse
import matplotlib.pyplot as plt

cameraMatrix = np.asarray([[606.209,0,320.046],
                        [0,606.719,238.926],
                        [0,0,1]],np.float32)

distCoeffs = np.asarray([[ 0, 0, 0, 0, 0]],np.float32)

#Coordonnées 3D réels de coins de la boite en mètres (m)
objPoints = np.array([[0,90,70],
                    [125,90,70],
                    [125,0,70],
                    [0,0,70]],np.float32).reshape((-1,1,3)) * 1e-3
#print('objPoints = ', objPoints)
name_file = "data14" 
cap = cv2.VideoCapture(f'../pyrealsense2/box_video_{name_file}.mp4') #path vers la vidéo sauvgardée précédements

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


j=4

mask = np.zeros_like(old_gray)
mask[:,:] = 255 

# Specifier les pixels des coins à suivre avec le KLT dans le PREMIER frame
p0 = np.array([[297.4,281.3],
               [529.9,284.3],
              [505.5,380.0],
              [308.1,372.2]],np.float32).reshape(-1,1,2) 


print('P0 :',p0)

# executer PnP sur le premier frame
retval, rvecs, tvecs=cv2.solvePnP(objPoints, p0, cameraMatrix, distCoeffs,cv2.SOLVEPNP_P3P)
# Create a mask image for drawing purposes
pose_0 = np.vstack([np.concatenate([np.linalg.inv(cv2.Rodrigues(rvecs)[0]), -tvecs], axis=-1),[0,0,0,1]])
print('retval = ',retval)
print("pose_0 = \n", pose_0)

#sauvegarder la pose, vecteurs rot et trans du premier frame : Sepcifer le chemin de sauvegarde
#Uncomment this block to start saving : préciser le chemin de sauvgarde à l'argument N° 0
#debut commentaire avec : (''')
'''
np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-poseMat.txt', pose_0 ,fmt='%.9f')
np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-rotVec.txt', rvecs ,fmt='%.9f')
np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-transVec.txt', tvecs ,fmt='%.9f')
'''
#fin commentaire avec : (''')
mask = np.zeros_like(old_frame)
j=5
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #print('P1 : ',p1)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        #print('good_new : ',good_new)
        good_old = p0[st==1]
        #print('good_old : ',good_old)
    # draw the tracks
    imgPoints = []
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        imgPoints.append([a,b])
        c,d = old.ravel()
        frame = cv2.circle(frame,(int(a),int(b)),5,color.tolist(),-1)
    imgPoints = np.array(imgPoints,np.float32).reshape((-1,1,2)) 
    print('imgPoints = ', imgPoints)
    
    
    print('J= ',j)
    
    
    # Calcul de la poses dans les frames     
    retval, rvecs, tvecs=cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs,cv2.SOLVEPNP_P3P)
    pose_0 = np.vstack([np.concatenate([np.linalg.inv(cv2.Rodrigues(rvecs)[0]), -tvecs], axis=-1),[0,0,0,1]])
    print('retval = ',retval)
    print("pose_0 = \n", pose_0)
    
    
    

    #Sauvegard (Sepcifer le chemin de sauvegarde)
    #Uncomment this block to start saving : préciser le chemin de sauvgarde à l'argument N° 0 
    '''
    np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-poseMat.txt', pose_0 ,fmt='%.9f')
    np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-rotVec.txt', rvecs ,fmt='%.9f')
    np.savetxt(f'../pyrealsense2/{name_file}_poses/frame-{j:06}-transVec.txt', tvecs ,fmt='%.9f')
    '''

    img = cv2.add(frame,mask)
    cv2.imshow('Tracking Cube',img)
    k = cv2.waitKey(30) & 0xff # waitkey 30 --> a chaque fois petit à chaqeuf ios on acceler la lecture de vidéo
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    j=j+1
    