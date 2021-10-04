import numpy as np
import cv2 
import argparse
import matplotlib.pyplot as plt

Calib_Mtrx = np.asarray([[606.209,0,320.046],
                        [0,606.719,238.926],
                        [0,0,1]],np.float32)

distCoeffs = np.asarray([[ 0, 0, 0, 0, 0]],np.float32)

#Specifer le chemin vers la vidéo
name_file = "data10"
cap = cv2.VideoCapture(f'../pyrealsense2/box_video_{name_file}.mp4')

color = np.array([255,255,0])
j=4
while(1):
    ret,frame = cap.read()
    #Execute SURF
    surf_KP = 500
    surf = cv2.xfeatures2d.SURF_create(surf_KP)
    kp,des = surf.detectAndCompute(frame,None)
    print('Number of Keypoints detected :',len(kp))
    nb_good_depth = len(kp)
    
    #Charger les paramètres de pose
    rotVect= np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-rotVec.txt')
    transVect= np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-transVec.txt')
    img_depths = np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-zDepth.txt')
    
    
    #calul 3D coords
    RCam = np.linalg.inv(cv2.Rodrigues(rotVect)[0])
    TCam =-np.dot(RCam ,transVect.reshape(-1,1))
    Pw_new = []
    kp_nz_D = []
    for i in range(len(kp)):
        Di = img_depths[int(kp[i].pt[1]),int(kp[i].pt[0])]
        if (Di!=0): # Ingorer les points avec la profondeur 0
            kp_nz_D.append(kp[i])
            nb_good_depth = nb_good_depth - 1
            kp_homg = np.array(kp[i].pt).tolist()
            kp_homg.append(1)
            kp_homg = np.asarray(kp_homg).reshape(-1,1)
            Pcam = Di * np.dot(np.linalg.inv(Calib_Mtrx),kp_homg)
            Pw = np.dot(RCam,Pcam) + TCam
            Pw_new.append(Pw)
            
    print('Number of Zero Depths : ', nb_good_depth)

    #Tracer les points SURF originaux
    frame = cv2.drawKeypoints(frame,kp_nz_D,None,(0,0,255))


    objPoints = np.array(Pw_new).reshape(-1,3)

    _2DPoints = cv2.projectPoints(objPoints,rotVect, transVect,Calib_Mtrx, distCoeffs)[0]
    #Tracer les SURF reprojetés
    for _2DP in _2DPoints:
        a,b = _2DP.ravel()
        frame = cv2.circle(frame,(int(a),int(b)),2,color.tolist(),-1)
    
    cv2.imshow('Reprojecting Surf',frame)
    k = cv2.waitKey(20) & 0xff # waitkey 30 --> a chaque fois petit on acceler la lecture de vidéo
    if k == 27:
        break
    j=j+1
    