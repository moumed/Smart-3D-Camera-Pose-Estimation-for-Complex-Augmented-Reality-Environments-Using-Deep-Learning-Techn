import numpy as np
import cv2 
import argparse
import matplotlib.pyplot as plt

cameraMatrix = np.asarray([[606.209,0,320.046],
                        [0,606.719,238.926],
                        [0,0,1]],np.float32)

distCoeffs = np.asarray([[ 0, 0, 0, 0, 0]],np.float32)

# Coordonnées 3D réels des 8 coins de la boite
objPoints = np.array([[0,90,0],
                      [125,90,0],
                      [125,90,70],
                      [0,90,70],
                      [0,0,0],
                      [125,0,0],
                      [125,0,70],
                      [0,0,70]],np.float32).reshape((-1,1,3)) * 1e-3
#print('objPoints = ', objPoints)

#Chemin vers le path des données sauvegardée
name_file = "data15"
cap = cv2.VideoCapture(f'../pyrealsense2/box_video_{name_file}.mp4')
color = np.array([255,0,0])

j=4
while(1):
    ret,frame = cap.read()
    #chargement de vecteurs rot et trans
    rotVect= np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-rotVec.txt')
    transVect= np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-transVec.txt')
    poseMat= np.loadtxt(f'../pyrealsense2/{name_file}/frame-{j:06}-poseMat.txt')
    imgPoints=[]
    #reprojections des points 3D de la boite 
    for _3DPoint in objPoints:
        _2DPoint = cv2.projectPoints(_3DPoint,rotVect, transVect,cameraMatrix, distCoeffs) [0]
        imgPoints.append(_2DPoint)
        a,b = _2DPoint.ravel()
        frame = cv2.circle(frame,(int(a),int(b)),9,color.tolist(),-1)
    imgPoints = np.asarray(imgPoints,np.int32).reshape(-1,2)
    print('imgPoints = ', imgPoints)
    
    #Tracer les coins 2D sur les frames
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[1].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[4].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[3].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[4].tolist()),(imgPoints[5].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[4].tolist()),(imgPoints[7].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[5].tolist()),(imgPoints[1].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[5].tolist()),(imgPoints[6].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[2].tolist()),(imgPoints[3].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[2].tolist()),(imgPoints[1].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[6].tolist()),(imgPoints[2].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[6].tolist()),(imgPoints[7].tolist()),(0,0,255),3)
    frame =  cv2.line(frame,(imgPoints[7].tolist()),(imgPoints[3].tolist()),(0,0,255),3)
    
    cv2.imshow('Reprojecting Cube',frame)
    k = cv2.waitKey(60) & 0xff # waitkey 30 --> a chaque fois petit à chaqeuf ios on acceler la lecture de vidéo
    if k == 27:
        break
    j=j+1
    