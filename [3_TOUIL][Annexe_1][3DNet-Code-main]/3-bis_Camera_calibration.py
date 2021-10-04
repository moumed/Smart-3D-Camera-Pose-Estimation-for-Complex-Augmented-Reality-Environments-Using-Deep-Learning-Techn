
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#uncomment on mac
'''os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['openmp'] = 'True'
'''
#Path vers le dossier des images RGB
imagesPathRight = "realsense_calibration_frames/"
#nombres de photos à utiliser pour la calibration
nbPhoto = 300
# nous définissons notre fonction
def get_cameraParams(nbPhoto,foldPhoto):
    ####Debut chargement des photos#####
    print("## Fonction réalisée par MOHAMED AMEZIANE TOUIL ...")
    print("## Chargement des photos ...")
    images = []
    for i in range(nbPhoto):
        filename = os.listdir(foldPhoto)[i]
        img = cv2.imread(os.path.join(foldPhoto,filename))
        if img is not None:
            images.append(img)
    print(len(images),'Ont été chargées')
    ###### Fin chargement des Photos #######
    print('\n')
    ###### Detection des points dans les images #####
    print("## Detection ... ")
    imageCorners2D = [] # Pour avoir les coordonnées des points trouvés dans l'image
    imageCorners3D = []
    Corners2D = []
    Corners2D_2 = []
    Corners3D = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    boardSize = (6,9) # Le shape des point d'intersections interne
    # nous allons remplir la matrice des coordonnées 3D
    for j in range(0,boardSize[1]*24,24): # 24 == taille du pixel en mm
        for i in range(0,boardSize[0]*24,24):
            Corners3D.append([i,j,0])
    Corners3D = np.asarray(Corners3D,np.float32)
    # nous allons maintenant parcourir toutes les 5 images pour avoir les coordonnées 2D et leurs associer les Coord 3D 
    imgIndex = []
    print("## Affichage des points détéctés ...")
    for i in range(len(images)):
        im = images[i]
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        trouve,Corners2D = cv2.findChessboardCorners(im,boardSize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK) 
        if trouve == True:
            # la fonction cornerSubPix
            Corners2D_2D = cv2.cornerSubPix(gray,Corners2D,(11,11),(-1,-1),criteria)
            imgIndex.append(i)
            imageCorners2D.append(Corners2D_2D)
            imageCorners3D.append(Corners3D)
            I = cv2.drawChessboardCorners(im, boardSize, Corners2D,trouve) # Pour afficher les points détectés 
            plt.figure()
            plt.imshow(I,cmap='gray')
            plt.show()
    print('Le nombres d images bien détectées est :',len(imageCorners2D),'les indices des images sont', imgIndex)
    print('\n')
    # Nous allons affichées les points détectés dans ces deux images
    print("## Affichage des points détéctés ...")
    
    # Nous récuperons alors les paramètres IO & EO de la camera
    print("## Affichage des Paramètres ...")
    ret, calibrationMatrix, distortionMatrix, rotationMatrix, translationMatrix = cv2.calibrateCamera(
                                                imageCorners3D,
                                                imageCorners2D, 
                                                (640, 480), 
                                                None, None)
    print('Calibration Matrix est \n',calibrationMatrix)
    print('Distortion Matrix est \n',distortionMatrix) # le vecteur des paramètres de distortion radial et tangential
    #print('Rotation Vector est \n',rotationMatrix)
    #print('Translation Vector est \n',translationMatrix)
    print('\n')
    #Paramètres E.O per image
    '''print("## Affichage des Paramètres EO ...")
    for i in range(len(rotationMatrix)):
        pose_Params = np.vstack([np.concatenate([np.linalg.inv(cv2.Rodrigues(rotationMatrix[i])[0]), -translationMatrix[i]], axis=-1),[0,0,0,1]])
        print('la pose de l image ', i,' est : \n',pose_Params)
    print('\n')'''
    
    
    return 

get_cameraParams(nbPhoto,imagesPathRight)