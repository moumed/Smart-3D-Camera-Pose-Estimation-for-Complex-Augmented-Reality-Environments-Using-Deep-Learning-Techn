import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras
from keras import layers, models
import os
import numpy as np
from pathlib import Path
from keras.regularizers import l2
import matplotlib.pyplot as plt
import glob
import cv2
from keras.models import load_model,model_from_json
import operator
import scipy
import scipy.spatial
import time



#from sklearn.model_selection import train_test_split
#uncommenct for macosx
'''os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
'''

# Read dataset image
def read_rgb_img(img_list, img):
    n = cv2.imread(img)
    n = cv2.cvtColor(n,cv2.COLOR_BGR2RGB)
    img_list.append(n)
    #print(img)
    return img_list

# Read dataset Pose
def read_txt(txt_list, txt):
    n = np.loadtxt(txt)
    txt_list.append(n)
    #print(txt)
    return txt_list

# Get RGB Patches
def get_Patches_data(rgbImg,zDepth,rotVect,transVect):
    
    K = np.asarray([[606.209,0,320.046],
                        [0,606.719,238.926],
                        [0,0,1]],np.float32)

    #distCoeffs = np.asarray([[ 0, 0, 0, 0, 0]],np.float32)
    
    # Extract SURF KP
    surf_HS = 2000
    surf = cv2.xfeatures2d.SURF_create(surf_HS)
    kp,des = surf.detectAndCompute(rgbImg,None)
    #print('Number of Keypoints detected :',len(kp))
    nb_KP = len(kp)
    nb_good_patch = nb_KP

    RCam = np.linalg.inv(cv2.Rodrigues(rotVect)[0])
    TCam =-np.dot(RCam ,transVect.reshape(-1,1))

    #kp_nz_D = []
    Patches_RGB = []
    Patches_3D = []
    Patches_2D = []
    patch_size = 25

    for i in range(len(kp)):
        #Patch RGB
        kp_rgb = np.array(rgbImg[int(kp[i].pt[1])-patch_size:int(kp[i].pt[1])+patch_size,
                                     int(kp[i].pt[0])-patch_size:int(kp[i].pt[0])+patch_size])
        #Patch 2D homogeneaous
        kp_homg_2D = np.array(kp[i].pt).tolist()
        kp_homg_2D.append(1)
        kp_homg_2D = np.asarray(kp_homg_2D).reshape(-1,1)

        #Patch Depth
        kp_Di = zDepth[int(kp[i].pt[1]),int(kp[i].pt[0])]
        # Remove patches without good shape (50,50,3) and with Depth = 0
        if (kp_Di!=0 and kp_rgb.shape == (patch_size*2,patch_size*2,3)):
            nb_good_patch = nb_good_patch - 1

            Pcam = kp_Di * np.dot(np.linalg.inv(K),kp_homg_2D)
            Pw = np.dot(RCam,Pcam) + TCam

            Patches_3D.append(Pw)
            Patches_2D.append(kp_homg_2D)
            Patches_RGB.append(kp_rgb)
            #kp_nz_D.append(kp[i])
    nb_ptch = nb_KP-nb_good_patch
    print('Number of Good Patches : ', nb_ptch)
    
    return Patches_RGB,Patches_3D,Patches_2D,nb_ptch

def affiche_evolution_apprentissage(history):
    # résumé de l'historique pour loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Erreur du modèle')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['apprentissage', 'test'], loc='upper right')
    plt.savefig("Model_Error.png")
    plt.show()

def draw_box(frame,imgPoints,color):
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[1].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[4].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[0].tolist()),(imgPoints[3].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[4].tolist()),(imgPoints[5].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[4].tolist()),(imgPoints[7].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[5].tolist()),(imgPoints[1].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[5].tolist()),(imgPoints[6].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[2].tolist()),(imgPoints[3].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[2].tolist()),(imgPoints[1].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[6].tolist()),(imgPoints[2].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[6].tolist()),(imgPoints[7].tolist()),color,3)
    frame =  cv2.line(frame,(imgPoints[7].tolist()),(imgPoints[3].tolist()),color,3)
    return frame

#Path to test data
Name_scene = "data15"
#RGB Images
#le chargement prend un peu du temps ..
path_rgb = glob.glob(f"../pyrealsense2/{Name_scene}/*-rgb.png")
list_ = []
path_rgb.sort()
seq_rgb = np.asarray([read_rgb_img(list_, img) for img in path_rgb][0])

#Rotation Vectors
path_rot = glob.glob(f"../pyrealsense2/{Name_scene}/*-rotVec.txt")
list_ = []
path_rot.sort()
seq_rot = np.asarray([read_txt(list_, txt) for txt in path_rot][0])

#Translation Vectors
path_trans = glob.glob(f"../pyrealsense2/{Name_scene}/*-transVec.txt")
list_ = []
path_trans.sort()
seq_trans = np.asarray([read_txt(list_, txt) for txt in path_trans][0])

#Depth Values
path_depths = glob.glob(f"../pyrealsense2/{Name_scene}/*-zDepth.txt")
list_ = []
path_depths.sort()
seq_depths = np.asarray([read_txt(list_, txt) for txt in path_depths][0])

print("Sequences Shapes : ",seq_rgb.shape,seq_depths.shape,seq_trans.shape,seq_rot.shape)

#Path to save model weights
path_model = "box_train"
# Load CNN regression model
f = Path(f'{path_model}/_3dNet_Save_3/_3dNet__structure.json')
_3dNet_structure = f.read_text()
# Recréer l'objet du  model Keras à partir des données json
_3dNet_ = model_from_json(_3dNet_structure)
# Recharger les poids entraînés du model
_3dNet_.load_weights(f"{path_model}/_3dNet_Save_3/_3dNet__weights.h5")
#load history 
_3dNet_history = np.load(f'{path_model}/_3dNet_Save_3/_3DNet_History.npy',allow_pickle=True).flatten()[0]

#afficher courbes d'apprentissage
affiche_evolution_apprentissage(_3dNet_history)

# Data Variables for evaluation
#uncomment for saving data for evaluation
#original_Rots=[]
#original_Trans=[]
#original_3D_List=[]
#predicted_3D_List=[]
#estimated_LM_Rots=[]
#estimated_LM_Trans=[]
#predicted_3D_inliers_list=[]
#original_3D_inliers_list=[]
#Surf_time_list=[]
#Pose_time_list=[]
#_3DNet_time_list=[]

for i in range(len(seq_rgb)):
    ###### Patch 2D_3D Extraction ########
    #surf_time_begin = time.time()
    
    _RGB_Patches,_3D_Patches,_2D_Patches,NB_Patches =  get_Patches_data(seq_rgb[i],seq_depths[i],seq_rot[i],seq_trans[i])
    _RGB_Patches = np.asarray(_RGB_Patches)
    _3D_Patches = np.asarray(_3D_Patches).reshape(-1,3)
    _2D_Patches = np.asarray(_2D_Patches)[:,:2,:].reshape(-1,2)
    
    #surf_time_end = time.time()
    #surf_time = surf_time_end - surf_time_begin
    #Surf_time_list.append(surf_time)
    
    #original_Rots.append(seq_rot[i])
    #original_Trans.append(seq_trans[i])
    
    ###### 3DNet Prediction ##########
    #_3dNet_time_begin = time.time()
    
    frame_i_norm_Patches = preprocess_input(_RGB_Patches)
    frame_i_GT_3D = _3D_Patches
    frame_i_GT_2D = _2D_Patches
    frame_i_Pred_3D = _3dNet_.predict(frame_i_norm_Patches)
    frame_i = np.copy(seq_rgb[i])
    
    #_3dNet_time_end = time.time()
    #_3dNet_time = _3dNet_time_end - _3dNet_time_begin
    #_3DNet_time_list.append(_3dNet_time)
    
    #original_3D_List.append(frame_i_GT_3D)
    #predicted_3D_List.append(frame_i_Pred_3D)

    ####### PnPRansac Pose Estimation #########
    #pose_time_begin = time.time()
    
    K = np.asarray([[606.209,0,320.046],
                            [0,606.719,238.926],
                            [0,0,1]],np.float32)
    retval, rvec, tvec, inliers= cv2.solvePnPRansac(frame_i_Pred_3D,
                                                    frame_i_GT_2D,
                                                    K,
                                                    np.array([]),
                                                    #reprojectionError=7,
                                                    #iterationsCount= 1000
                                                )

    ######### Levenberg-Marquardt Pose Refinement #######
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_MAX_ITER,20,1)
    for l in range(500):
        rvecLM,tvecLM= cv2.solvePnPRefineLM(frame_i_Pred_3D,
                                frame_i_GT_2D,
                                K,
                                np.array([]),
                                rvec, 
                                tvec,
                                criteria=criteria
                                )
    #pose_time_end = time.time()
    #pose_time = pose_time_end - pose_time_begin
    #Pose_time_list.append(pose_time)
    
    #estimated_LM_Rots.append(rvecLM)
    #estimated_LM_Trans.append(tvecLM)
    
    #predicted_3D_inliers_list.append(frame_i_Pred_3D[inliers].reshape(-1,3))
    #original_3D_inliers_list.append(frame_i_GT_3D[inliers].reshape(-1,3))
    

    ######## Box AR Reconstruction #######
    objPoints = np.array([[0,90,0],
                        [125,90,0],
                        [125,90,70],
                        [0,90,70],
                        [0,0,0],
                        [125,0,0],
                        [125,0,70],
                        [0,0,70]],np.float64).reshape((-1,1,3)) * 1e-3
    projected_box_LMPose = cv2.projectPoints(objPoints,rvecLM,tvecLM,K,np.array([]))[0]
    projected_box_OriginPose=cv2.projectPoints(objPoints,seq_rot[i],seq_trans[i],K,np.array([]))[0]

    for _2d_LM,_2d_Ori in zip(projected_box_LMPose,projected_box_OriginPose):
        a1,b1 = _2d_LM.ravel()
        frame_i = cv2.circle(frame_i,(int(a1),int(b1)),9,[255,0,0],-1)
        a2,b2 = _2d_Ori.ravel()
        frame_i = cv2.circle(frame_i,(int(a2),int(b2)),9,[0,255,0],-1)

    frame_i = draw_box(frame_i,np.asarray(projected_box_LMPose,np.int32).reshape(-1,2),(255,0,0))
    #frame_i = draw_box(frame_i,np.asarray(projected_box_OriginPose,np.int32).reshape(-1,2),(0,255,0))

    screen = cv2.cvtColor(frame_i, cv2.COLOR_RGB2BGR)

    cv2.imshow("reprojection results",screen)

    # if pressed escape exit program
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break


#path name scene to save data variables
#uncomment for evaluation
#Name_scene = "data15_2000surf"
#np.save(f"{Name_scene}_original_Rots",np.asarray(original_Rots))
#np.save(f"{Name_scene}_original_trans",np.asarray(original_Trans))
#np.save(f"{Name_scene}_original_3D",np.asarray(original_3D_List))
#np.save(f"{Name_scene}_predicted_3D",np.asarray(predicted_3D_List))
#np.save(f"{Name_scene}_estimated_LM_Rots",np.asarray(estimated_LM_Rots).reshape(-1,3))
#np.save(f"{Name_scene}_estimated_LM_Trans",np.asarray(estimated_LM_Trans).reshape(-1,3))
#np.save(f"{Name_scene}_predicted_3D_inliers",np.asarray(predicted_3D_inliers_list))
#np.save(f"{Name_scene}_original_3D_inliers",np.asarray(original_3D_inliers_list))
#np.save(f"{Name_scene}_Surf_Patches_3D_Time",np.asarray(Surf_time_list))
#np.save(f"{Name_scene}_Poses_Time",np.asarray(Pose_time_list))
#np.save(f"{Name_scene}_3DNet_Time",np.asarray(_3DNet_time_list))