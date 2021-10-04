from keras.models import load_model,model_from_json
from keras import layers, models
import keras
from pathlib import Path
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error
import operator
import scipy
import os

from prettytable import PrettyTable


os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

Name_scene = "data14"
#RGB Images
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

#path to save evaluation data
Name_scene ='data15_2000surf'

original_Rots = np.load(f'{Name_scene}_original_Rots.npy')
original_Trans = np.load(f'{Name_scene}_original_trans.npy')

#estimated_Poses_ransac_list = np.load('Fire_Ransac_Poses_List.npy')


estimated_LM_Rots = np.load(f'{Name_scene}_estimated_LM_Rots.npy')
estimated_LM_Trans = np.load(f'{Name_scene}_estimated_LM_Trans.npy')


predicted_3D_List = np.load(f'{Name_scene}_predicted_3D.npy',allow_pickle=True)


original_3D_List = np.load(f'{Name_scene}_original_3D.npy',allow_pickle=True)


predicted_3D_inliers_list= np.load(f'{Name_scene}_predicted_3D_inliers.npy',allow_pickle=True)


original_3D_inliers_list= np.load(f'{Name_scene}_original_3D_inliers.npy',allow_pickle=True)

Surf_time = np.load(f'{Name_scene}_Surf_Patches_3D_Time.npy')
Pose_time = np.load(f'{Name_scene}_Poses_Time.npy')
_3DNet_time =np.load(f'{Name_scene}_3DNet_Time.npy')

print('The mean of Surf Patch Extraction time execution per frame (Surf_t) : ',np.mean(Surf_time))
print('The mean of 3DNet Prediction time execution per frame (Pred_t) : ',np.mean(_3DNet_time))
print('The mean of Pose Estimation time execution per fram (Pose_t) : ',np.mean(Pose_time))


N_i = []
for i in range(len(predicted_3D_inliers_list)):
    N_i.append(len(predicted_3D_inliers_list[i]))
print('The Mean of inliers per frame (N_i) : ', np.mean(N_i))


N_kp = []
for i in range(len(predicted_3D_List)):
    N_kp.append(len(predicted_3D_List[i]))
print('The Mean of Key points per frame (N_kp) : ', np.mean(N_kp))

frames_mean = []
for i in range(len(predicted_3D_List)):
    frame_i_distances = []
    for j in range(len(predicted_3D_List[i])):
        frame_i_distances.append(scipy.spatial.distance.cdist(
            original_3D_List[i][j].reshape(-1,3),
            predicted_3D_List[i][j].reshape(-1,3),
        'euclidean'))
    frames_mean.append(np.mean(frame_i_distances))
Err_p = np.mean(frames_mean)
print('The mean of distance error between predictions and ground truths on the set of all predictions (Err_p) : \n',Err_p)

    
frames_mean_i = []
for i in range(len(predicted_3D_inliers_list)):
    frame_i_distances_i = []
    for j in range(len(predicted_3D_inliers_list[i])):
        frame_i_distances_i.append(scipy.spatial.distance.cdist(
            original_3D_inliers_list[i][j].reshape(-1,3),
            predicted_3D_inliers_list[i][j].reshape(-1,3),
        'euclidean'))
    frames_mean_i.append(np.mean(frame_i_distances_i))
Err_i = np.mean(frames_mean_i)
print('The mean of distance error between predictions and ground truths on the set of inliers (Err_i) : \n',Err_i)
    
Rot_err=[]
Trans_err=[]
for i in range(len(original_Trans)):
    rot_1=original_Rots[i].reshape(-1,3)
    rot_2=estimated_LM_Rots[i].reshape(-1,3)
    r1=cv2.Rodrigues(rot_1)[0]
    r2=cv2.Rodrigues(rot_2)[0]
    r_O_LM, _ = cv2.Rodrigues(r1.dot(r2.T)) # Liens --> http://www.boris-belousov.net/2016/12/01/quat-dist/
    Rot_err.append(np.linalg.norm(r_O_LM))
    
    trans_1 = original_Trans[i]
    trans_2 = estimated_LM_Trans[i]
    t_O_LM=scipy.spatial.distance.cdist(
            trans_1.reshape(-1,3),
            trans_2.reshape(-1,3),'euclidean')
    Trans_err.append(t_O_LM)
    
print('The Median Rotation error between Original Translation and Estimated one for all frames \n (R_err) : ',np.median(Rot_err))

print('The Median Translation error between Original Rotation and Estimated one for all frames  \n (T_err) : ',np.median(Trans_err))



    

    










