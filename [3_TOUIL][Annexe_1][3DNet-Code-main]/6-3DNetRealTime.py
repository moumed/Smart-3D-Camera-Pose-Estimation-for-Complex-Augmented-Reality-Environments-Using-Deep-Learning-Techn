## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras
from keras import layers, models
import os
from pathlib import Path
from keras.models import model_from_json
import time



#from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

# Get RGB Patches
def get_Patches_data(rgbImg,zDepth):
    
    
    
    # Extract SURF KP
    surf_HS = 1500
    surf = cv2.xfeatures2d.SURF_create(surf_HS)
    kp,des = surf.detectAndCompute(rgbImg,None)
    #print('Number of Keypoints detected :',len(kp))
    nb_KP = len(kp)
    nb_good_patch = nb_KP


    #kp_nz_D = []
    Patches_RGB = []
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
            Patches_2D.append(kp_homg_2D)
            Patches_RGB.append(kp_rgb)
            #kp_nz_D.append(kp[i])
    nb_ptch = nb_KP-nb_good_patch
    print('Number of Good Patches : ', nb_ptch)
    
    return Patches_RGB,Patches_2D,nb_ptch

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

path_model = "Evaluation_Data_Test/box_train"
# Load CNN regression model
f = Path(f'{path_model}/_3dNet_Save_3/_3dNet__structure.json')
_3dNet_structure = f.read_text()
# Recréer l'objet du  model Keras à partir des données json
_3dNet_ = model_from_json(_3dNet_structure)
# Recharger les poids entraînés du model
_3dNet_.load_weights(f"{path_model}/_3dNet_Save_3/_3dNet__weights.h5")
#load history 
_3dNet_history = np.load(f'{path_model}/_3dNet_Save_3/_3DNet_History.npy',allow_pickle=True).flatten()[0]


#img_indexes = np.loadtxt('./pyrealsense2/img_indices.txt')
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.pose)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)




# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
nb_frames = 1500
try:
    i=0
    while nb_frames>0:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        #color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        #print("color_intrinsics = " ,color_intrinsics)

        #depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        #print("depth_intrinsics = " ,depth_intrinsics)


        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_m = depth_image.astype(float) * depth_scale
        ###### Patch 2D_3D Extraction ########
        _RGB_Patches,_2D_Patches,NB_Patches =  get_Patches_data(cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB),depth_m)
        _RGB_Patches = np.asarray(_RGB_Patches)
        _2D_Patches = np.asarray(_2D_Patches)[:,:2,:].reshape(-1,2)

        ###### 3DNet Prediction ##########        
        frame_i_norm_Patches = preprocess_input(_RGB_Patches)
        frame_i_GT_2D = _2D_Patches
        frame_i_Pred_3D = _3dNet_.predict(frame_i_norm_Patches)
        frame_i = np.copy(color_image)
        

        ####### PnPRansac Pose Estimation #########        
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
        

        for _2d_LM in projected_box_LMPose:
            a1,b1 = _2d_LM.ravel()
            frame_i = cv2.circle(frame_i,(int(a1),int(b1)),9,[255,0,0],-1)

        frame_i = draw_box(frame_i,np.asarray(projected_box_LMPose,np.int32).reshape(-1,2),(255,0,0))
            
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((frame_i, depth_colormap))
        #cv2.imwrite(f"./pyrealsense2/data15/frame-{i:06}-depth.png", depth_colormap)
        #cv2.imwrite(f"./pyrealsense2/data15/frame-{i:06}-rgb.png", color_image)
        #np.savetxt(f'./pyrealsense2/data15/frame-{i:06}-zDepth.txt',depth_m)
        i=i+1
        nb_frames = nb_frames-1
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
