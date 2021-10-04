import numpy as np
import cv2 
import matplotlib.pyplot as plt
import glob

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

def get_Patches_data(rgbImg,zDepth,rotVect,transVect):
    
    K = np.asarray([[606.209,0,320.046],
                        [0,606.719,238.926],
                        [0,0,1]],np.float32)

    #distCoeffs = np.asarray([[ 0, 0, 0, 0, 0]],np.float32)
    
    # Extract SURF KP
    surf_HS = 500
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

    print('Number of Good Patches : ', nb_KP-nb_good_patch)
    
    return Patches_RGB,Patches_3D,Patches_2D


# Load Data : data folders name
Group_scenes = ["data10","data11","data12","data13","data14","data15"]

for Name_scene in Group_scenes:
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
    
    Patches_RGB_list= []
    Patches_3D_list=[]
    Patches_2D_list=[]
    for rgbImg , zDepth ,rotVect ,transVect in zip(seq_rgb,seq_depths,seq_rot,seq_trans):
        _RGB_Patches,_3D_Patches,_2D_Patches =  get_Patches_data(rgbImg,zDepth,rotVect,transVect)
        _RGB_Patches = np.asarray(_RGB_Patches)
        _3D_Patches = np.asarray(_3D_Patches)
        _2D_Patches = np.asarray(_2D_Patches)[:,:2,:]

        for rgb_ptch,_3d_ptch,_2d_ptch in zip(_RGB_Patches,_3D_Patches,_2D_Patches):
            Patches_RGB_list.append(rgb_ptch)
            Patches_3D_list.append(_3d_ptch)
            Patches_2D_list.append(_2d_ptch)

    Patches_RGB_list=np.asarray(Patches_RGB_list)
    Patches_3D_list =np.asarray(Patches_3D_list)
    Patches_2D_list = np.asarray(Patches_2D_list)
    
    #Path to save data
    np.save(f"{Name_scene}_Patches_RGB_list",Patches_RGB_list)
    np.save(f"{Name_scene}_Patches_3D_list",Patches_3D_list)
    np.save(f"{Name_scene}_Patches_2D_list",Patches_2D_list)
    
    print("Lists Shapes :", Patches_RGB_list.shape,Patches_3D_list.shape,Patches_2D_list.shape)

