import cv2
import numpy as np
import glob

def read_rgb_img(img_list, img):
    n = cv2.imread(img)
    n = cv2.cvtColor(n,cv2.COLOR_BGR2RGB)
    img_list.append(n)
    #print(img)
    return img_list


name_file = "data15"
path_rgb = glob.glob(f"../pyrealsense2/{name_file}/*-rgb.png") #Specifier le chemin aux données acquises par le module d'acquisition


list_ = []
path_rgb.sort()
seq_images = np.asarray([read_rgb_img(list_, img) for img in path_rgb][0])
height, width, layers = seq_images[0].shape
size = (width,height)

# out = cv2.VideoWriter('Fire/test_video_1.mp4',cv2.VideoWriter_fourcc(*'MJPG'), float(12), size) # for test data
out = cv2.VideoWriter(f'../pyrealsense2/box_video_{name_file}.mp4',cv2.VideoWriter_fourcc(*'MJPG'), float(11), size)# en arguement N° 0, specifier le path de sauvgarde de la vidéo

for i in range(len(seq_images)):
    img = cv2.cvtColor(seq_images[i],cv2.COLOR_BGR2RGB)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, 'frame' + str(i), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(img)
out.release()