import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras
from keras import layers, models
import os
import numpy as np
from pathlib import Path
from keras.regularizers import l2

import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
#uncomment for MACOSx
'''os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
'''

def affiche_evolution_apprentissage(history):
    # résumé de l'historique pour loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erreur du modèle')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['apprentissage', 'test'], loc='upper right')
    #Path to save model learning curve
    plt.savefig("_3dNet_Save_3/Model_Error.png")

#Path to saved numpy RGB patches and 3D coordinates
Group_scenes = ["data10","data11","data12","data13","data14","data15"]

a10 = np.load(f"{Group_scenes[0]}_Patches_RGB_list.npy")
b10 = np.load(f"{Group_scenes[0]}_Patches_3D_list.npy")
#c10 = np.load(f"{Group_scenes[0]}_Patches_2D_list.npy")

a11 = np.load(f"{Group_scenes[1]}_Patches_RGB_list.npy")
b11 = np.load(f"{Group_scenes[1]}_Patches_3D_list.npy")
#c11 = np.load(f"{Group_scenes[1]}_Patches_2D_list.npy")

a12 = np.load(f"{Group_scenes[2]}_Patches_RGB_list.npy")
b12 = np.load(f"{Group_scenes[2]}_Patches_3D_list.npy")
#c12 = np.load(f"{Group_scenes[2]}_Patches_2D_list.npy")

a13 = np.load(f"{Group_scenes[3]}_Patches_RGB_list.npy")
b13 = np.load(f"{Group_scenes[3]}_Patches_3D_list.npy")
#c13 = np.load(f"{Group_scenes[3]}_Patches_2D_list.npy")

a14 = np.load(f"{Group_scenes[4]}_Patches_RGB_list.npy")
b14 = np.load(f"{Group_scenes[4]}_Patches_3D_list.npy")
#c14 = np.load(f"{Group_scenes[4]}_Patches_2D_list.npy")

#training data
x_train =  np.concatenate((a10,a11,a12,a13),0)
y_train =  np.concatenate((b10,b11,b12,b13),0)

#testing data
x_test =  a14 #np.concatenate((a14,a15),0)
y_test =  b14 #np.concatenate((b14,b15),0)

print("max - min Train data: ",y_train.max()," , ", y_train.min())
print("max - min test data: ",y_test.max()," , ", y_test.min())

# normalize rgb patches data (data preprocessing)
X_train = preprocess_input(x_train)
X_test = preprocess_input(x_test)

# noramlize 3D labels data
Y_train = y_train.reshape(-1,3)
Y_test = y_test.reshape(-1,3)

X_train.shape,Y_train.shape,X_test.shape,Y_test.shape


# CNN parameters
nb_kernels = 16
val_batch_size = 2048
val_epcohs = 500
val_dropout = 0.5
val_weight_decay = 1e-5

#Sequential Model
_3dNet_ = models.Sequential()

#Convolutional layers
_3dNet_.add(layers.convolutional.Conv2D(nb_kernels,(3,3),padding='valid',kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu',input_shape=(50,50,3)))
_3dNet_.add(layers.BatchNormalization())
_3dNet_.add(layers.MaxPooling2D((3,3),strides=(2,2)))


_3dNet_.add(layers.convolutional.Conv2D(nb_kernels*2,(3,3),padding='same',kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
_3dNet_.add(layers.BatchNormalization())
_3dNet_.add(layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))

_3dNet_.add(layers.convolutional.Conv2D(nb_kernels*4,(3,3),kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
_3dNet_.add(layers.BatchNormalization())
_3dNet_.add(layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))

_3dNet_.add(layers.convolutional.Conv2D(nb_kernels*8,(3,3),kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu',padding='same'))
_3dNet_.add(layers.BatchNormalization())
_3dNet_.add(layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))

_3dNet_.add(layers.convolutional.Conv2D(nb_kernels*16,(3,3),kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu',padding='same'))
_3dNet_.add(layers.BatchNormalization())
_3dNet_.add(layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))

#MLP Layers
_3dNet_.add(layers.Flatten())
_3dNet_.add(layers.Dense(512,kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
_3dNet_.add(layers.Dropout(val_dropout))
_3dNet_.add(layers.Dense(256,kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
_3dNet_.add(layers.Dropout(val_dropout))

#Regression Layers
_3dNet_.add(layers.Dense(3,activation='linear'))

#Compiling Model
_3dNet_.compile(optimizer=keras.optimizers.Adam(),loss='mse')

_3dNet_.summary()

#Fitting Model
history = _3dNet_.fit(X_train, Y_train, 
   validation_data=(X_test, Y_test), 
   epochs=val_epcohs, 
   batch_size=val_batch_size)

# Enregistrer la structure du réseau
model_structure = _3dNet_.to_json()
f = Path("_3dNet_Save_3/_3dNet__structure.json")
f.write_text(model_structure)
# Sauvegarde des poids appris par le réseau
#Path to save
_3dNet_.save_weights("_3dNet_Save_3/_3dNet__weights.h5")
#Save History
#Path to save
np.save("_3dNet_Save_3/_3DNet_History",history.history)
#Save Plot
affiche_evolution_apprentissage(history)






