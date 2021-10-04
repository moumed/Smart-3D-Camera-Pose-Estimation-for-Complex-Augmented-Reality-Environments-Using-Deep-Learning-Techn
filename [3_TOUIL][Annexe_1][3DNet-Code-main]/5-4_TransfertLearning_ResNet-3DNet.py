import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input
import keras
from keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.regularizers import l2
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

def affiche_evolution_apprentissage(history):
    # résumé de l'historique pour loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erreur du modèle')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['apprentissage', 'test'], loc='upper right')
    plt.savefig("_3dNet_Save_3/Model_Error.png")
    
Group_scenes = ["data10","data11","data12","data13","data14","data15"]


a10 = np.load(f"{Group_scenes[0]}_Patches_RGB_list.npy")
b10 = np.load(f"{Group_scenes[0]}_Patches_3D_list.npy")

a11 = np.load(f"{Group_scenes[1]}_Patches_RGB_list.npy")
b11 = np.load(f"{Group_scenes[1]}_Patches_3D_list.npy")

a12 = np.load(f"{Group_scenes[2]}_Patches_RGB_list.npy")
b12 = np.load(f"{Group_scenes[2]}_Patches_3D_list.npy")

a13 = np.load(f"{Group_scenes[3]}_Patches_RGB_list.npy")
b13 = np.load(f"{Group_scenes[3]}_Patches_3D_list.npy")

a14 = np.load(f"{Group_scenes[4]}_Patches_RGB_list.npy")
b14 = np.load(f"{Group_scenes[4]}_Patches_3D_list.npy")

x_train =  np.concatenate((a10,a11,a12,a13),0)
y_train =  np.concatenate((b10,b11,b12,b13),0)

x_test =  a14 #np.concatenate((a14,a15),0)
y_test =  b14 #np.concatenate((b14,b15),0)

print("max - min Train data: ",y_train.max()," , ", y_train.min())
print("max - min test data: ",y_test.max()," , ", y_test.min())

X_train = preprocess_input(x_train)
X_test = preprocess_input(x_test)

Y_train = y_train.reshape(-1,3)
Y_test = y_test.reshape(-1,3)


# Chargement du modèle RNA pré-entrainé sans la couche de prédiction
features_extraction_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
include_top=False,input_shape=X_train.shape[1:])
#Freez the FE Model
#features_extraction_model.trainable = False
for lay in features_extraction_model.layers:
    lay.trainable = False

#Création du model séquentiel MLP

# MLP parameters
nb_neurones = 2048
val_batch_size = 2048
val_epcohs = 500
val_dropout = 0.5
val_weight_decay = 1e-5

#MLP Layers
Regression_model = models.Sequential()
#Regression_model.add(layers.InputLayer(input_shape=X_train.shape[1:],name='Input_Layer'))
Regression_model.add(features_extraction_model)
Regression_model.add(layers.Flatten())
Regression_model.add(layers.Dense(512,kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
Regression_model.add(layers.Dropout(val_dropout))
Regression_model.add(layers.Dense(256,kernel_regularizer=l2(val_weight_decay), bias_regularizer=l2(val_weight_decay),activation='relu'))
Regression_model.add(layers.Dropout(val_dropout))
#Regression Layers
Regression_model.add(layers.Dense(3,activation='linear'))

#Compiling Model
Regression_model.compile(optimizer=keras.optimizers.Adam(),loss='mse')
Regression_model.summary()

#Fitting Model
Regression_model_history = Regression_model.fit(X_train,Y_train,
                            epochs=val_epcohs,
                             validation_data=(X_test,Y_test),
                             batch_size=val_batch_size)
# Save neural network
from pathlib import Path
# Enregistrer la structure du réseau
model_structure = Regression_model.to_json()
f = Path("_3dNet_Save_3/_3dNet__structure.json")
f.write_text(model_structure)
# Sauvegarde des poids appris par le réseau
Regression_model.save_weights("_3dNet_Save_3/_3dNet__weights.h5")
#Save History
np.save("_3dNet_Save_3/_3DNet_History",Regression_model_history.history)
#Save Plot
affiche_evolution_apprentissage(Regression_model_history)
