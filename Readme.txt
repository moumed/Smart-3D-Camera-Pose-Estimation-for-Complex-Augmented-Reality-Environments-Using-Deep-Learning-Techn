0 - box_train -> _3dNet_Save_3 : contient les poids du modèles 3DNet entrainé.

1 - Acquisition de données avec le code dans le fichier : 1_Recorde_RGBD_align-depth2color.py

2 - Après l’acquisition des images RGB avec le module 1 : créer une séquence vidéo à partir de l’ensemble d’images acquises avec la D435i, pour faire le suivi KLT : 2_frames_to_Video.py

3 - Faire le suivi KLT et calcul des poses avec PnP : 3_Savee_poses_KLT_P3P.py

3-bis - Code pour la calibration de la caméra (Facultatif) : 3_bis_Camera_calibration.py

4 - Faire le test sur les poses calculées : 4_Test_Saved_KLT-P3P_Poses.py

4_2 - Faire les test sur les points SURF dans nos données acquises : 4_2_Test_SURF_KP.py 

4_3 - Tester les coordonées 3D des points SURF et faire la reprojection : 4_3_Test_SURF_Reproject_3DCorrds.py

5 - Creation du dataset Patches RGB + 3D coordinates : 5-Create_Patches_RGB-3Dcoords_Data.py

5_2 - Creation du model 3DNet et apprentissage : 5-2-_3DNet_CNN_Regression_Model.py

5_3 - Tester le modèle 3DNet après fin d’apprentissage et sauvegarder les données pour l’évaluation : 5-3_Test_3DNet_and_Save_DATA_For_Evaluation.py

5_4 - Transfert learning avec ResNet : 5-4_TransfertLearning_ResNet-3DNet.py

6 - Code d’exécution en Temps réel avec D435i et 3DNet : 6-3DNetRealTime.py

7 - Evaluation du 3DNet et l’estimation de pose : 7-box_3DNet_Evaluation.py

8 - Packages_3DNet_Env.yml : la configuration de l’environnement anaconda.

9 - Python_V1.sh : le fichier bash à exécuter sur Cassiopee.

Sur Cassiopee : 
- Créer un dossier pour chaque apprentissage 
- s’assurer que tous les packages sont bien installés
- Transferer le dataset Numpy contenant patch rgb et patch 3D coordinates
- Créer un dossier vide du nom « _3DNet_Save »
- Creer le fichier bash : « Python_V1.sh »
- executer la commande :    sbatch  Python_V1.sh

Listes des packages : 
- Pyrealsense2
- numpy = 1.20.2
- OpenCV + OpencvContrib = 4.5.2
- glob2 = 0.7
- matplotlib = 3.4.2
- keras = 2.3.1
- tensorflow = 2.0.0
- scipy = 1.6.2
- sciait-learn = 0.24.2
- pretty-table = 2.1.0
  

NB : Il faut faire attention au chemin pour le chargement et la sauvegarde des données dans les codes, penser à les vérifier et les modifier selon votre besoin pour éviter les bugs inutiles. Parfois certains dossiers doivent être créés ou modifier les paths  dans le code vers votre propre dossier ou vous souhaiter sauvegarder les données. Opencv avec les modules de Opencv-Contrib sont obligatoires pour le projet, il est nécessaire de compiler la librairie from source code en activant le flag OPENCV_ENABLE_NONFREE. Pour la partie transfert learning il est conseillé de téléchargé les poids du ResNet50 sur votre machine et spécifier son chemin dans le code.

Pour toute question m’hésitez pas à nous contacter :
Mohamed-ameziane.touil@etu.univ-amu.fr
Fakhreddine.ababsa@ensam.eu



