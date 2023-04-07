from math import inf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import pickle
from sklearn.metrics import classification_report
import pandas as pd
from pydub import AudioSegment
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import learning_curve,cross_val_score
from joblib import dump
from features_functions import compute_features


training_set = pd.read_csv("notebook_audio/training_set.csv")

""" Nettoyage des données en vue d'extraction des fichiers audio voiture et camion """

training_set.rename(columns={'2ceUOv8A3FE': 'nom',
                             '20.000': 'donnée_1',
                             '30.000': 'donnée_2',
                             'Train horn,Train': 'type_target',
                            '/m/0284vy3,/m/07jdr': 'a_sup'}, inplace=True, errors='raise')

training_set['type_target'].value_counts().index.tolist()

""" Localisation des audios voitures et camions """

training_set_loc = training_set.loc[(training_set['type_target'] == 'Car') | 
                                    (training_set['type_target'] == 'Truck')]

""" Regroupement des audios par catégorie, un dataset voiture et un dataset camion"""

grouped_truck = training_set_loc.loc[training_set_loc['type_target'] == 'Truck']
grouped_car = training_set_loc.loc[training_set_loc['type_target'] == 'Car']

""" Ecoute des audios pour déterminer si le son voiture/camion est tout au long de l'enregistrement.
    Etablissement d'une liste pour chaque catégorie"""

liste_car = ['Yz_TbckEJQpI_30.000_40.000','Y-6sfz8rZ6qM_20.000_30.000','Y-7B8biMUFq8_0.000_9.000',
             'Y-0khyUGUoto_30.000_40.000','Y-7jJtGYm1_U_30.000_40.000',
             'Y-0pX5r9yfXk_30.000_40.000','Y-7lhLBnJtGs_30.000_40.000','Y-7y2MyFd9p4_30.000_40.000',
             'Y-1uR6toEH3A_80.000_90.000','Y-7yHd8yVL7o_10.000_20.000',]
liste_truck = ['Y3nUKdavRnik_7.000_17.000','Y-74wyz6FYhg_18.000_28.000','Y3oKPh1-DpfU_10.000_20.000',
               'Y-8SGyYoVIPU_30.000_40.000','Y-ARxJXMlSl0_30.000_40.000',
               'Y-AoDMSvnACY_30.000_40.000','Y-BCOY2XYUas_30.000_40.000','Y-BXMX3yEPto_30.000_40.000',
               'Y-3yB8Z1-Ow8_30.000_40.000','Y-458eoazpK8_28.000_38.000','Y-1V2ReGbbtM_7.000_17.000']

""" Création de 2 fonction pour segmenter les audios voitures et camions """

def Decoupe_voiture(nb, dossier):
    for i, list_truck in enumerate(liste_truck):
        audio = AudioSegment.from_file('notebook_audio/partie_2/unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/' + list_truck + '.wav', format='wav')

        # Déterminer la longueur de chaque segment
        segment_length = len(audio) // nb

        # Découper le fichier audio en segments de longueur égale
        for j in range(nb):
            start = j * segment_length
            end = (j + 1) * segment_length
            segment = audio[start:end]

            # Enregistrer le segment découpé avec un compteur
            segment.export('notebook_audio/partie_2/'+ dossier + 't' +list_truck + '_' + str(j) + '.wav', format='wav')

def Decoupe_camion(nb, dossier):
    for i, list_car in enumerate(liste_car):
        audio = AudioSegment.from_file('notebook_audio/partie_2/unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/' + list_car + '.wav', format='wav')

        # Déterminer la longueur de chaque segment
        segment_length = len(audio) // nb

        # Découper le fichier audio en segments de longueur égale
        for j in range(nb):
            start = j * segment_length
            end = (j + 1) * segment_length
            segment = audio[start:end]

            # Enregistrer le segment découpé avec un compteur
            segment.export('notebook_audio/partie_2/' + dossier + 'c' +list_car + '_' + str(j) + '.wav', format='wav')

""" Création des fichier grâce aux fonctions """

nb = 10
dossier = "donnee_transforme/"
Decoupe_voiture(nb, dossier)
Decoupe_camion(nb, dossier)

""" Récupération des labels de chaque audio """

filenames = os.listdir("notebook_audio/partie_2/donnee_transforme/")

""" Importation des audios, Garde 71 features"""

learningFeatures =[]
input_sig = []
for filename in filenames:
    file = "notebook_audio/partie_2/donnee_transforme/{}".format(filename)
    input_sig, sr = librosa.load(file, sr=None)
    
    # Compute the signal in three domains
    sig_sq = input_sig**2
    sig_t = input_sig / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))
    
    # Compute the features and store them
    features_list = []
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
    features_vector = np.array(features_list)[np.newaxis,:]
    # Store the obtained features in a np.arrays
    learningFeatures.append(features_vector)# 2D np.array with features_vector in it, for each signal

""" Création de la target """

v = np.zeros(100)
c = np.ones(110)
learningLabels = np.concatenate((v, c))

""" train test split plus test des données sur model svm avec export du model"""

learningFeatures = np.asarray(learningFeatures).reshape(210, 71)
# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)

X_train, X_test, y_train, y_test = train_test_split(learningFeatures, learningLabels, test_size=0.3, random_state=42)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(learningFeatures)
learningFeatures_scaled = scaler.transform(learningFeatures)
learningFeatures_scaled_2 = scaler.transform(X_test)
model.fit(learningFeatures_scaled, learningLabelsStd)

# # Export the scaler and model on disk
dump(scaler, "SCALER_")
dump(model, "SVM_MODEL_")

""" Export des données lié au model """

np.save('X_test.npy', X_test)
np.save('learningFeatures_scaled_2.npy', learningFeatures_scaled_2)
np.save('y_test.npy', y_test)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)