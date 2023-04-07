from sklearn.model_selection import learning_curve,cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
from joblib import load
from tempfile import TemporaryFile
outfile = TemporaryFile()
from features_functions import compute_features
from partie_2_1_2 import Decoupe_camion, Decoupe_voiture
from pydub import AudioSegment
import os
import librosa
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm


""" Importation du model"""

scaler = load("SCALER_")
model = load("SVM_MODEL_")

""" Prédiction et Rapport de classification plus import des données lié au model """

X_test = np.load('X_test.npy')
learningFeatures_scaled_2 = np.load('learningFeatures_scaled_2.npy')
y_test = np.load('y_test.npy')
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

prediction = model.predict(learningFeatures_scaled_2)
print(classification_report(y_test, prediction))

""" Création d'une fonction pour sortir une learning curve """

def plot_learning_curve (nom_model, jeu_entrainement, target_entrainement) :
    # Génération de la courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(nom_model, jeu_entrainement, target_entrainement, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calcul des scores moyens et des écarts types pour les ensembles d'entraînement et de test
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Tracé de la courbe d'apprentissage avec redimensionnement de la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Courbe d'apprentissage")
    ax.set_xlabel("Taille de l'ensemble d'entraînement")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    ax.legend(loc="best")
    plt.show()

""" Plot la learning curve """

plot_learning_curve(model,X_train,y_train)

""" Amélioration des résultats en divisant par 20 et non par 10 avec réentrainement avec les nouvelles données """

liste_car = ['Yz_TbckEJQpI_30.000_40.000','Y-6sfz8rZ6qM_20.000_30.000','Y-7B8biMUFq8_0.000_9.000',
             'Y-0khyUGUoto_30.000_40.000','Y-7jJtGYm1_U_30.000_40.000',
             'Y-0pX5r9yfXk_30.000_40.000','Y-7lhLBnJtGs_30.000_40.000','Y-7y2MyFd9p4_30.000_40.000',
             'Y-1uR6toEH3A_80.000_90.000','Y-7yHd8yVL7o_10.000_20.000',]
liste_truck = ['Y3nUKdavRnik_7.000_17.000','Y-74wyz6FYhg_18.000_28.000','Y3oKPh1-DpfU_10.000_20.000',
               'Y-8SGyYoVIPU_30.000_40.000','Y-ARxJXMlSl0_30.000_40.000',
               'Y-AoDMSvnACY_30.000_40.000','Y-BCOY2XYUas_30.000_40.000','Y-BXMX3yEPto_30.000_40.000',
               'Y-3yB8Z1-Ow8_30.000_40.000','Y-458eoazpK8_28.000_38.000','Y-1V2ReGbbtM_7.000_17.000']

dossier = "donnee_transforme_bis/"
nb = 20
Decoupe_voiture(nb, dossier)
Decoupe_camion(nb, dossier)

filenames_bis = os.listdir("notebook_audio/partie_2/donnee_transforme_bis/")

# LOOP OVER THE SIGNALS
#for all signals:
learningFeatures =[]
input_sig = []
for filename in filenames_bis:
    file = "notebook_audio/partie_2/donnee_transforme_bis/{}".format(filename)
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

v = np.zeros(100)
c = np.ones(120)
learningLabels = np.concatenate((v, c))

learningFeatures = np.asarray(learningFeatures).reshape(220, 71)
# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)

X_train, X_test, y_train, y_test = train_test_split(learningFeatures, learningLabels, test_size=0.3, random_state=2)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(learningFeatures)
learningFeatures_scaled = scaler.transform(learningFeatures)
learningFeatures_scaled_2 = scaler.transform(X_test)
model.fit(learningFeatures_scaled, learningLabelsStd)

# # Export the scaler and model on disk
# dump(scaler, "SCALER_bis")
# dump(model, "SVM_MODEL_bis")

prediction = model.predict(learningFeatures_scaled_2)
print(classification_report(y_test, prediction))

plot_learning_curve(model,X_train,y_train)