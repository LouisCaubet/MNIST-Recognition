from mnist import MNIST
import numpy as np

def load_mnist():

    # On charge les fichiers de MNIST avec la bibliothèque mnist.
    mndata = MNIST('./data')
    mndata.gz = True
    img_entr, labels_entr = mndata.load_training()
    img_test, labels_test = mndata.load_testing()

    # On va transformer les données brutes en données adaptées à notre algorithme d'entrainement
    
    # On découpe l'ensemble 'training' en set d'entrainement et set de validation
    split = int(len(img_entr) * 0.9)
    entrees_entrainement = [np.reshape(img_entr[i], (784,1)) for i in range(split)]
    resultats_entrainement = [matrice_resultat(labels_entr[i]) for i in range(split)]

    entrainement = list(zip(entrees_entrainement, resultats_entrainement))

    entrees_validation = [np.reshape(img_entr[i], (784,1)) for i in range(split, len(img_entr))]
    resultats_validation = [labels_entr[i] for i in range(split, len(img_entr))]

    validation = list(zip(entrees_validation, resultats_validation))

    entrees_test = [np.reshape(img_test[i], (784,1)) for i in range(len(img_test))]

    evaluation = list(zip(entrees_test, labels_test))

    return (entrainement, validation, evaluation)
    

def matrice_resultat(j):
    """Renvoie un vecteur 10x1 avec un coefficient 1 à la je position
    et des zeros ailleurs. Ceci permet de convertir un chiffre en sortie
    attendue du réseau."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e