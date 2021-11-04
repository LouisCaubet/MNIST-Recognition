import random
import numpy as np

import time

# Fonction sigmoid vectorielle rapide
from scipy.special import expit as sigmoid

import mnist_loader


# Fonction d'activation vectorielles

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


class Reseau:
    """Représente un réseau de neurones"""

    def __init__(self, nb_couches, neurones_par_couche, nb_entrees, nb_sorties):
        """Construit un réseau de neurones avec nb_couches couches cachées de 
           neurones_par_couches neurones, avec une couche d'entrée de nb_entrees 
           neurones et une couche de sortie de nb_sorties neurones"""        
        
        # poids est une liste de matrices contenant les poids pour chaque couche
        self.poids = list()
        # idem pour biais
        self.biais = list()
        
        # Copie des paramètres en champs.
        self.nb_couches, self.neurones_par_couche, self.nb_entrees, self.nb_sorties = nb_couches, neurones_par_couche, nb_entrees, nb_sorties


        # Initialisation avec des valeurs aléatoires

        # Première couche
        self.poids.append(np.random.randn(neurones_par_couche, nb_entrees))
        self.biais.append(np.random.randn(neurones_par_couche, 1))

        # couches cachées
        for couche in range(1, nb_couches):
            self.poids.append(np.random.randn(neurones_par_couche, neurones_par_couche))
            self.biais.append(np.random.randn(neurones_par_couche, 1))

        # couche de sortie
        self.poids.append(np.random.randn(nb_sorties, neurones_par_couche))
        self.biais.append(np.random.randn(nb_sorties, 1))

    
    def normalise_entree(self, entree):
        """Fonction d'activation particulière pour transformer l'entrée dans [0,1]. 
           Les valeurs d'entrée sont des couleurs, donc dans [0, 255]"""
        return entree / 255

    def feedforward(self, entree):
        """Détermine la sortie du réseau pour l'entrée input, 
        en mémorisant les entrées pondérées et sorties intermédiaires."""

        # Stocke les entrées pondérées
        entrees_ponderees = []
        # Stocke les sorties
        sorties = [self.normalise_entree(entree)]
        
        # Propagation
        for couche in range(self.nb_couches+1):
            entrees_ponderees.append(np.dot(self.poids[couche], sorties[couche]) + self.biais[couche])
            sorties.append(sigmoid(entrees_ponderees[couche]))

        return (entrees_ponderees, sorties)

    def execute(self, entree):
        _, sorties = self.feedforward(entree)
        return sorties[self.nb_couches + 1]


    def retropropagation(self, entree, sortie_attendue):
        """Détermine le gradient de la fonction coût par l'algorithme de retropropagation"""

        # Calcul du gradient

        # nabla_w est une liste de matrices (même forme que poids mais avec des dérivées partielles)    
        nabla_w = [np.zeros(np.shape(w)) for w in self.poids]

        # nabla_b est de la forme de biais mais avec des dérivées partielles
        nabla_b = [np.zeros(np.shape(b)) for b in self.biais]


        # Phase 1 : Feedforward
        (entrees_ponderees, sorties) = self.feedforward(entree)   

        # Erreur en sortie
        erreur = (sorties[-1] - sortie_attendue) * sigmoid_prime(entrees_ponderees[-1])

        # Gradient en sortie
        nabla_b[-1] = erreur
        nabla_w[-1] = np.dot(erreur, sorties[-2].transpose())

        # Phase 2 : Rétropropagation    
        for couche in range(2, self.nb_couches + 2):
            # Retropropagation de l'erreur
            erreur = np.dot(self.poids[-couche+1].transpose(), erreur) * sigmoid_prime(entrees_ponderees[-couche])
            # Calcul du gradient
            nabla_w[-couche] = np.dot(erreur, sorties[-couche-1].transpose())
            nabla_b[-couche] = erreur 

        return (nabla_w, nabla_b)


    def descente_gradient(self, batch, eta):
        """Applique la descente de gradient sur le réseau avec batch comme set d'entrainement 
           et eta comme pas d'apprentissage"""        

        # On calcule la moyenne des gradients partiels sur batch
        grad_w = [np.zeros(w.shape) for w in self.poids]
        grad_b = [np.zeros(b.shape) for b in self.biais]

        for i in range(len(batch)):
            (entree, sortie_attendue) = batch[i]
            nabla_w, nabla_b = self.retropropagation(entree, sortie_attendue)

            grad_w = [grad_w[j] + nabla_w[j] for j in range(len(grad_w))]
            grad_b = [grad_b[j] + nabla_b[j] for j in range(len(grad_b))]
           

        # On applique la descente de gradient
        self.poids = [self.poids[i] - (eta/len(batch))*grad_w[i] for i in range(len(self.poids))]
        self.biais = [self.biais[i] - (eta/len(batch))*grad_b[i] for i in range(len(self.biais))]


    def SGD(self, set_entrainement, set_test, epochs, taille_batch, eta):
        """Execute une descente de gradient stochastique avec les arguments suivants:
             - set_entrainement: l'ensemble des données d'entrainement.
             - set_test: ensemble des données de validation          
             - epochs: le nombre d'itérations de descente de gradient
             - taille_batch: la taille des batchs souhaitée
             - eta: le pas d'apprentissage"""

        # On évalue le réseau généré aléatoirement
        print("Score du réseau aléatoire :", self.evalue(set_test), "/", len(set_test))

        for i in range(epochs):
            # On mélange le set d'entrainement
            random.shuffle(set_entrainement)
            
            # On le découpe en batchs d'entrainement
            batchs = [set_entrainement[k:k+taille_batch] for k in range(0, len(set_entrainement), taille_batch)]
            
            print("Epoch:", i)

            # On exécute la descente de gradient sur chacun des batchs
            for j in range(len(batchs)):
                self.descente_gradient(batchs[j], eta)
                if j%100 == 0:    
                    print("    Batch:", j, "/", len(batchs), end="\r")
            
            print("    Batch:", len(batchs), "/", len(batchs), end="\r")
            print()
            score = self.evalue(set_test)
            print("Score:", score, "/", len(set_test), " : ", score/len(set_test)*100, " %")


    def evalue(self, set_test):
        """Évalue la performance du réseau sur le set de test."""

        test_results = [(np.argmax(self.execute(x)), y) for (x,y) in set_test]
        return np.sum(int(x == y) for x,y in test_results)


# Code d'exécution

start = time.time()

# On charge mnist
set_entrainement, set_validation, set_test = mnist_loader.load_mnist()

# on construit le réseau
reseau = Reseau(2, 15, 784, 10)

# on entraine le réseau sur 30 epochs, avec des mini-batchs de 10 et eta=0.3
reseau.SGD(set_entrainement, set_validation, 30, 10, 3)

# on evalue le reseau sur l'ensemble d'evaluation
print("Evaluation :", reseau.evalue(set_test), "/", len(set_test))

# Temps d'execution
print()
print("Temps total d'exécution: ", int(time.time() - start))

# 71 lines of non-comment code.