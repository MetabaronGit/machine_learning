from sklearn import neighbors # Samotný KNN algoritmus
# from sklearn import preprocess # Normalizujeme data
from sklearn.model_selection  import train_test_split # Rozdělíme data na "trénovací" a testovací 80% ku 20%
import numpy as np
from sklearn.datasets import load_iris # Funkce pro načtění datasetu

X, Y= load_iris(True) # parametr určuje, zda chceme data rozdělit na X a Y, což my chceme

print(X[0], Y[0])

print(np.unique(Y)) #unik8tn9 hodnoty

# Rozdělíme si data na trénovací část a testovací (80% ku 20%):
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# vypíšeme si jejich tvar:
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape