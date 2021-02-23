import numpy as np
from sklearn.tree import DecisionTreeClassifier # Importujeme Rozhodovací strom pro klasifikaci
from sklearn.tree import plot_tree # Importujeme funkci pro vizualizaci stromu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# načtení datasetu a rozdělení na cvičnou a testovací část
X, Y = load_iris(True) # Načteme si Iris dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# vytvoření stromu a přizpůsobení ho datům (fit)
strom = DecisionTreeClassifier()
strom = strom.fit(X_train, Y_train)

# test úspěšnosti stromu
print(strom.score(X_test, Y_test))

# vizualizace stromu
plt.figure(figsize = (15,10)) # Vytvoříme "plátno", aby byla vizualizace větší v Jupyter Noteboku
plot_tree(strom, feature_names=["petal length", "petal width", "sepal length", "sepal width"])
plt.show()