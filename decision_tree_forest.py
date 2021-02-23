import numpy as np
from sklearn.ensemble import RandomForestClassifier # Importujeme les rozhodovacích stromů pro klasifikaci
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# načtení datasetu
X, Y = load_iris(True) # Načteme si Iris dataset
# pro učení použijeme jen 50% dat
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5)

# vytvoření lesa a jeho "fitnutí"
les = RandomForestClassifier(n_estimators=5, max_depth=1)
les = les.fit(X_train, Y_train)

# score
print(les.score(X_test, Y_test))