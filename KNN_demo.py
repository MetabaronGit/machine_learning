import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # Budeme používat pro sčítání sousedů
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split


class KNN():

    def __init__(self, dataset_x, dataset_y, k):
        assert len(dataset_x) == len(dataset_y) # Počet vzorků se shoduje
        assert len(dataset_x) >= k # K není větší než počet vzorků
        self.dataset = np.array( list( zip(X, Y) ) )
        self.k = k

    def predict(self):
        pass

# vytvoření demo datasetu
X = [
    [2, 3],
    [5, 6],
    [8, 2],
    [6, 3]
    ]

Y = [
    2,
    2,
    1,
    1,
    ]

print("Vlastnosti:", X)
print("Label:", Y)

colors = {1: "r", 2: "b"}

scatter_data = list(zip(*X))

# Každému bodu přiřadíme barvu podle skupiny 1 => červená 2 => modrá
plt.scatter(scatter_data[0], scatter_data[1], c=[colors[i] for i in Y])
plt.show()

# vytvoření modelu
model = KNN(X, Y, 3)
# warning
# self.dataset = np.array( list( zip(X, Y) ) )

# vypsání vnitřní reprezentace
print(model.dataset)

