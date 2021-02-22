import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # Budeme používat pro sčítání sousedů
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split


class KNN():
    def __init__(self, dataset_x, dataset_y, k):
        assert len(dataset_x) == len(dataset_y)  # Počet vzorků se shoduje
        assert len(dataset_x) >= k  # K není větší než počet vzorků
        self.dataset = np.array(list(zip(X, Y)))
        self.k = k

    def distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))  # Euklidovská vzdálenost
    # zatím bez váhy jednotlivých sousedů
    # váha je 1 / vzdálenost

    def predict(self, x):
        nearest_points = sorted(self.dataset, key=lambda _x: self.distance(x, _x[0]))[:self.k]
        classes = list(zip(*nearest_points))[1]
        return Counter(classes).most_common(1)[0][0]

    def score(self, x, y):
        right = 0
        total = 0
        for _x, _y in zip(x, y):
            pred = self.predict(_x)
            if pred == _y:
                right += 1
            total += 1
        return right / total

# vytvoření demo datasetu
# vlastnosti = souřadnice x, y
X = [
    [2, 3],
    [5, 6],
    [8, 2],
    [6, 3]
    ]
# label = barva
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

# setřídění pole podle elementu x index 1
# klíč je element a hodnota je počet výskytů
pole = [ [5, 4], [8, 2], [1, 3], [12, 1] ]
print(sorted(pole, key=lambda x: x[1]))

# vrací n elementů s největším počtem výskytů
# jako list tuplů
pole = [ 2, 1, 5, 3, 1, 6, 8, 1, 3, 1 ]
print(Counter(pole).most_common(2))

# iris dataset
# -------------------------------------------------------
X, Y= load_iris(True)
# rozdělení datasetu na cvičnou a testovací část  (80% ku 20%):
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

model = KNN(X_train, Y_train, 5)
print(model.score(X_test, Y_test))