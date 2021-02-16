# gradientní metoda optimalizace

import matplotlib.pyplot as plt
import numpy as np

# příprava dat pro optimalizaci
X = np.array([1, 2])
Y = np.array([4, 8])



class Regressor():

    def __init__(self):
        # Náhodná inicializace
        self.a = np.random.random()
        self.b = np.random.random()

    def pred(self, x, a, b):
        # Lineární regrese
        y = a * x + b
        return y

    def mse(self, y, out):
        # cost funkce Mean Squared Error
        # loss = ((výsledek - žádaný_výsledek) ** 2) / n
        error = (y - out) ** 2
        return error.mean()

# hlavní učící smyčka
r = Regressor()
# Maximální krok
krok = 0.8
# Rychlost učení => 0.01 * krok
lr = 0.01

# Kolikrát iterujeme data
for i in range(5000):
    # Jednotlivé body
    for x, y in zip(X, Y):

        # Možnosti a jejich výsledky => loss
        moznosti = []
        vysledky = []

        # Vytvoříme 4 možné pohyby v rozsahu 0 až krok
        for i in (krok * np.random.random(), -krok * np.random.random()):
            for j in (krok * np.random.random(), -krok * np.random.random()):
                moznosti.append([i, j])

        # Vyzkoušíme tyto možnosti a výsledek uložíme
        for m in moznosti:
            loss = r.mse(y, r.pred(x, r.a + m[0], r.b + m[1]))
            vysledky.append(loss)

        # Pokud je nejlepší výsledek horší než stávající, přeskočíme ho
        if min(vysledky) > r.mse(y, r.pred(x, r.a, r.b)):
            continue

        # Pokud je výsledek vhodný, optimalizujeme a a b
        r.a += moznosti[vysledky.index(min(vysledky))][0] * lr
        r.b += moznosti[vysledky.index(min(vysledky))][1] * lr

out = r.pred(X, r.a, r.b)
loss = r.mse(Y, out)
print("Výsledky: {} Loss: {}".format(str(out), loss))

# finální graf
plt.plot(X, r.pred(X, r.a, r.b))
plt.scatter(X, Y, c="r")
plt.show()
