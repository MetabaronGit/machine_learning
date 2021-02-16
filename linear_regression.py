# lineární regrese
# lineární funkce y = a * x + b

# import celých, nebo pouze částí knihoven pod aliasem
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definujeme si vstupní data, v tomto případě 1 a 3
x = np.array([[1], [3]])
# Definujeme si výstup, který chceme.
# Pokud je vstup 1, chceme výstup 3
# Pokud je vstup 3, chceme výstup 5
y = np.array([[3], [5]])

#print(x.base)   #jestli má pole vlastní data (None), nebo je to jen odkaz
#print(x.dtype)  #typ prvků v poli
#print(x.ndim)   #kolik dimenzí má pole

# Vytvoříme regressor neboli lineární funkci
reg = LinearRegression()
# Optimalizujeme
reg.fit(x, y)
# jak si funkce vede ověříme metodou score
print(reg.score(x, y))

# Pomocí funkce plot() se vykreslí čárový graf
plt.plot(x, reg.predict(x))
# Pomocí funkce scatter() vykreslíme jednotlivé body v grafu
# V tomto případě body [1, 3] a [3, 5]
plt.scatter(x, y, c="r") # v parametru c specifikujeme styl bodů, chceme je červené
plt.show()  #finální vykreslení grafu na obrazovku

# hledané hodnoty
a = reg.coef_
b = reg.intercept_
print("Sklon je {0} a bias je {1}".format(a, b))



# regrese na reálných datech
# reálná data ze souboru csv
data = pd.read_csv("gold_prices_monthly_csv.csv")
print(data.head()) #vypsání hlavičky (prvních pět řádků souboru csv)

# numpy pole od 0 do 833 ve dne 14.6.2019
x = np.arange(len(data))
# Ceny převedeme na numpy pole
y = data["Price"].to_numpy()

#přerovnání formátu dat do požadovaného vzoru
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
print(x.shape)

reg = LinearRegression()
reg.fit(x, y)
reg.score(x, y)

# zobrazení grafu
plt.plot(x, reg.predict(x))
plt.scatter(x, y, c="r")
plt.show()

# předpovídaná hodnota pro index 906 (1.7.2025)
print(reg.predict([[906]]).item())