# lineární regrese
# lineární funkce y = a * x + b

# import celých, nebo pouze částí knihoven pod aliasem
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Definujeme si vstupní data, v tomto případě 1 a 3
x = np.array([[1], [3]])
# Definujeme si výstup, který chceme.
# Pokud je vstup 1, chceme výstup 3
# Pokud je vstup 3, chceme výstup 5
y = np.array([[3], [5]])

print(x.base)   #jestli má pole vlastní data (None), nebo je to jen odkaz
print(x.dtype)  #typ prvků v poli
print(x.ndim)   #kolik dimenzí má pole