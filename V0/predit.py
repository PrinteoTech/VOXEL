import numpy as np
# import tensorflow as tf 
# import math 

alphabet = "abcdefghijklmnopqrstuvwxyz"
nbLetter = {char: idx for idx, char in enumerate(alphabet)}

nbToLetter = {idx: char for char, idx in nbLetter.items()}

testnbLetter = "chat"
tableau = [nbLetter[letter] for letter in testnbLetter]

vTab = len(tableau)
i = 0
x = []

for i in range(vTab - 1): 
    x = x + [tableau[i]]
    predict = tableau[i + 1]
    y = predict
    i = i+1

x_np = np.array(x)
x_np = x_np.reshape(1, 3, 1)
y_np = np.array(y)
print("X:", x_np)
print("Y:", y_np)


# testNbToLetter = [2, 7, 0, 19]
# texte1 = "".join(nbToLetter[num] for num in testNbToLetter)
# print(texte1) 