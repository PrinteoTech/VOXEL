# import numpy as np
# import tensorflow as tf 
# import math 

alphabet = "abcdefghijklmnopqrstuvwxyz"
nbLetter = {char: idx for idx, char in enumerate(alphabet)}

nbToLetter = {idx: char for char, idx in nbLetter.items()}

testnbLetter = "abcd"
texte1 = "".join(str(nbLetter[letter]) for letter in testnbLetter)
print(texte1)

# testNbToLetter = [2, 7, 0, 19]
# texte1 = "".join(nbToLetter[num] for num in testNbToLetter)
# print(texte1) 