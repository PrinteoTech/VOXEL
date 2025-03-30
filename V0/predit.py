import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

# import math 

alphabet = "abcdefghijklmnopqrstuvwxyz-"
nbLetter = {char: idx + 1 for idx, char in enumerate(alphabet)}

nbToLetter = {idx: char for char, idx in nbLetter.items()}
                                     
tableauMots = ["chat", "maison", "soleil", "arbre", "travail", "musique", 'amour', "electrique", "clavier", "origine"]
print(len(tableauMots))

j = 0

while j < len(tableauMots):  
    Letter = tableauMots[0 + j]

    if len(Letter) < 10:
        while len(Letter) < 10:
            Letter += "-"

    print(Letter)

    tableau = [nbLetter[letter] for letter in Letter]

    seq_length = 3
    y = []
    x = []

    for i in range(len(tableau) - seq_length): 
        x.append(tableau[i : i + seq_length]) 
        y.append(tableau[i + seq_length])      

    x_np = np.array(x)
    y_np = np.array(y)

    for x_seq, y_val in zip(x_np, y_np):
        print(f"X: {x_seq.flatten()}  Y: {y_val}")
    j += 1

# testNbToLetter = [2, 7, 0, 19]
# texte1 = "".join(nbToLetter[num] for num in testNbToLetter)
# print(texte1) 

model = Sequential()
model.add(Embedding(
    input_dim=28,  
    output_dim=32, 
    embeddings_initializer="glorot_uniform",
    embeddings_regularizer=tf.keras.regularizers.l2(0.001),
    embeddings_constraint=tf.keras.constraints.max_norm(2.0),
    mask_zero=False,  
    input_length=3))

test_input = np.array([x_np[0]])

embedded_output = model.predict(test_input)

print("Shape des embeddings:", embedded_output.shape)
print("Valeur des embeddings:", embedded_output)
