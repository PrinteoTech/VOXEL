import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import os
import datetime

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

def predict_next_letter(sequence):
    seq_encoded = np.array([[nbLetter[char] for char in sequence]])
    predicted_index = np.argmax(model.predict(seq_encoded))
    predicted_letter = nbToLetter[predicted_index]
    return predicted_letter

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

model.add(tf.keras.layers.LSTM(
    units=50,
    activation='tanh',
    return_sequences=False))

model.add(tf.keras.layers.Dense(
    units=28,
    activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if os.path.exists("model_best.h5"):
    model.load_weights("model_best.h5")
    print("Poids du modèle chargés !")

model.fit(x_np, y_np, epochs=500, batch_size=1, callbacks=[tensorboard_callback])
model.save("model_best.h5")

sequence_test = "cha"  # Essaie de prédire la lettre après "cha"
predicted = predict_next_letter(sequence_test)
print(f"Séquence test : {sequence_test} -> Lettre prédite : {predicted}")