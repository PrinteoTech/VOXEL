import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import os
import datetime
import unicodedata
from sklearn.model_selection import train_test_split  # Import pour la division train/test
from tensorflow.keras.callbacks import EarlyStopping  # Import pour l'arrêt précoce

# Fonction pour enlever les accents
def remove_accents(word):
    return ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')

alphabet = "-abcdefghijklmnopqrstuvwxyz"
nbLetter = {char: idx for idx, char in enumerate(alphabet)}

nbToLetter = {idx: char for char, idx in nbLetter.items()}
                                     
# Liste de mots
tableauMots = [
    "chat", "maison", "soleil", "arbre", "travail", "musique", "amour", "électrique", "clavier", "origine",
    "ordinateur", "voiture", "jardin", "école", "hôpital", "avion", "montagne", "rivière", "galaxie", "planète",
    "papillon", "éléphant", "bateau", "téléphone", "avion", "forêt", "ciel", "montagne", "océan", "brouillard",
    "météorite", "étoile", "lune", "merveille", "énergie", "chimie", "physique", "astronomie", "galaxie", "explorer",
    "ordinateur", "bureau", "film", "lumière", "chanson", "voix", "nuage", "arctique", "planète", "nature", 
    "santé", "robot", "télévision", "réalité", "science", "recherche", "légende", "histoire", "génétique", "progrès",
    "technologie", "innovation", "révolution", "data", "art", "mathématiques", "évolution", "satellite", "horizon",
    "ordinateur", "bulle", "code", "batterie", "hydrogène", "éléctronique", "biodiversité", "réalité", "télévision",
    "robotique", "système", "astronaute", "océanographie", "exploration", "univers", "voiture", "puzzle", "calme",
    "intelligence", "lumière", "saison", "aurore", "voyage", "efficacité", "espace", "modèle", "vélo", "solaire",
    "fusion", "climat", "hydroélectrique", "nucléaire", "énergie", "antimatière", "moteur", "microprocesseur",
    "fusion", "champ", "champ magnétique", "robotique", "turbine", "moteur", "énergie", "climat", "économique",
    "accélérateur", "résistance", "invention", "imagination", "création", "transformation", "mutation", "batterie",
    "planète", "mercure", "jupiter", "mars", "saturne", "venus", "terre", "robotique", "connecté", "intelligente"
]

# Retirer les accents des mots
tableauMots = [remove_accents(word) for word in tableauMots]
print(len(tableauMots))

j = 0

# Préparation des séquences de lettres
x = []
y = []

while j < len(tableauMots):  
    Letter = tableauMots[0 + j]
    Letter = Letter.replace(" ", "-")
    if len(Letter) < 10:
        while len(Letter) < 10:
            Letter += "-"

    print(Letter)

    tableau = [nbLetter[letter] for letter in Letter]

    seq_length = 3
    for i in range(len(tableau) - seq_length): 
        x.append(tableau[i : i + seq_length]) 
        y.append(tableau[i + seq_length])      

    j += 1

x_np = np.array(x)
y_np = np.array(y)

# Afficher les séquences et labels
for x_seq, y_val in zip(x_np, y_np):
    print(f"X: {x_seq.flatten()}  Y: {y_val}")

# Fonction pour prédire la prochaine lettre
def predict_next_letter(sequence):
    seq_encoded = np.array([[nbLetter[char] for char in sequence]])
    predicted_index = np.argmax(model.predict(seq_encoded))
    predicted_letter = nbToLetter[predicted_index]
    return predicted_letter

# Modèle LSTM
model = Sequential()
model.add(Embedding(
    input_dim=28,  
    output_dim=32, 
    embeddings_initializer="glorot_uniform",
    embeddings_regularizer=tf.keras.regularizers.l2(0.001),
    embeddings_constraint=tf.keras.constraints.max_norm(2.0),
    mask_zero=True,  
    input_length=3))

model.add(tf.keras.layers.LSTM(
    units=100,
    activation='tanh',
    return_sequences=False))

model.add(tf.keras.layers.Dense(
    units=28,
    activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Enregistrement des logs pour TensorBoard
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Chargement des poids si disponibles
if os.path.exists("model_best.h5"):
    model.load_weights("model_best.h5")
    print("Poids du modèle chargés !")

# Séparation des données en train et validation (modification ici)
x_train, x_val, y_train, y_val = train_test_split(x_np, y_np, test_size=0.2, random_state=42)  # Modification ajoutée

# Entraînement du modèle avec validation et EarlyStopping (modification ici)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Modification ajoutée
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[tensorboard_callback, early_stopping])  # Modification ajoutée

# Sauvegarde des poids du modèle
model.save("model_best.h5")

# Test sur une séquence donnée
sequence_test = "cha"
predicted = predict_next_letter(sequence_test)
print(f"Séquence test : {sequence_test} -> Lettre prédite : {predicted}")

# Affichage des probabilités des lettres
seq_encoded = np.array([[nbLetter[char] for char in sequence_test]])  # Convertir la séquence en indices
probs = model.predict(seq_encoded)
print(f"Probabilités des lettres : {probs}")
print("______________________________________")
print(f"Index prédict : {np.argmax(probs)}")
print(f"Lettre correspondante : {nbToLetter[np.argmax(probs)]}")

# Affichage du test
print("______________________________________")
print(f"Encodage de 'cha' : {[nbLetter[c] for c in sequence_test]}")
