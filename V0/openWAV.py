import csv
import librosa
import sounddevice as sd

# Lecture du fichier metadata.csv
with open('metadata.csv', 'r') as file:
    reader = csv.reader(file, delimiter='|')
    first_row = next(reader)  # Récupère la première ligne
    filename = first_row[0]  # Nom du fichier
    text = first_row[1]      # Texte
    print(f"Le nom du fichier est: {filename}, son texte est: {text}")

# Charger le fichier audio avec librosa
audio_data, sample_rate = librosa.load(filename, sr=None)  # sr=None pour conserver le taux d'échantillonnage original
 
# Lecture et lecture de l'audio avec sounddevice
sd.play(audio_data, sample_rate)  # Utiliser 'audio_data' et 'sample_rate' de librosa
status = sd.wait()  # Attendre la fin de la lecture
