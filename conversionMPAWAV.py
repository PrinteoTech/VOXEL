import os
import subprocess

# Dossier contenant les fichiers MP3
dossier_mp3 = './segments_audio'
# Dossier où enregistrer les fichiers WAV
dossier_wav = './segments_wav'

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(dossier_wav):
    os.makedirs(dossier_wav)

# Parcourir tous les fichiers dans le dossier MP3
for fichier in os.listdir(dossier_mp3):
    if fichier.endswith('.mp3'):
        chemin_mp3 = os.path.join(dossier_mp3, fichier)
        chemin_wav = os.path.join(dossier_wav, fichier.replace('.mp3', '.wav'))
        
        # Convertir MP3 en WAV avec FFmpeg
        commande = ['ffmpeg', '-i', chemin_mp3, chemin_wav]
        subprocess.run(commande)
        print(f'Conversion de {fichier} terminée')

print('Toutes les conversions sont terminées !')
