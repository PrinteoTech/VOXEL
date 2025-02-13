import re
import subprocess
import os

#nom des fichier

fichier_audio = "data1h.mp3"
fichier_srt = "data1h.srt"
dossier_sortie = "segments_audio"

#Creer un dossier pour les segments audios
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

#expression pour extraire les timestamps
pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)")

#lecture du fichier srt
with open(fichier_srt, "r", encoding="utf-8") as f:
    contenu = f.read()

#trouver les lignes du srt
lignes = pattern.findall(contenu)

#decoupe l'audio pour chaque ligne
for numero, debut, fin, text in lignes:
    #timestamps pour ffmepg
    debut = debut.replace(",", ".")
    fin = fin.replace(",", ".")

    #nom du fichier de sortie
    fichier_sortie = os.path.join(dossier_sortie, f"segment_{numero}.mp3")

    #commande ffmpeg
    commande = [
        "ffmpeg",
        "-i", fichier_audio,
        "-ss", debut,
        "-to", fin,
        "-c", "copy",
        fichier_sortie
    ]

    #execute la commande
    print(f"Decoupe de {debut} a {fin} : {fichier_sortie}")
    subprocess.run(commande)

print("Decoupage Termine ! ")
