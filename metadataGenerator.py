import os

dossier_wav = "./segments_wav"
dossier_txt = "./segments_textes"
fichier_metadata = "./metadata.csv"

fichiers_wav = sorted([f for f in os.listdir(dossier_wav) if f.endswith('.wav')])

with open(fichier_metadata, 'w', encoding='utf-8') as f:
    for fichier_wav in fichiers_wav:
        nom_fichier_txt = fichier_wav.replace('.wav', '.txt')
        chemin_txt = os.path.join(dossier_txt, nom_fichier_txt)

        if not os.path.exists(chemin_txt):
            print(f"Fichier texte manquant pour: {fichier_wav}")
            continue

        with open(chemin_txt, 'r', encoding='utf-8') as txt_file:
            texte = txt_file.read().strip()

            chemin_audio = os.path.join(dossier_wav, fichier_wav).replace('\\', '/')

            f.write(f"{chemin_audio}|{texte}\n")


print(f"Le fichier de metadata a ete genere: {fichier_metadata}")