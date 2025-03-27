import csv

with open('../donnees/metadata.csv', 'r') as file:
    reader = csv.reader(file, delimiter='|')
    for row in reader:
        # Récupérer la partie avant le '|'
        filename = row[0].split('|')[0]  # Nom du fichier
        text = row[1].split('|')[0]      # Texte
        print(f"Le nom du fichier est: {filename}, son texte est: {text}")
