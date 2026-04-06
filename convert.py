import pandas as pd

# Charger le fichier Excel
df = pd.read_excel("data/patients_dakar.xlsx")

# Convertir en CSV
df.to_csv("data/patients_dakar.csv", index=False)

print("Conversion terminée ✅")