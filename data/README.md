# Data

Ce dossier ne contient pas les donnees brutes dans le depot Git.

Dataset Hugging Face :

`https://huggingface.co/datasets/saadtaleb/precipitations-era5-stations`

## Option 1 - Telechargement automatique

Depuis la racine du projet :

```powershell
python scripts/download_data.py
```

Les fichiers seront places dans `data/raw/`.

## Option 2 - Dossier local personnalise

Si vous avez deja telecharge les donnees ailleurs :

```powershell
python scripts/run_occurrence_models.py --data-dir C:\chemin\vers\donnees
```

## Fichiers attendus

- `station.data.81.10.txt`
- `ERA5.slp.81.10.txt`
- `ERA5.d2.81.10.txt`
- `ERA5.lon.81.10.txt`
- `ERA5.lat.81.10.txt`

Par defaut, les scripts utilisent `data/raw/` si ce dossier existe, sinon `../Data`.
