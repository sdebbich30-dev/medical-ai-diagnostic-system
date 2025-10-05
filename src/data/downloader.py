import pandas as pd
from pathlib import Path
import logging

class MedicalDataDownloader:
    def __init__(self, base_data_dir=\"data/raw\"):
        self.base_data_dir = Path(base_data_dir)
        self.setup_directories()
    
    def setup_directories(self):
        directories = ['chestxray14', 'mimic-cxr', 'chexpert', 'dental-xrays']
        for dir_name in directories:
            (self.base_data_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(\"✅ Structure de dossiers créée\")
    
    def download_chestxray14_sample(self):
        print(\"📥 Création des données sample...\")
        sample_data = {
            'Image_Index': ['00000001_000.png', '00000001_001.png'],
            'Finding_Labels': ['Cardiomegaly', 'No Finding'],
            'Patient_Age': [45, 45],
            'Patient_Gender': ['M', 'M']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(self.base_data_dir / 'chestxray14' / 'sample_labels.csv', index=False)
        print(\"✅ Fichier sample créé dans data/raw/chestxray14/\")

if __name__ == \"__main__\":
    downloader = MedicalDataDownloader()
    downloader.download_chestxray14_sample()
