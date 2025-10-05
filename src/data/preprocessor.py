import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class MedicalPreprocessor:
    def __init__(self):
        self.processed_data = None
    
    def load_sample_data(self):
        print(\"📊 Chargement des données sample...\")
        sample_data = {
            'image_id': [f'img_{i}' for i in range(100)],
            'patient_id': [f'pat_{i//2}' for i in range(100)],
            'age': np.random.normal(50, 15, 100).astype(int),
            'gender': np.random.choice(['M', 'F'], 100),
            'pathology_A': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'pathology_B': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        }
        self.df = pd.DataFrame(sample_data)
        return self.df
    
    def run_pipeline(self):
        print(\"🚀 Démarrage du prétraitement...\")
        self.load_sample_data()
        
        # Nettoyage
        self.df = self.df.drop_duplicates()
        
        # Split train/test
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        
        # Sauvegarde
        Path(\"data/processed\").mkdir(parents=True, exist_ok=True)
        train_df.to_csv(\"data/processed/train_data.csv\", index=False)
        test_df.to_csv(\"data/processed/test_data.csv\", index=False)
        
        print(f\"✅ Données sauvegardées: Train {len(train_df)}, Test {len(test_df)}\")
        return train_df, test_df

if __name__ == \"__main__\":
    preprocessor = MedicalPreprocessor()
    train_df, test_df = preprocessor.run_pipeline()
