import os
import pickle
from pathlib import Path


def save_model(model, filename="trained_model", path=None):
    if path == None:
        path = Path(os.getcwd())
    filepath = path / f"{filename}.sav"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model saved: {filepath}")