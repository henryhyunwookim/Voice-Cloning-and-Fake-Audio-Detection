import os
import pickle
from pathlib import Path
from shutil import copy2


def save_model(model, filename="trained_model", path=None):
    if path == None:
        path = Path(os.getcwd())
    filepath = path / f"{filename}.sav"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model saved: {filepath}")


def copy_original(output_folder,
                source_audio_file, source_audio_path,
                target_audio_file, target_audio_path,
                source_text_file, source_text_path):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        print(f'{output_folder} created.')
    else:
        print(f'{output_folder} already exists.')

    # Copy with file permission; dest can be a folder
    if not os.path.exists(output_folder / source_audio_file):
        copy2(src=source_audio_path, dst=output_folder / source_audio_file)
        print(f'{source_audio_file} copied in {output_folder}.')
    else:
        print(f'{source_audio_file} already exists in {output_folder}.')

    if not os.path.exists(output_folder / target_audio_file):
        copy2(src=target_audio_path, dst=output_folder / target_audio_file)
        print(f'{target_audio_file} copied in {output_folder}.')
    else:
        print(f'{target_audio_file} already exists in {output_folder}.')
        
    if not os.path.exists(output_folder / source_text_file):
        copy2(src=source_text_path, dst=output_folder / source_text_file)
        print(f'{source_text_file} copied in {output_folder}.')
    else:
        print(f'{source_text_file} already exists in {output_folder}.')