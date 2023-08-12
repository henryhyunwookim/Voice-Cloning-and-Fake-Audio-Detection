import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook
from numpy import asarray
from collections import defaultdict
from PIL import Image
from numpy import asarray
from tqdm import tqdm
import gdown
from pydub import AudioSegment
import random

from utils.nlp import clean_text


def load_data(file_name, folder_name=None):
    parent_path = Path(os.getcwd()).parent
    if folder_name:
        # path = parent_path / folder_name / file_name
        path = Path("/".join([os.getcwd(), folder_name, file_name]))
    else:
        # path = parent_path / file_name
        path = Path("/".join([os.getcwd(), file_name]))
    
    file_type = file_name.split(".")[1]
    if file_type=="csv":
        output = pd.read_csv(path)
    elif file_type== "xlsx":
        output = pd.read_excel(path)
    else:
        raise f"Failed to load data - invalid file type {file_type}"
    
    print(output.info(), "\n")
    print(output.describe(), "\n")

    return output


def check_file_downloaded(file_name, download_dir):
    file_path = download_dir / file_name
    if os.path.exists(file_path):
        print(f"File {file_name} already exists in {download_dir}.")
        return True
    else:
        print(f"File {file_name} doesn't exist in {download_dir}.")
        return False


def extract_zip_file(file_path, download_path, file_name):
    if os.path.exists(download_path + "\\" + file_name.split(".")[0]):
        print(f"{file_name} already extracted in {download_path}.")
    else:
        with ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(path=download_path)
        print(f"{file_name} extracted in {download_path}.")


def load_images(download_path, as_array=False):
    files_dict = defaultdict(dict)
    for f1 in os.listdir(download_path):
        if f1 == "images":
            print(f"Loading files in {download_path}\\{f1}")
            for f2 in os.listdir(download_path + f"\\{f1}"): # training or testing
                if files_dict.get(f2, "") == "":
                    files_dict[f2] = defaultdict(dict)

                print(f"Loading files in {download_path}\\{f1}\\{f2}")
                for f3 in os.listdir(download_path + f"\\{f1}\\{f2}"): # flip or notflip
                    if files_dict.get(f3, "") == "":
                        files_dict[f3] = defaultdict(dict)
                    
                    print(f"Loading files in {download_path}\\{f1}\\{f2}\\{f3}")
                    for f4 in tqdm(os.listdir(download_path + f"\\{f1}\\{f2}\\{f3}")):
                        if files_dict.get(f4, "") == "":
                            files_dict[f4] = defaultdict(dict)
                        
                        # Load each image file and convert it into a 3d (RGB) array.
                        jpg_file_path = download_path + f"\\{f1}\\{f2}\\{f3}\\{f4}"
                        image = Image.open(jpg_file_path)
                        if as_array:
                            image_array = asarray(image)
                            files_dict[f2][f3][f4] = image_array
                        else:
                            files_dict[f2][f3][f4] = image

    return files_dict


def get_image_shape(array_dict):
    image_shape = None
    for k, v in array_dict.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                while image_shape == None:
                    image_array = asarray(v3)
                    image_shape = image_array.shape
                    print(f"Image shape: {image_shape}")
    return image_shape


def load_dataframes(file_path, full_sheet_names):
    data_dfs = {}
    wb = load_workbook(file_path)
    sheet_names = wb.sheetnames
    for full_sheet_name, sheet_name in zip(full_sheet_names, sheet_names):
        sheet= wb[sheet_name]
        data = [[cell.value for cell in row] for row in sheet.iter_rows()]
        data_dfs[full_sheet_name] = pd.DataFrame(data[1:-1], columns=data[0])

    print(f'{len(data_dfs)} DataFrames loaded with the following sheet names:\n')
    for sheet_name in full_sheet_names:
        print(sheet_name)

    return data_dfs


def download_from_gdrive(file_id, file_name, download_dir, root_dir):
  if not check_file_downloaded(file_name, download_dir):
    os.chdir(download_dir)
    gdown.download(id=file_id, output=file_name)
    os.chdir(root_dir)


def get_variables_for_voice_cloning(source_audio_subpath, target_audio_subpath,
                                    root_dir, timit_dir, train_csv):
    # Get source details
    source_speaker_id = source_audio_subpath.split('/')[2]
    source_audio_file = source_audio_subpath.split('/')[3]
    source_file_id = source_audio_file.split('.')[0]
    source_audio_path = timit_dir / 'data' / source_audio_subpath
    print(f'Source path: {source_audio_path}')

    # Get source text
    source_text_subpath = train_csv[(train_csv['speaker_id']==source_speaker_id) &
            (train_csv['filename']==source_file_id+'.TXT')]['path_from_data_dir'].iloc[0]
    source_text_file = source_text_subpath.split('/')[3]
    source_text_path = timit_dir / 'data' / source_text_subpath
    with open(source_text_path) as txt:
        raw_source_text = txt.read().split()[2:]
    source_text = clean_text(raw_source_text, remove_whitespace=True, remove_punctuation=True, lower=True)

    # Get target details
    target_speaker_id = target_audio_subpath.split('/')[2]
    target_audio_file = target_audio_subpath.split('/')[3]
    target_file_id = target_audio_file.split('.')[0]
    target_audio_path = timit_dir / 'data' / target_audio_subpath
    print(f'Target path: {target_audio_path}')

    # Get output details
    output_folder = root_dir / 'output' / f'{source_speaker_id}-{source_file_id}_to_{target_speaker_id}-{target_file_id}'
    output_filename = f'{source_speaker_id}-{source_file_id}_to_{target_speaker_id}-{target_file_id}.wav'

    return source_speaker_id, source_audio_file, source_file_id, source_audio_path,\
        target_speaker_id, target_audio_file, target_file_id, target_audio_path,\
        source_text_file, source_text_path, source_text, output_folder, output_filename


def get_concat_audio(target_audio_path, target_concat_dir, target_speaker_id,
                    add_silent=500, n_duplicate_concat=10):
    target_parent_dir = os.path.split(target_audio_path)[0]
    target_output_path = target_concat_dir / f'{target_speaker_id}_concat.wav'
    
    if os.path.exists(target_output_path):
        print(f'{target_output_path} already exists.')
    
    else:
        print(f'Creating a concatenated audio file for target speaker {target_speaker_id}.')
        concat_audio = AudioSegment.empty()
        file_list = [file for file in os.listdir(target_parent_dir) if 'WAV.wav' in file]

        for i in range(n_duplicate_concat):
            random.shuffle(file_list)
            for file in file_list:
                wav_path = target_parent_dir / Path(file)
                audio = AudioSegment.from_wav(wav_path) + AudioSegment.silent(add_silent) # default: 1000ms=1 second
                concat_audio += audio

        concat_audio.export(target_output_path, 'wav')
    
    return target_output_path