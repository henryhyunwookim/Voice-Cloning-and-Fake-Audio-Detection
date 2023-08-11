from TTS.api import TTS
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from voice_cloning.generation import speech_generator, save_sound
from datetime import datetime
from pydub import AudioSegment
import os


def process_audio(audio_path, timit_dir, sr=None, n_mfcc=None):
    X_list = []
    y_list = []
    for audio_subpath in tqdm(audio_path):
        audio_path = timit_dir / 'data' / audio_subpath

        y = audio_subpath.split('/')[2]
        y_list.append(y)

        audio_array, sample_rate = librosa.load(audio_path, sr=None)
        # Set sr to None to get original sampling rate. Otherwise the default is 22050.
        # print(f'Shape of audio array: {audio_array.shape}')

        mfccs_features = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=n_mfcc) # n_mfcc=40 => larger values result in more features
        # print(f'Shape of mfccs features: {mfccs_features.shape}')
        
        X = np.mean(mfccs_features.T, axis=0) # Normalize features into the same scale
        # print(f'Shape of scaled mfccs features: {X.shape}')
        X_list.append(X)

        # scaled_X = StandardScaler().fit_transform(X.reshape(-1, 1)).reshape(1, -1)[0]
        # X_list.append(scaled_X)

        # S = np.abs(librosa.stft(audio_array))
        # transformed_s = librosa.amplitude_to_db(S, ref=np.max)
        # normalized_s = np.mean(transformed_s.T, axis=0)
        # X_list.append(normalized_s)

    df = pd.DataFrame(X_list)
    df['label'] = y_list

    X = df.drop(['label'], axis=1)
    y = df['label']

    return X, y, df


def clone_voice(target_audio_path, source_audio_path, source_text, output_dir, output_filename,
                tools=['voice_cloning', 'multilingual_tts', 'en_tts'],
                noise_reduction=False, adjust_decibel=0, progress_bar=False, gpu=False):
    """
    noise_reduction: only for voice_cloning model
    progress_bar, gpu: only for tts models
    """
    
    start = datetime.now()

    generated_wav = None
    if 'voice_cloning' in tools:
        # Source: https://pypi.org/project/Voice-Cloning/
        # 1. Clone using an external reference voice
        vc_output_path = output_dir / f'vc_{output_filename}'
        if os.path.exists(vc_output_path):
            print(f'{vc_output_path} already exists.')
        else:
            print(f'Cloning from source audio to target audio by speech_generator.')
            # This will generate a synthesized sound based on the reference sound/voice and text.
            generated_wav = speech_generator(
                sound_path = target_audio_path, # Reference sound file
                speech_text = source_text # Reference speech text
                )

            # Save the sound by speech generator as a file
            save_sound(generated_wav,
                filename=str(vc_output_path).replace('.wav', ''),
                noise_reduction=noise_reduction)

            # Adjust volume of the generated file
            if adjust_decibel != 0:
                (AudioSegment.from_wav(vc_output_path) + adjust_decibel).export(vc_output_path, 'wav')
            print()
            
    if 'multilingual_tts' in tools:
        # Multilingual voice conversion model - voice to voice
        multi_output_path = output_dir / f'multi_{output_filename}'
        if os.path.exists(multi_output_path):
            print(f'{multi_output_path} already exists.')
        else:
            print('Cloning from source audio to target audio by multilingual model.')
            multi_tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24",
                            progress_bar=progress_bar,
                            gpu=gpu)
            multi_tts.voice_conversion_to_file(
                source_wav=source_audio_path,
                target_wav=target_audio_path,
                file_path=multi_output_path
            )
            
            # Adjust volume of the generated file
            if adjust_decibel != 0:
                (AudioSegment.from_wav(multi_output_path) + adjust_decibel).export(multi_output_path, 'wav')

    if 'en_tts' in tools:
        # English voice conversion models - text to speech/voice
        en_models = [model for model in TTS.list_models() if '/en/' in model]
        for en_model in en_models:
            try:
                en_model_name = en_model.split('/')[-1]
                en_output_path = output_dir / f'{en_model_name}_{output_filename}'
                if os.path.exists(en_output_path):
                    print(f'{en_output_path} already exists.')
                else:
                    print(f'Cloning from source audio to target audio by {en_model_name} model.')
                    en_tts = TTS(en_model)
                    en_tts.tts_with_vc_to_file(
                        source_text,
                        speaker_wav=target_audio_path,
                        file_path=en_output_path
                    )
                    
                    # Adjust volume of the generated file
                    if adjust_decibel != 0:
                        (AudioSegment.from_wav(en_output_path) + adjust_decibel).export(en_output_path, 'wav')
            except Exception as e:
                print(e)
                print(f'Failed to load {en_model_name}.')

    duration = datetime.now() - start
    print(f'\nVoice cloning completed in {duration}.')

    recognized_tools = ['voice_cloning', 'multilingual_tts', 'en_tts']
    unrecognized_tools = [tool for tool in tools if tool not in recognized_tools]
    if len(unrecognized_tools) > 0:
        print(f'Vocie cloning/conversion failed! - Unrecognized tools {unrecognized_tools}')
        print(f'Recognized tools are {recognized_tools}.')