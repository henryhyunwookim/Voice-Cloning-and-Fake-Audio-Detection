import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


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