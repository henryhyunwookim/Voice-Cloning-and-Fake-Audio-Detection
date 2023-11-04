# Voice Cloning and Fake Audio Detection

### <b>Background</b>

We are a technology company working in the Cyber Security industry. We focus on building systems that help individuals and organizations to have safe and secure digital presence by providing cutting edge technologies to our customers. We create products and services that ensure our customers security using data driven technologies to understand whether audio and video media is authentic or fake.

Our goal in this project is to build algorithms that can synthesize spoken audio by converting a speaker’s voice to another speaker’s voice with the end goal to detect if any spoken audio is pristine or fake.

### <b>Data Description</b>

There are two datasets you can utilize in this project. Both datasets are publicly available sources.

<u>TIMIT Dataset</u>:

The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. TIMIT contains a total of 6300 sentences, 10 sentences spoken by each of 630 speakers from 8 major dialect regions of the United States.

Dataset Link: https://github.com/philipperemy/timit

<u>CommonVoice Dataset</u>:

Common Voice is part of Mozilla's initiative to help teach machines how real people speak. Common Voice is a corpus of speech data read by users on the Common Voice website (https://commonvoice.mozilla.org/), and based upon text from a number of public domain sources like user submitted blog posts, old books, movies, and other public speech corpora. Its primary purpose is to enable the training and testing of automatic speech recognition (ASR) systems.

Dataset Link: https://commonvoice.mozilla.org/en/datasets

### <b>Goal(s)</b>

Build a machine learning system to detect if a spoken audio is synthetically generated or not. In order to achieve this, first, build a voice cloning system given a speaker’s spoken audio that clones the source speaker’s voice to the target speaker’s voice. Next, build a machine learning system which detects if any spoken audio is a natural speech or synthetically generated by machine.

For the voice cloning system (VC), you can utilize the TIMIT dataset as it consists of aligned text-audio data with various speakers. For the fake audio detection system (FAD) you can utilize the CommonVoice dataset as it consists of thousands of naturally spoken audio which could be used as golden spoken audio by humans as positive examples and creating negative examples using the voice cloning system as automatic data/label generator. Since the CommonVoice English dataset is large, you can use a subset of it by sampling the dataset.

### <b> Success Metrics</b>

Use Word Error Rate (WER) for automatic evaluation of the voice cloning (VC) system for the speech generation part and also report speaker classification accuracy to assess the performance of the generated audio’s target speaker. For the fake audio detection (FAD) system evaluate the performance of the models using F-score via positive labels coming from the groundtruth dataset and negative labels generated by the VC.

### <b> Results</b>

<u>Voice Cloning (VC) System</u>

1. Algorithms

    We tested 17 different algorithms and identified two with superior performance as follows:
    - vits: An English voice conversion model from the <a href='https://pypi.org/project/TTS/'>TTS</a> library.
    - speech_generator: A speech generator from the <a href='https://pypi.org/project/Voice-Cloning/'>Voice_Cloning</a> package.

2. Model Performance
    - Word Error Rate (WER) for transcription<br>
    Evaluation of speech generation algorithms showed a WER of 0.12 for the 'vits' model and 0.34 for the 'speech_generator.' Despite the suboptimal WER for 'speech_generator,' it was still utilized due to its favorable impact on speaker classification accuracy. Several other evaluation measures were also adopted, and it's worth mentioning that the 'vits' model demonstrated a notable Character Error Rate (CER) of 0.04.
   
    - Speaker classification accuracy<br>
    A neural network model was constructed to assess speaker classification accuracy, achieving a score of 0.83. Considering the limited number of very short audio files available for each speaker, this accuracy is deemed reasonably good.

<u>Fake Audio Detection (FAD) System</u>

1. Model Performance
    - F-score for binary classification (authentic vs. synthesized speeches)<br>
        Another neural network model was developed to evaluate binary classification between authentic and synthesized speeches. The model achieved a perfect F-score on the test data, signifying its capability to distinguish fake audio with the highest precision and recall.

<u>Future Work</u>

The knowledge and models derived from this project exhibit significant potential for addressing diverse business challenges involving audio data. Future iterations will focus on enhancing system robustness and versatility by incorporating a broader range of source data for system development and model training.

### <b>Notebook and Installation</b>

For more details, you may refer to <a href='https://github.com/henryhyunwookim/Voice-Cloning-and-Fake-Audio-Detection/blob/main/VoiceCloningAndFakeAudioDetection.ipynb'>this notebook</a> directly.

To run VoiceCloningAndFakeAudioDetection.ipynb locally, please clone or fork this repo and install the required packages by running the following command:

pip install -r requirements.txt

##### <i>* Associated with Apziva</i>
