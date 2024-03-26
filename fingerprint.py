import librosa
import json
import librosa.display
from scipy.spatial.distance import cosine
import numpy as np
import glob
import pandas as pd


class audioSpectogram():

    def __init__(self,path, id = 0):
        self.path = path
        self.id = id
        self.samples = None
        self.sample_rate = None
        self.hash = None
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        self.spectHash = None
        self.mffcHash = None
        self.tonnetzHash = None
        self.chromaHash = None

        self.readAudio()    
        self.getData()

    def __del__(self):
        self.path = None
        self.samples = None
        self.sample_rate = None
        self.hash = None
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        self.spectHash = None
        self.mffcHash = None
        self.tonnetzHash = None
        self.chromaHash = None

    def readAudio(self):
                
        self.samples , self.sample_rate = librosa.load(self.path)
    
    def getData(self):
        if self.sample_rate:
            stft = librosa.stft(y = self.samples)

            chroma = librosa.feature.chroma_stft(S = stft)

            chroma_mean = np.mean(chroma, axis = 1)

            zero_crossings = librosa.feature.zero_crossing_rate(self.samples)

            mfcc = librosa.feature.mfcc(sr = self.sample_rate, S = stft)
            std_mfcc = np.std(mfcc, axis = 1)

            energy = np.sum(np.abs(stft), axis = 0)


            fingerprint = np.concatenate([np.angle(chroma_mean), np.abs(chroma_mean), std_mfcc, zero_crossings.flatten(), energy])
        
        return fingerprint
    
def loopFolder(path:str,extension:str):
    Files = glob.glob(path + "/*" + extension)
    return Files

def cosine_similarity(a, b):
    return 1 - cosine(a, b)


def fingerprint_from_string(fp_str):
    return np.fromstring(fp_str[1:-1], sep=' ')

def compare(path):
    df = pd.read_csv('hashing.csv')

    input_fingerprint = audioSpectogram(path)
    input_fingerprint = input_fingerprint.getData()
    
    similarity_scores = [cosine_similarity(input_fingerprint, fingerprint_from_string(fp_str)) for fp_str in df['Fingerprint']]

    result_df = df[['Person', 'Phrase']].copy()
    result_df['Similarity Score'] = similarity_scores
    result_df.to_csv('result.csv', index=False)
    return similarity_scores, result_df

# compare('./AudioFingerprints/Arsany/Open middle door/record5.wav')


    