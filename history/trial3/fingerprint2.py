import matplotlib.pyplot as plt
from tempfile import mktemp
import librosa
import imagehash
from imagehash import hex_to_hash
import json
from PIL import Image
from scipy import signal
import librosa.display
import glob
import tempfile
import functools

class audioSpectogram():

    def __init__(self, path: str, id: int = 0, hashingMode: str = "perception"):
        self.path = path
        self.id = id
        self.hashMode = hashingMode
        self.samples = None
        self.sample_rate = None
        self.hash = None
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        self.spectHash = None
        self.mffcHash = None
        self.melHash = None
        self.chromaHash = None

        self.readAudio()
        self.setData()

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
        self.melHash = None
        self.chromaHash = None

    def readAudio(self):
        self.samples, self.sample_rate = librosa.load(self.path, duration=3.0, mono=True)

    def setData(self):
        spect, colorMesh = self.spectro()
        ft1, ft2, ft3 = self.features(colorMesh)
        self.spectHash = self.hashFile(spect, self.hashMode)
        self.mffcHash = self.hashFile(ft1, self.hashMode)
        self.chromaHash = self.hashFile(ft2, self.hashMode)
        self.melHash = self.hashFile(ft3, self.hashMode)

    def getData(self):
        return (self.spectHash, self.mffcHash, self.chromaHash, self.melHash, self.times, self.frequencies,
                self.spectrogram)

    def spectro(self):
        sampleFreqs, sampleTime, colorMesh = signal.spectrogram(self.samples, fs=self.sample_rate)
        X = librosa.stft(self.samples)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=self.sample_rate)
        plt.tight_layout(pad=0)

        # Use NamedTemporaryFile to create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            spectFig = tmpfile.name
            plt.savefig(spectFig, bbox_inches='tight', transparent=True, pad_inches=0)

        return spectFig, colorMesh

    def hashFile(self, figure: str, mode: str = "perception"):
        if mode == "average":
            hash = imagehash.average_hash(Image.open(figure))
        elif mode == "perception":
            hash = imagehash.phash(Image.open(figure))
        elif mode == "difference":
            hash = imagehash.dhash(Image.open(figure))
        elif mode == "wavelet":
            hash = imagehash.whash(Image.open(figure))
        return hash

    def features(self, c):
        mfccs = librosa.feature.mfcc(y=self.samples, sr=self.sample_rate, S=c, n_mels=128)
        librosa.display.specshow(mfccs)
        plt.tight_layout(pad=0)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            mffcFig = tmpfile.name
            plt.savefig(mffcFig, bbox_inches='tight', transparent=True, pad_inches=0)

        chroma = librosa.feature.chroma_stft(y=self.samples, sr=self.sample_rate, S=c)
        librosa.display.specshow(chroma)
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            chromaFig = tmpfile.name
            plt.savefig(chromaFig, bbox_inches='tight', transparent=True, pad_inches=0)

        melSpect = librosa.feature.melspectrogram(y=self.samples, sr=self.sample_rate, S=c)
        librosa.display.specshow(melSpect)
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            melFig = tmpfile.name
            plt.savefig(melFig, bbox_inches='tight', transparent=True, pad_inches=0)

        return mffcFig, chromaFig, melFig


def loopFolder(path:str,extension:str):
    Files = glob.glob(path + "/*" + extension) # list of all (mp3/or else) files in the folder
    return Files

def stripName(path:str):
    names  = path.split('/')
    songName = names[-1]
    songNameLst = songName.split('.')
    songName = songNameLst[0]
    return songName

def readFromTxt():
    hashFile = open("hashing-output/hashing.txt",'r')
    songsData =  hashFile.read().split('\n')[1:-2]
    return songsData


def extractFromline(dataLine:str):
    dataLine = list(map(functools.partial(str.split,sep = ','),dataLine))
    for i in range(len(dataLine)):

        dataLine[i] = list(map(functools.partial(str.split,sep = ':'),dataLine[i]))
        for j in range(len(dataLine[i])):
            
            dataLine[i][j] = dataLine[i][j][-1]
    return dataLine


def sortedScores():
    scoreFile = open("./similarityScore.txt",'r')
    songsData =  scoreFile.read().split('\n')[:-1]#read lines into a list
    songsData = list(map(functools.partial(str.split,sep = ','),songsData))
    sortedSongsScores = sorted(songsData, key=lambda x: x[5],reverse=True)
    return sortedSongsScores



def compare(inputMixedSong : 'audioSpectogram'):
    with open("./hashing.json", "r") as hashFile:
        hash_data = json.load(hashFile)

    hashFile = open("./hashing.txt","w") 

    similarityScore = open("./similarityScore.txt","w")

    MixedSpectHash,MixedMffcHash,MixedChromaHash,MixedMelHash,MixedTimes,MixedFrequencies,MixedSpectrogram = inputMixedSong.getData()
    

    hashFile.write("{ \n") 
    for songName, hash_values in hash_data.items():

        print("NAME: ",songName)    
       
        spectScore = 100 - (hex_to_hash(hash_values["spectHash"]) - MixedSpectHash)
        mffcScore = 100 - (hex_to_hash(hash_values["mffcHash"]) - MixedMffcHash)
        chromaScore = 100 - (hex_to_hash(hash_values["chromaHash"]) - MixedChromaHash)
        melScore = 100 - (hex_to_hash(hash_values["melHash"]) - MixedMelHash)
        # totalScore = (spectScore + mffcScore + melScore + chromaScore)/4

        # mode 1
        # totalScore = (2 * mffcScore + melScore + spectScore + chromaScore) / 5
        
        # mode 2
        totalScore = (2 * mffcScore + 2 * melScore + spectScore + chromaScore) / 6
       
        similarityScore.write("{},{},{},{},{},{}\n".format(songName,spectScore,mffcScore,chromaScore, melScore,totalScore))
        
    hashFile.write("} \n")     
    hashFile.close()
    similarityScore.close()

plt.close('all')
