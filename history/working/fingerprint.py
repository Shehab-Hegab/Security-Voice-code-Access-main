import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
from tempfile import mktemp
import numpy as np
import librosa
import imagehash
from PIL import Image
import librosa.display
import glob
import gc
import functools
import soundfile as sf
from operator import itemgetter

class audioSpectogram():

    def __init__(self,path:str,id:int = 0,hashingMode:str="perception"):
        self.path = path
        self.id = id
        self.hashMode = hashingMode
        self.samples = None
        self.sample_rate = None
        self.lenght = None
        self.hash = None
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        self.spectHash = None
        self.mffcHash = None
        self.tonnetzHash = None
        self.chromaHash = None

        print("objct {} created :)".format(self.id))
        self.readAudio()    
        self.setData()

    def __del__(self):
        self.path = None
        self.samples = None
        self.sample_rate = None
        self.lenght = None
        self.hash = None
        self.spectrogram = None
        self.frequencies = None
        self.times = None
        self.spectHash = None
        self.mffcHash = None
        self.tonnetzHash = None
        self.chromaHash = None
        print("objct {} deleted :X".format(self.id))

    def readAudio(self):
                
        self.samples , self.sample_rate = librosa.load(self.path, duration=3.0,mono = True)
                

    def setData(self):
        spect = self.spectro()
        ft1,ft2,ft3 = self.features()
        self.spectHash = self.hashFile(spect,self.hashMode)
        self.mffcHash  = self.hashFile(ft1,self.hashMode)
        self.tonnetzHash = self.hashFile(ft2,self.hashMode)
        self.chromaHash = self.hashFile(ft3,self.hashMode)
        
    def getData(self):
        return (self.spectHash,self.mffcHash,self.tonnetzHash,self.chromaHash,self.times,self.frequencies,self.spectrogram)    

    def spectro(self):
        X = librosa.stft(self.samples)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=self.sample_rate)
        plt.tight_layout(pad = 0)
        spectFig = mktemp('.png')
        plt.savefig(spectFig,bbox_inches='tight', transparent=True, pad_inches=0)
        return spectFig
    
    
    def hashFile(self,figure:str,mode:str = "perception"): # Hashing function , returns a string
        if mode == "average":
            hash = imagehash.average_hash(Image.open(figure))
        elif mode == "perception":    
            hash = imagehash.phash(Image.open(figure))
        elif mode == "difference":
            hash = imagehash.dhash(Image.open(figure))
        elif mode == "wavelet":
            hash = imagehash.whash(Image.open(figure))    
        # print(hash)
        return hash

    def features(self):
        mfccs = librosa.feature.mfcc(y=self.samples, sr=self.sample_rate)
        librosa.display.specshow(mfccs)
        plt.tight_layout(pad = 0)
        mffcFig = mktemp('.png')
        plt.savefig(mffcFig,bbox_inches='tight', transparent=True, pad_inches=0)

        tonnetz = librosa.feature.tonnetz(y=self.samples, sr=self.sample_rate)
        librosa.display.specshow(tonnetz)
        plt.tight_layout(pad = 0)
        tonnetzFig = mktemp('.png')
        plt.savefig(tonnetzFig,bbox_inches='tight', transparent=True, pad_inches=0)
        
        chroma = librosa.feature.chroma_cqt(y=self.samples, sr=self.sample_rate)
        librosa.display.specshow(chroma)
        plt.tight_layout()
        chromaFig = mktemp('.png')
        plt.savefig(chromaFig,bbox_inches='tight', transparent=True, pad_inches=0)
        return (mffcFig,tonnetzFig,chromaFig)


def loopFolder(path:str,extension:str):
    Files = glob.glob(path + "/*" + extension) # list of all (mp3/or else) files in the folder
    return Files

def stripName(path:str):
    names  = path.split('/')
    songName = names[-1]
    songNameLst = songName.split('.')
    songName = songNameLst[0]
    return songName

def  generateTxt(inputMixedSong : 'audioSpectogram',hashMode:str="perception"):
    files =  loopFolder('./AudioFingerprints/ahmad/Unlock the gate','.wav')

    hashFile = open("./hashing.txt","w") 

    similarityScore = open("./similarityScore.txt","w")

    MixedSpectHash,MixedMffcHash,MixedTonnetzHash,MixedChromaHash,MixedTimes,MixedFrequencies,MixedSpectrogram = inputMixedSong.getData()

    hashFile.write("{ \n") 
    for i in range(len(files)):
        songName =  stripName(files[i])

        print("NAME: ",songName)

        song = audioSpectogram(files[i],i,hashMode)
        spectHash,mffcHash,tonnetzHash,chromaHash,times,frequencies,spectrogram = song.getData()        
        
        hashFile.write("Name:{},spectHash:{},mffcHash:{},tonnetzHash:{},chromaHash:{}\n".format(songName,spectHash,mffcHash,tonnetzHash,chromaHash))
       
       
        spectScore = 100 - (spectHash - MixedSpectHash)
        mffcScore = 100 - (mffcHash - MixedMffcHash)
        tonnetzScore = 100 - (tonnetzHash - MixedTonnetzHash)
        chromaScore = 100 - (chromaHash - MixedChromaHash)
        totalScore = (spectScore + mffcScore + tonnetzScore + chromaScore)/4
       
        similarityScore.write("{},{},{},{},{},{}\n".format(songName,spectScore,mffcScore,tonnetzScore,chromaScore,totalScore))
        
        
        del song
        # gc.collect()
    hashFile.write("} \n")     
    hashFile.close()
    similarityScore.close()


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
