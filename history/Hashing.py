import matplotlib.pyplot as plt
from tempfile import mktemp
import numpy as np
import librosa
import imagehash
from PIL import Image
import librosa.display
import glob
import json

class audioSpectogram():

    def __init__(self,path:str,id:int = 0,hashingMode:str="difference"):
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
    
    
    def hashFile(self,figure:str,mode:str = "difference"): # Hashing function , returns a string
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

def  generateTxt(hashMode:str="difference"):
    files =  loopFolder('./AudioFingerprints/ahmad/Unlock the gate','.wav')

    hashFile = open("./hashing.txt","w") 

    hashFile.write("{ \n") 
    for i in range(len(files)):
        songName =  stripName(files[i])

        print("NAME: ",songName)

        song = audioSpectogram(files[i],i,hashMode)
        spectHash,mffcHash,tonnetzHash,chromaHash,times,frequencies,spectrogram = song.getData()        
        
        hashFile.write("Name:{},spectHash:{},mffcHash:{},tonnetzHash:{},chromaHash:{}\n".format(songName,spectHash,mffcHash,tonnetzHash,chromaHash))
        
        
        del song

    hashFile.write("} \n")     
    hashFile.close()
    


def generateTxt(hashMode: str = "difference"):
    files = loopFolder('./AudioFingerprints/ahmad/Open middle door', '.wav')

    hash_data = {}

    for i in range(len(files)):
        songName = stripName(files[i])
        print("NAME: ", songName)

        song = audioSpectogram(files[i], i, hashMode)
        spectHash, mffcHash, tonnetzHash, chromaHash, times, frequencies, spectrogram = song.getData()

        hash_data[songName] = {
            "spectHash": str(spectHash),
            "mffcHash": str(mffcHash),
            "tonnetzHash": str(tonnetzHash),
            "chromaHash": str(chromaHash)
        }

        del song

    with open("./hashing.json", "w") as hashFile:
        json.dump(hash_data, hashFile, indent=2)



def retrieveHashes():
    with open("./hashing.json", "r") as hashFile:
        hash_data = json.load(hashFile)

    for songName, hash_values in hash_data.items():
        print("Name:", songName)
        print("spectHash:", hash_values["spectHash"])
        print("mffcHash:", hash_values["mffcHash"])
        print("tonnetzHash:", hash_values["tonnetzHash"])
        print("chromaHash:", hash_values["chromaHash"])
        print()


generateTxt()
