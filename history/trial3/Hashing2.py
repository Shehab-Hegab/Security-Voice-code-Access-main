import json
from fingerprint2 import audioSpectogram, loopFolder, stripName

def HashFiles(path, hashMode: str = "perception"):
    files = loopFolder(path, '.wav')

    hash_data = {}

    for i in range(len(files)):
        songName = stripName(files[i])
        print("NAME: ", songName)

        song = audioSpectogram(files[i], i, hashMode)
        spectHash, mffcHash, chromaHash, melHash, times, frequencies, spectrogram = song.getData()

        hash_data[songName] = {
            "spectHash": str(spectHash),
            "mffcHash": str(mffcHash),
            "chromaHash": str(chromaHash),
            "melHash": str(melHash)
        }

        del song

    with open("./hashing.json", "w") as hashFile:
        json.dump(hash_data, hashFile, indent=2)




HashFiles('./AudioFingerprints/hazem/Open middle door')