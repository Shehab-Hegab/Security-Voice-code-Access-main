import json
from fingerprint import audioSpectogram, loopFolder, stripName

def HashFiles(path, hashMode: str = "difference"):
    files = loopFolder(path, '.wav')

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




HashFiles('./AudioFingerprints/hazem/Open middle door')