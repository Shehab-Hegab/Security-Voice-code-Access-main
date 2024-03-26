import json
import os
import pandas as pd
from fingerprint import audioSpectogram, loopFolder

def HashFiles(root_path):
    person_folders = loopFolder(root_path, '')

    hash_data = {'Person': [], 'Phrase': [], 'Fingerprint': []}

    for person_folder in person_folders:
        person_name = os.path.basename(person_folder)
        print("Processing person:", person_name)

        phrase_folders = loopFolder(person_folder, '')

        for phrase_folder in phrase_folders:
            phrase_name = os.path.basename(phrase_folder)
            print("  - Phrase:", phrase_name)

            files = loopFolder(phrase_folder, '.wav')
            for i in range(len(files)):
                record = os.path.basename(files[i])
                print("    - File:", record)

                song = audioSpectogram(files[i], i)
                fingerprint = song.getData()

                hash_data['Person'].append(person_name)
                hash_data['Phrase'].append(phrase_name)
                hash_data['Fingerprint'].append(fingerprint)

                del song

    df = pd.DataFrame(hash_data)
    df.to_csv('./hashing.csv', index=False)


HashFiles('./AudioFingerprints/')
