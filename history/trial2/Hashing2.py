import hashlib
from operator import itemgetter
from typing import List, Tuple
import librosa

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure, iterate_structure

CONNECTIVITY_MASK = 2
DEFAULT_AMP_MIN = 10
DEFAULT_FS = 44100
DEFAULT_FAN_VALUE = 5
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_WINDOW_SIZE = 4096
FINGERPRINT_REDUCTION = 20
MAX_HASH_TIME_DELTA = 200
MIN_HASH_TIME_DELTA = 0
PEAK_NEIGHBORHOOD_SIZE = 10
PEAK_SORT = True

def fingerprint(channel_samples: List[int],
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    plt.figure(figsize=(12, 8))

    # Plot the waveform
    plt.subplot(2, 1, 1)
    plt.title('Waveform')
    plt.plot(np.arange(len(channel_samples)) / Fs, channel_samples)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot the spectrogram
    plt.subplot(2, 1, 2)
    arr2D, freqs, times = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))

    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))

    plt.imshow(arr2D, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.colorbar(label='Intensity (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

    local_maxima = get_2D_peaks(arr2D, channel_samples, plot=False, amp_min=amp_min)


    return generate_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.array, channel_samples, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN) \
        -> List[Tuple[List[int], List[int]]]:
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    detected_peaks = local_max != eroded_background

    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    amps = amps.flatten()

    # Use abs(amps) for filtering
    filter_idxs = np.where(abs(amps) > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    # print("Detected Peaks:")
    # print(list(zip(freqs, times)))
    # print("Filtered Peaks:")
    # print(list(zip(freqs_filter, times_filter)))

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))


def generate_hashes(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE) -> List[Tuple[str, int]]:
    idx_freq = 0
    idx_time = 1

    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value + 1):
            if (i + j) < len(peaks):
                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8')).hexdigest()[:FINGERPRINT_REDUCTION]
                    hashes.append((h, t1))
                    print(f"Hash: {h}, Time: {t1}")

    return hashes


def read_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    audio_data, sr = librosa.load(file_path, sr=DEFAULT_FS)
    # print(f"Audio data shape: {audio_data.shape}, Sampling rate: {sr}")
    return audio_data, sr

def export_hashes_to_txt(hashes: List[Tuple[str, int]], output_file: str):
    with open(output_file, 'w') as f:
        for hash_value, offset in hashes:
            f.write(f"{hash_value} {offset}\n")

audio_data, sr = read_audio_file("./AudioFingerprints/Arsany/Open middle door/record2.wav")
hashes = fingerprint(audio_data, Fs=sr)
print(f"Number of hashes generated: {len(hashes)}")
export_hashes_to_txt(hashes, "./output_hashes.txt")