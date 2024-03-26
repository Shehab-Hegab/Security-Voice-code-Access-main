from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtMultimedia import QSound
import pyaudio
import wave
import os
# import fingerprint
import numpy as np
import pyqtgraph as pg
import librosa

class RecordThread(QThread):
    finished = pyqtSignal()

    def __init__(self, frames, stream, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.stream = stream

    def run(self):
        while not self.isInterruptionRequested():
            data = self.stream.read(1024)
            self.frames.append(data)
        self.finished.emit()


class SpeakerRecog(QMainWindow):
    def __init__(self):
        super(SpeakerRecog, self).__init__()
        uic.loadUi("./Recorder.ui", self)
        self.setWindowTitle("SpeakerRecog")
        self.show()

        self.RecordButton.clicked.connect(self.record_handler)
        self.recording = False
        self.granted = 0
        self.frames = []

    def record_handler(self):
        if not self.recording:
            self.recording = True
            self.frames = []

            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024)

            self.record_thread = RecordThread(self.frames, self.stream)

            self.record_thread.finished.connect(self.stop_recording)
            self.record_thread.start()

            self.AccessLabel.setText("Recording!")
            background_color = QColor(0, 0, 0)
            self.AccessLabel.setStyleSheet(f"background-color: {background_color.name()};")
            self.AccessLabel.setAlignment(Qt.AlignCenter)

            QApplication.processEvents()
            
        else:
            self.recording = False
            self.record_thread.requestInterruption()

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        self.name, self.word_code, self.trial = self.NameEdit.text(), self.PasscodeEdit.text(), self.TrialEdit.text()
        sound_file = wave.open(f"./AudioFingerprints/{str(self.name)}/{str(self.word_code)}/record{str(self.trial)}.wav", "wb")
        sound_file.setnchannels(2)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(self.frames))
        sound_file.close()

        self.plot_spectrogram(f"./AudioFingerprints/{str(self.name)}/{str(self.word_code)}/record{str(self.trial)}.wav")
        self.access()


    def plot_spectrogram(self, path):
        self.SpectWidget.clear()
        y, sr = librosa.load(path, sr=44100)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr
        )

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mel_spectrogram_db = np.transpose(mel_spectrogram_db)

        img = pg.ImageItem()
        img.setImage(mel_spectrogram_db)

        self.SpectWidget.addItem(img)
        # self.SpectWidget.setLabel('left', 'Frequency', units='KHz')
        # self.SpectWidget.setLabel('bottom', 'Time', units='s')

        colormap = pg.colormap.getFromMatplotlib('hot')
        img.setLookupTable(colormap.getLookupTable())

        self.SpectWidget.getViewBox().autoRange()
        self.SpectWidget.show()


    def access(self):
        self.AccessLabel.setText("Recorded!")
        background_color = QColor(0, 150, 0)
        self.AccessLabel.setStyleSheet(f"background-color: {background_color.name()};")
        # QSound.play("./assets/Access Granted.wav")
        self.AccessLabel.setAlignment(Qt.AlignCenter)





    def show_message_box(self, title):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(f"This is a {title}.")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)

        result = msg_box.exec_()
                


# Main Code
# ----------
def main():
    app = QApplication([])
    window = SpeakerRecog()
    app.exec_()

if __name__ == "__main__":
    main()
