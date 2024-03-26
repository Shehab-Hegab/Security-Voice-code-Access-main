from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtMultimedia import QSound
import pyaudio
import wave
import os
import fingerprint
import numpy as np
import pyqtgraph as pg
import librosa
import logging

logging.basicConfig(filename='application.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

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
        uic.loadUi("./MainWindow.ui", self)
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
        self.AccessLabel.setText("Processing!")
        background_color = QColor(0, 0, 0)
        self.AccessLabel.setStyleSheet(f"background-color: {background_color.name()};")
        self.AccessLabel.setAlignment(Qt.AlignCenter)

        QApplication.processEvents()
        
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        sound_file = wave.open("./assets/myrecording.wav", "wb")
        sound_file.setnchannels(2)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(self.frames))
        sound_file.close()


        self.plot_spectrogram('./assets/myrecording.wav')
        
        QApplication.processEvents()

        self.granted = 0

        self.compare()


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

        colormap = pg.colormap.getFromMatplotlib('hot')
        img.setLookupTable(colormap.getLookupTable())

        self.SpectWidget.getViewBox().autoRange()
        self.SpectWidget.show()


    def access(self):
        if self.granted == 1:
            self.AccessLabel.setText("Access Granted!")
            background_color = QColor(0, 150, 0)
            self.AccessLabel.setStyleSheet(f"background-color: {background_color.name()};")
            QSound.play("./assets/Access Granted.wav")
            logging.info('Vocal Passcode Matched: Access Granted')
        elif self.granted == 0:
            self.AccessLabel.setText("Access Denied!")
            background_color = QColor(255, 0, 0)
            self.AccessLabel.setStyleSheet(f"background-color: {background_color.name()};")
            QSound.play("./assets/Access Denied.wav")
            logging.info('Vocal Passcode did not Match: Access Denied')

        self.AccessLabel.setAlignment(Qt.AlignCenter)


    # def compare(self):
    #     recorded_audio = fingerprint.audioSpectogram('./assets/myrecording.wav')   
    #     fingerprint.compare(recorded_audio)
    #     print("OOF,done hashing")                 
    #     scores = fingerprint.sortedScores()
    #     for score in scores:
    #         if float(score[5]) > 80:
    #             print(float(score[5]))
    #             self.granted = 1
    #             break
    #     self.access()
        
    def compare(self):
        recorded_audio = fingerprint.audioSpectogram('./assets/myrecording.wav')   
        fingerprint.compare(recorded_audio)
        scores = fingerprint.sortedScores()

        # Clear existing contents and reset row count
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()

        column_names = ["Record", "spectScore", "mffcScore", "tonnetzScore", "chromaScore", "totalScore"]
        self.tableWidget.setColumnCount(len(column_names))
        self.tableWidget.setHorizontalHeaderLabels(column_names)

        for score in scores:
            if float(score[5]) > 85:
                print(float(score[5]))
                self.granted = 1
                break

            # Insert a new row
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)

            # Insert data into each cell
            for col_position, column_name in enumerate(column_names):
                value = score[col_position]
                item = QTableWidgetItem(str(value))
                self.tableWidget.setItem(row_position, col_position, item)

        self.tableWidget.resizeColumnsToContents()
        self.access()




# Main Code
# ----------
def main():
    app = QApplication([])
    window = SpeakerRecog()
    app.exec_()

if __name__ == "__main__":
    main()
