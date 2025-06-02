import sys

import cv2
from PyQt5.QtCore import QUrl, Qt, QTime
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QSlider, QLabel
)


class DualVideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

        self.setWindowTitle("Dual Video Player")
        self.setGeometry(100, 100, 1200, 600)

        # Players and widgets
        self.player1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget1 = QVideoWidget()
        videoWidget2 = QVideoWidget()

        # self.frame_time_ms will now be a float, representing exact milliseconds per frame
        self.frame_time_ms = 1000.0 / 30.0  # Default to 30 FPS for initialization

        # Controls
        self.openBtn = QPushButton('Open Videos')
        self.playBtn = QPushButton('Play/Pause')
        self.slider = QSlider(Qt.Horizontal)
        self.timeLabel = QLabel("00:00.000 / 00:00.000")
        self.frameLabel = QLabel("Frame: N/A")  # Label for frame number

        # NEW: Playback speed buttons
        self.speed100Btn = QPushButton('100%')
        self.speed50Btn = QPushButton('50%')
        self.speed25Btn = QPushButton('25%')
        self.speed10Btn = QPushButton('10%')

        self.slider.setRange(0, 0)
        self.openBtn.clicked.connect(self.openFiles)
        self.playBtn.clicked.connect(self.playPause)
        self.slider.sliderMoved.connect(self.setPosition)

        self.player1.positionChanged.connect(self.syncPosition)
        self.player1.durationChanged.connect(self.durationChanged)

        # NEW: Connect speed buttons
        self.speed100Btn.clicked.connect(lambda: self.setPlaybackSpeed(1.0))
        self.speed50Btn.clicked.connect(lambda: self.setPlaybackSpeed(0.5))
        self.speed25Btn.clicked.connect(lambda: self.setPlaybackSpeed(0.25))
        self.speed10Btn.clicked.connect(lambda: self.setPlaybackSpeed(0.1))

        # Layouts
        videoLayout = QHBoxLayout()
        videoLayout.addWidget(videoWidget1)
        videoLayout.addWidget(videoWidget2)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.playBtn)
        controlLayout.addWidget(self.openBtn)
        controlLayout.addWidget(self.slider)
        controlLayout.addWidget(self.timeLabel)
        controlLayout.addWidget(self.frameLabel)

        # NEW: Add speed buttons to control layout
        speedControlLayout = QHBoxLayout() # Create a sub-layout for speed buttons
        speedControlLayout.addWidget(QLabel("Speed:")) # Optional label
        speedControlLayout.addWidget(self.speed100Btn)
        speedControlLayout.addWidget(self.speed50Btn)
        speedControlLayout.addWidget(self.speed25Btn)
        speedControlLayout.addWidget(self.speed10Btn)

        layout = QVBoxLayout()
        layout.addLayout(videoLayout)
        layout.addLayout(controlLayout)
        layout.addLayout(speedControlLayout) # NEW: Add the speed control layout
        self.setLayout(layout)

        self.player1.setVideoOutput(videoWidget1)
        self.player2.setVideoOutput(videoWidget2)

    def openFiles(self):
        file1, _ = QFileDialog.getOpenFileName(self, "Open First Video")
        file2, _ = QFileDialog.getOpenFileName(self, "Open Second Video")
        if file1 and file2:
            # Extract FPS from one of the videos
            cap = cv2.VideoCapture(file1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self.frame_time_ms = 1000.0 / fps  # Ensure it's a float division
                print(f"Video FPS: {fps}, Calculated frame_time_ms: {self.frame_time_ms:.3f}")
            else:
                self.frame_time_ms = 1000.0 / 30.0  # Fallback to 30 FPS if not found
                print(f"Warning: Could not get FPS. Using default frame_time_ms: {self.frame_time_ms:.3f}")

            self.player1.setMedia(QMediaContent(QUrl.fromLocalFile(file1)))
            self.player2.setMedia(QMediaContent(QUrl.fromLocalFile(file2)))
            self.playPause()

    def playPause(self):
        if self.player1.state() == QMediaPlayer.PlayingState:
            self.player1.pause()
            self.player2.pause()
        else:
            self.player1.play()
            self.player2.play()

    def syncPosition(self, position):
        self.slider.setValue(position)
        if abs(self.player2.position() - position) > 50:
            self.player2.setPosition(position)
        self.updateTimeLabel(position)
        self.updateFrameLabel(position)

    def setPosition(self, position):
        # Ensure position is within valid range and cast to int for setPosition
        duration = self.player1.duration()
        if duration > 0:
            position = max(0, min(position, duration))
        else:
            position = max(0, position)

        self.player1.setPosition(int(position))
        self.player2.setPosition(int(position))

    def durationChanged(self, duration):
        self.slider.setRange(0, duration)

    def updateTimeLabel(self, position):
        curr_time = QTime(0, 0).addMSecs(position)
        total_time = QTime(0, 0).addMSecs(self.player1.duration())
        self.timeLabel.setText(
            f"{curr_time.toString('mm:ss.zzz')} / {total_time.toString('mm:ss.zzz')}"
        )

    def updateFrameLabel(self, position):
        if self.frame_time_ms > 0:
            frame_number = round(position / self.frame_time_ms)
            self.frameLabel.setText(f"Frame: {int(frame_number)}")
        else:
            self.frameLabel.setText("Frame: N/A")

    # NEW METHOD: Set playback speed for both players
    def setPlaybackSpeed(self, rate):
        self.player1.setPlaybackRate(rate)
        self.player2.setPlaybackRate(rate)
        print(f"Setting playback speed to: {rate * 100}%")


    def keyPressEvent(self, event):
        current_pos_ms = self.player1.position()

        if event.key() == Qt.Key_Right:
            new_time = current_pos_ms + self.frame_time_ms
            new_time = round(new_time)
            self.setPosition(new_time)
        elif event.key() == Qt.Key_Left:
            new_time = current_pos_ms - self.frame_time_ms
            new_time = round(new_time)
            if new_time < 0:
                new_time = 0
            self.setPosition(new_time)
        elif event.key() == Qt.Key_Space:
            self.playPause()
        super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = DualVideoPlayer()
    player.show()
    sys.exit(app.exec_())