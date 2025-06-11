import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QSpinBox, QHBoxLayout, QLineEdit, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class VideoToImagesApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extrator de Frames")
        self.setGeometry(100, 100, 600, 600)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.frame_index = 0
        self.is_playing = False

        # Seletor de vídeo
        self.video_label = QLabel("Vídeo selecionado:")
        self.video_path = QLineEdit()
        self.video_path.setReadOnly(True)
        self.select_video_btn = QPushButton("Selecionar Vídeo")
        self.select_video_btn.clicked.connect(self.select_video)

        # Metadados
        self.video_info_label = QLabel("Informações do vídeo:")
        self.video_info = QLabel("Nenhum vídeo selecionado.")

        # Pasta de saída
        self.output_label = QLabel("Pasta de saída:")
        self.output_path = QLineEdit("thermal_images")
        self.select_output_btn = QPushButton("Selecionar Pasta")
        self.select_output_btn.clicked.connect(self.select_output_folder)

        # FPS
        self.fps_label = QLabel("Extrair quantos frames:")
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 1000)
        self.fps_input.setValue(1)

        self.unit_label = QLabel("por:")
        self.unit_selector = QComboBox()
        self.unit_selector.addItems(["Segundo", "Minuto", "Hora"])

        # Botão de extração
        self.extract_btn = QPushButton("Extrair Frames")
        self.extract_btn.clicked.connect(self.extract_frames)

        # Status
        self.status = QLabel("Status: Aguardando ação...")

        # Visualização de vídeo
        self.video_display = QLabel()
        self.video_display.setFixedSize(480, 270)
        self.video_display.setStyleSheet("background-color: black")

        # Controles de vídeo
        self.play_btn = QPushButton("▶️ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")

        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.stop_btn.clicked.connect(self.stop_video)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        row1 = QHBoxLayout()
        row1.addWidget(self.video_path)
        row1.addWidget(self.select_video_btn)
        layout.addLayout(row1)

        layout.addWidget(self.video_info_label)
        layout.addWidget(self.video_info)

        layout.addWidget(self.output_label)
        row2 = QHBoxLayout()
        row2.addWidget(self.output_path)
        row2.addWidget(self.select_output_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(self.fps_label)
        row3.addWidget(self.fps_input)
        row3.addWidget(self.unit_label)
        row3.addWidget(self.unit_selector)
        layout.addLayout(row3)

        layout.addWidget(self.extract_btn)
        layout.addWidget(self.status)

        layout.addWidget(QLabel("Pré-visualização do vídeo:"))
        layout.addWidget(self.video_display)

        video_controls = QHBoxLayout()
        video_controls.addWidget(self.play_btn)
        video_controls.addWidget(self.pause_btn)
        video_controls.addWidget(self.stop_btn)
        layout.addLayout(video_controls)

        self.setLayout(layout)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path.setText(file_path)

            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self.video_info.setText(
                    f"Frames: {total_frames} | FPS: {fps:.2f} | "
                    f"Duração: {duration:.2f}s | Resolução: {width}x{height}"
                )
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.update_frame()
            else:
                self.video_info.setText("Erro ao carregar vídeo.")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_index += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(self.video_display.width(), self.video_display.height())
                self.video_display.setPixmap(pixmap)
            else:
                self.timer.stop()

    def play_video(self):
        if self.cap and not self.is_playing:
            self.is_playing = True
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / fps))

    def pause_video(self):
        self.is_playing = False
        self.timer.stop()

    def stop_video(self):
        if self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            self.is_playing = False
            self.update_frame()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída")
        if folder:
            self.output_path.setText(folder)

    def extract_frames(self):
        video_path = self.video_path.text()
        output_folder = self.output_path.text()

        if not os.path.exists(video_path):
            self.status.setText("Erro: Caminho do vídeo inválido.")
            return

        os.makedirs(output_folder, exist_ok=True)
        output_end = output_folder + "_end"
        os.makedirs(output_end, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.status.setText("Erro ao abrir o vídeo.")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        user_fps = self.fps_input.value()
        unit = self.unit_selector.currentText()

        seconds_factor = {"Segundo": 1, "Minuto": 60, "Hora": 3600}
        interval = int(fps_video / (user_fps / seconds_factor[unit])) if user_fps > 0 else 1

        self.status.setText(f"Extraindo a cada {interval} frames...")

        frame_index = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % interval == 0:
                filename = os.path.join(output_folder, f"frame_{frame_index:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1

            frame_index += 1

        cap.release()
        self.status.setText(f"{saved} frames extraídos para '{output_folder}'.")
def get_widget():
    return VideoToImagesApp()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = VideoToImagesApp()
    window.show()
    sys.exit(app.exec_())