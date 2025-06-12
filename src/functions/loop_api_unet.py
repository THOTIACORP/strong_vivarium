import sys
import os
import base64
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QLineEdit, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class ImageProcessor(QThread):
    update_preview = pyqtSignal(str, dict)  # (image_path, response_json)
    update_status = pyqtSignal(str)

    def __init__(self, api_url, input_folder, output_folder):
        super().__init__()
        self.api_url = api_url
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self):
        extensoes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        os.makedirs(self.output_folder, exist_ok=True)
        arquivos = [f for f in os.listdir(self.input_folder)
                    if os.path.splitext(f)[1].lower() in extensoes]

        for arquivo in arquivos:
            caminho = os.path.join(self.input_folder, arquivo)
            self.update_status.emit(f"Enviando: {arquivo}")

            with open(caminho, "rb") as f:
                try:
                    resposta = requests.post(self.api_url, files={"image": (arquivo, f, "image/png")})
                    if resposta.status_code == 200:
                        self.update_preview.emit(caminho, resposta.json())
                    else:
                        self.update_status.emit(f"Erro na API: {resposta.status_code}")
                except Exception as e:
                    self.update_status.emit(f"Erro ao conectar: {e}")


class MaskSaverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UNet Preview - Thumbnail Viewer")
        self.resize(1200, 720)

        self.api_url = ""
        self.input_folder = ""
        self.output_folder = ""

        layout = QVBoxLayout()

        # Campos de entrada
        self.api_input = QLineEdit("http://localhost:8000/unet_freedom_track_image")
        self.api_input.setPlaceholderText("URL da API")

        self.input_label = QLabel("📁 Pasta de entrada: (não selecionada)")
        self.output_label = QLabel("💾 Pasta de saída: (não selecionada)")

        # Botões
        self.btn_select_input = QPushButton("Selecionar pasta de entrada")
        self.btn_select_output = QPushButton("Selecionar pasta de saída")
        self.btn_start = QPushButton("▶️ Iniciar")

        self.btn_select_input.clicked.connect(self.select_input_folder)
        self.btn_select_output.clicked.connect(self.select_output_folder)
        self.btn_start.clicked.connect(self.start_processing)

        # Labels de imagens
        self.label_sent = QLabel("Imagem enviada")
        self.label_original = QLabel("Imagem original (API)")
        self.label_mask = QLabel("Máscara gerada")

        for label in [self.label_sent, self.label_original, self.label_mask]:
            label.setFixedSize(256, 256)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black;")

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_sent)
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_mask)

        # JSON
        self.json_display = QTextEdit()
        self.json_display.setReadOnly(True)
        self.json_display.setPlaceholderText("Resposta da API...")

        layout.addWidget(QLabel("🔗 URL da API:"))
        layout.addWidget(self.api_input)
        layout.addWidget(self.input_label)
        layout.addWidget(self.btn_select_input)
        layout.addWidget(self.output_label)
        layout.addWidget(self.btn_select_output)
        layout.addWidget(self.btn_start)
        layout.addLayout(image_layout)
        layout.addWidget(QLabel("📄 JSON da resposta:"))
        layout.addWidget(self.json_display)

        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecione a pasta de entrada")
        if folder:
            self.input_folder = folder
            self.input_label.setText(f"📁 Pasta de entrada: {folder}")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecione a pasta de saída")
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"💾 Pasta de saída: {folder}")

    def start_processing(self):
        if not self.api_input.text() or not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "Atenção", "Preencha a URL da API e selecione ambas as pastas.")
            return

        self.thread = ImageProcessor(
            self.api_input.text(),
            self.input_folder,
            self.output_folder
        )
        self.thread.update_preview.connect(self.update_display)
        self.thread.update_status.connect(lambda msg: self.json_display.setPlainText(msg))
        self.thread.start()

    def update_display(self, image_path, json_data):
        # Mostrar a imagem enviada (do disco)
        try:
            pixmap_sent = QPixmap(image_path).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_sent.setPixmap(pixmap_sent)
        except Exception as e:
            self.label_sent.setText(f"Erro ao carregar imagem enviada:\n{e}")

        # Mostrar JSON
        self.json_display.setPlainText(str(json_data))

        # --------- Original recebida (da API via base64) ---------
        original_b64 = json_data.get("image", "")
        if original_b64.startswith("data:image"):
            original_b64 = original_b64.split(",", 1)[1]
        try:
            original_data = base64.b64decode(original_b64)
            original_img = QImage.fromData(original_data)
            original_pixmap = QPixmap.fromImage(original_img).scaled(
                256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_original.setPixmap(original_pixmap)
        except Exception as e:
            self.label_original.setText(f"Erro: {e}")

        # --------- Máscara ---------
        mask_b64 = json_data.get("mask", "")
        if mask_b64.startswith("data:image"):
            mask_b64 = mask_b64.split(",", 1)[1]
        try:
            img_data = base64.b64decode(mask_b64)
            image = QImage.fromData(img_data)
            pixmap = QPixmap.fromImage(image).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_mask.setPixmap(pixmap)

            # Salvar a máscara
            nome = os.path.basename(image_path)
            saida = os.path.join(self.output_folder, os.path.splitext(nome)[0] + "_unet.png")
            with open(saida, "wb") as f:
                f.write(img_data)
        except Exception as e:
            self.label_mask.setText(f"Erro: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskSaverApp()
    window.show()
    sys.exit(app.exec_())
