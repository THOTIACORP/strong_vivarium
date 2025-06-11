import sys
import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QScrollArea, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt

# Cores para as classes
colors = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

batch_size = 20

def load_mask_cv(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color
    # Convert BGR to RGB for Qt
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    return color_mask

def np_to_qpixmap(np_img):
    height, width, channel = np_img.shape
    bytes_per_line = 3 * width
    qimg = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MaskViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualização de Máscaras Discretas")
        self.setWindowIcon(QIcon("../../icon.ico"))
        self.setGeometry(100, 100, 900, 600)

        # Solicita ao usuário o diretório
        self.mask_dir = QFileDialog.getExistingDirectory(self, "Selecione a pasta com as máscaras")

        if not self.mask_dir:
            QMessageBox.warning(self, "Aviso", "Nenhuma pasta foi selecionada. O programa será encerrado.")
            sys.exit()

        # Lista os arquivos da pasta
        self.all_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])

        if not self.all_files:
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem PNG encontrada na pasta.")
            sys.exit()

        self.selected_file = self.all_files[0]
        self.offset = 0

        # Layout principal
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Label da imagem selecionada
        self.selected_label = QLabel("Imagem Selecionada")
        self.selected_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.selected_label)

        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_display)

        # Área de rolagem para miniaturas
        self.scroll = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scroll_widget)
        self.main_layout.addWidget(self.scroll)

        # Botões
        btn_layout = QHBoxLayout()
        self.load_more_btn = QPushButton("🔄 Carregar mais imagens")
        self.load_all_btn = QPushButton("📂 Carregar todas as imagens")
        btn_layout.addWidget(self.load_more_btn)
        btn_layout.addWidget(self.load_all_btn)
        self.main_layout.addLayout(btn_layout)

        self.load_more_btn.clicked.connect(self.load_more)
        self.load_all_btn.clicked.connect(self.load_all)
        self.choose_folder_btn = QPushButton("📁 Escolher outra pasta")
        btn_layout.addWidget(self.choose_folder_btn)
        self.choose_folder_btn.clicked.connect(self.choose_new_folder)


        # Atualiza exibição inicial
        self.update_display()
    def choose_new_folder(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Selecione uma nova pasta com as máscaras")
        if new_dir:
            self.mask_dir = new_dir
            self.all_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])
            if not self.all_files:
                QMessageBox.warning(self, "Aviso", "Nenhuma imagem PNG encontrada na nova pasta.")
                return
            self.selected_file = self.all_files[0]
            self.offset = 0
            self.update_display()
    def update_display(self):
        # Atualiza imagem grande
        img = load_mask_cv(os.path.join(self.mask_dir, self.selected_file))
        pixmap = np_to_qpixmap(img)
        scaled = pixmap.scaled(256, 256, Qt.KeepAspectRatio)
        self.image_display.setPixmap(scaled)

        # Remove miniaturas antigas
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        files_to_show = self.all_files[: self.offset + batch_size]
        thumbs_per_row = 5

        for idx, file in enumerate(files_to_show):
            thumb_img = load_mask_cv(os.path.join(self.mask_dir, file))
            thumb_pixmap = np_to_qpixmap(thumb_img).scaled(100, 100, Qt.KeepAspectRatio)

            thumb_label = QLabel()
            thumb_label.setPixmap(thumb_pixmap)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setObjectName(file)

            # Evento de clique na miniatura
            thumb_label.mousePressEvent = lambda event, f=file: self.on_thumbnail_click(f)

            row = idx // thumbs_per_row
            col = idx % thumbs_per_row
            self.scroll_layout.addWidget(thumb_label, row, col)

    def on_thumbnail_click(self, file):
        self.selected_file = file
        self.update_display()

    def load_more(self):
        if self.offset + batch_size < len(self.all_files):
            self.offset += batch_size
            self.update_display()

    def load_all(self):
        self.offset = len(self.all_files)
        self.update_display()

def get_widget():
    return MaskViewer()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskViewer()
    window.show()
    sys.exit(app.exec_())
